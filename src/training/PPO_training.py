from collections import defaultdict
import sys
import argparse
import pickle

sys.path.insert(0, '/root/causal_world/')
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    Compose,
    StepCounter,
    TransformedEnv,
    InitTracker,
    TensorDictPrimer,
)
from src.env.CausalEnv import CausalWorldEnv
from config.env_constants import *
from src.agents.controller_attention_through_with_mask import *
from torchrl.objectives import ClipPPOLoss
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type

from causal_world.task_generators.task import generate_task
from torchrl.objectives.value import GAE
from tqdm import tqdm
from matplotlib import pyplot as plt
from src.utils.helper_functions import *


# wrapper function to pass to the policy module
def planner_func(observation, hidden_states, cell_states):
    return planner_with_split(lazy_planner, observation, hidden_states, cell_states, model_params)


# helper function to get the hidden states specs
def get_hidden_states_specs(model_params):
    return UnboundedContinuousTensorSpec((2, 2, model_params['start_hidden_size']))


if __name__ == '__main__':

    # env params
    env_freq = 250
    frame_skip = 5
    action_mode = 'joint_positions'

    # training params
    clip_epsilon = (0.2)
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 2e-4
    lr = 1e-3
    max_grad_norm = 1.0

    ap = argparse.ArgumentParser()
    ap.add_argument("--task",
                    required=False,
                    default='pushing',
                    help="the task id")
    ap.add_argument("--model_levels",
                    required=False,
                    default=3,
                    help="the number of levels in the model")
    ap.add_argument("--model_start_hidden_size",
                    required=False,
                    default=512,
                    help="the number of hidden units in the first layer of the model, must be devisible by 2**(num_levels-1)")
    # ap.add_argument("--num_parallel_envs",
    #                 required=False,
    #                 default=16,
    #                 help="the number of parallel environments when collecting data. The number depends on the number of your cpu cores.")
    ap.add_argument("--sub-batch_size",
                    required=False,
                    default=25,
                    help="the sub batch size for training")
    ap.add_argument("--num_epochs",
                    required=False,
                    default=4,
                    help="the number of epochs for training")
    ap.add_argument("--total_frames",
                    required=False,
                    default=1e6,
                    help="the total number of frames to train")
    ap.add_argument("--frames_per_batch",
                    required=False,
                    default=500,
                    help="the number of frames per batch")
    args = vars(ap.parse_args())
    sub_batch_size = args['sub_batch_size']
    num_epochs = args['num_epochs']
    frames_per_batch = args['frames_per_batch']
    total_frames = args['total_frames']
    model_params = dict()
    model_params['num_levels'] = args['model_levels']
    model_params['start_hidden_size'] = args['model_start_hidden_size']
    task_id = args['task']
    # num_parallel_envs=args['num_parallel_envs']

    env = CausalWorldEnv(task_params=EnvConstants.TASK_PARAMS[task_id],
                         task=generate_task(task_generator_id=task_id), enable_visualization=False,
                         action_mode=action_mode)
    env = TransformedEnv(
        env,
        Compose(StepCounter(),
                InitTracker(),
                TensorDictPrimer(hidden_states=get_hidden_states_specs(model_params),
                                 cell_states=get_hidden_states_specs(model_params)),
                )
    )
    lazy_planner = LazyPlanner(num_levels=model_params['num_levels'],
                               start_hidden_size=model_params['start_hidden_size'],
                               task_params=EnvConstants.TASK_PARAMS['pushing'], device=device, dropout=0).to(device)

    policy_module = TensorDictModule(
        planner_func, in_keys=['observation', 'hidden_states', 'cell_states'],
        out_keys=['loc', 'scale', ('next', 'hidden_states'), ('next', 'cell_states')]
    )
    # Build distribution with the output of the planners
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=['loc', 'scale'],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
        },
        return_log_prob=True,
        safe=True,
    )

    # Build value function
    value_module = ValueNetwork(task_params=EnvConstants.TASK_PARAMS['pushing'], device=device).to(device)
    value_module = ValueOperator(
        module=value_module,
        in_keys=['observation'],
    )

    data_collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        value_target_key=advantage_module.value_target_key,
        critic_coef=1.0,
        gamma=0.99,
        loss_critic_type="smooth_l1",
    )

    replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    # Training

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # outer loop to collect batches of data
    for i, tensordict_data in enumerate(data_collector):
        # inner loop to perform epochs of training
        for _ in range(num_epochs):
            # calculate the advantages
            with torch.no_grad():
                advantage_module(tensordict_data)
            # flatten the batch_size dimension (note we may have several workers collecting data, resulting in a batch_size
            # of higher dimension)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            # for sub_batch, calculate the loss and backprop
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata)
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # restrict the size of the gradient
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        # logs["step_count"].append(tensordict_data["step_count"].max().item())
        # stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our env horizon).
            # The ``rollout`` method of the env can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        # pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    # save the model
    torch.save(policy_module.state_dict(),
               "positions,attention_through,512_3,1e6/pushing_policy_second_joint_positions.pt")
    with open(
            '../training_logs/pushing_policy_attention_through.pkl', 'wb') as f:
        pickle.dump(logs, f)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 1, 2)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.show()

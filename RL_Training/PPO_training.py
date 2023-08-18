from collections import defaultdict
import sys
import argparse

sys.path.insert(0, '/root/causal_world/')

import torch
from causal_world.envs import CausalWorld
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage, LazyMemmapStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    InitTracker,
    TensorDictPrimer,
)
from torchrl.envs.libs.gym import GymEnv
from Env.CausalEnv import CausalWorldEnv
from env_constants import *
from Models.hierarchical_controller import *
from torchrl.objectives import ClipPPOLoss
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from causal_world.task_generators.task import generate_task
from torchrl.objectives.value import GAE
from tqdm import tqdm
from matplotlib import pyplot as plt

device = "cpu" if not torch.has_cuda else "cuda:0"

# helper function to split tensor at given index
def split_2dtensor_on2nd(tensor, index):
    return tensor[:, :index], tensor[:, index:]


# helper function to split tensor at given index
def split_3dtensor_on3rd(tensor, index):
    return tensor[:, :, :index], tensor[:, :, index:]


# helper function to break down the hidden states into a list of hidden states and pass them to the planner,then
# process the next hidden states again into the required format
def planner_with_split(planner, observation, hidden_states, cell_states, model_params, device=device):
    # always add a batch_size dimension here. In the end remove the dimension again if the input has not batch_size
    # if no batch_size, add a batch_size dimension
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.unsqueeze(0)
        cell_states = cell_states.unsqueeze(0)
    if observation.dim() == 1:
        observation = observation.unsqueeze(0)
    # hidden states size (batch_size, 2 (layers at each level), 2, start_hidden_size)
    hidden_states = torch.flatten(hidden_states, start_dim=2)
    cell_states = torch.flatten(cell_states, start_dim=2)
    hidden_states_list = []
    cell_states_list = []
    # move observation to device
    observation = observation.to(device)
    for i in range(model_params['num_levels']):
        hidden_state, hidden_states = split_3dtensor_on3rd(hidden_states, model_params['start_hidden_size'] // (2 ** i))
        cell_state, cell_states = split_3dtensor_on3rd(cell_states, model_params['start_hidden_size'] // (2 ** i))
        hidden_states_list.append(hidden_state)
        cell_states_list.append(cell_state)
    # the lists passed t the planner: (num_levels, batch_size, 2 (layers), hidden_size_at_level)
    loc, scale, new_hidden_states, new_cell_states = planner(observation, hidden_states_list, cell_states_list)
    # returned hidden_states_lists: (num_levels, batch_size, 2, hidden_size_at_level)
    # the first rows: shape (batch_size, 2, start_hidden_size)
    new_hidden_states_first_row = new_hidden_states[0]
    new_cell_states_first_row = new_cell_states[0]
    # second rows: shape (batch_size, 2, start_hidden_size-difference)
    new_hidden_states_second_row = torch.cat(new_hidden_states[1:], dim=2)
    new_cell_states_second_row = torch.cat(new_cell_states[1:], dim=2)
    difference = model_params['start_hidden_size'] - new_hidden_states_second_row.shape[2]
    new_hidden_states_second_row = torch.cat(
        (new_hidden_states_second_row, torch.zeros(new_hidden_states_second_row.shape[0], 2, difference, device=device)), dim=2)
    new_cell_states_second_row = torch.cat(
        (new_cell_states_second_row, torch.zeros(new_cell_states_second_row.shape[0], 2, difference, device=device)), dim=2)
    new_hidden_states = torch.stack((new_hidden_states_first_row, new_hidden_states_second_row), dim=1)
    new_cell_states = torch.stack((new_cell_states_first_row, new_cell_states_second_row), dim=1)
    # if batch_size was 1, remove the batch_size dimension
    if new_hidden_states.shape[0] == 1:
        new_hidden_states = new_hidden_states.squeeze(0)
        new_cell_states = new_cell_states.squeeze(0)
        loc = loc.squeeze(0)
        scale = scale.squeeze(0)
    return loc, scale, new_hidden_states, new_cell_states

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
    action_mode = 'joint_torques'


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
                    default=6,
                    help="the number of levels in the model")
    ap.add_argument("--model_start_hidden_size",
                    required=False,
                    default=4096,
                    help="the number of hidden units in the first layer of the model, must be devisible by 2**(num_levels-1)")
    ap.add_argument("--num_parallel_envs",
                    required=False,
                    default=16,
                    help="the number of parallel environments when collecting data. The number depends on the number of your cpu cores.")
    ap.add_argument("--sub-batch_size",
                    required=False,
                    default=256,
                    help="the sub batch size for training")
    ap.add_argument("--num_epochs",
                    required=False,
                    default=6,
                    help="the number of epochs for training")
    ap.add_argument("--total_frames",
                    required=False,
                    default=1e7,
                    help="the total number of frames to train")
    ap.add_argument("--frames_per_batch",
                    required=False,
                    default=10000,
                    help="the number of frames per batch")
    args = vars(ap.parse_args())
    sub_batch_size=args['sub_batch_size']
    num_epochs=args['num_epochs']
    frames_per_batch=args['frames_per_batch']
    total_frames=args['total_frames']
    model_params=dict()
    model_params['num_levels']=args['model_levels']
    model_params['start_hidden_size']=args['model_start_hidden_size']
    task_id=args['task']
    num_parallel_envs=args['num_parallel_envs']



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
    lazy_planner = LazyPlanner(num_levels=model_params['num_levels'], start_hidden_size=model_params['start_hidden_size'],
                               task_params=EnvConstants.TASK_PARAMS['pushing'], device=device).to(device)




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
    pbar = tqdm(total=total_frames * frame_skip)
    eval_str = ""

    # outer loop to collect batches of data
    for i, tensordict_data in enumerate(data_collector):
        # inner loop to perform epoches of training
        for _ in range(num_epochs):
            # calculate the advantages
            with torch.no_grad():
                advantage_module(tensordict_data)
            # flatten the batch_size dimension (note we may have several workers collecting data, resulting in a batch_size
            # of higher dimension)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
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
            pbar.update(tensordict_data.numel() * frame_skip)
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
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
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()

    # save the model
    torch.save(policy_module.state_dict(), "pushing_policy_second_joint_positions.pt")
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()

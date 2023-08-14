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

from causal_world.task_generators.task import generate_task
from torchrl.objectives.value import GAE

device = "cpu" if not torch.has_cuda else "cuda:0"

# env params
env_freq = 250
frame_skip = 5
action_mode = 'joint_torques'

#training params
sub_batch_size = 64
num_epochs = 10
clip_epsilon = (0.2)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-3
frames_per_batch = 1000
total_frames = 1e6
lr = 2e-4
max_grad_norm = 1.0

# model params
model_params = {
    'num_levels': 5,
    'start_hidden_size': 512,
}

# helper function to split tensor at given index
def split_2dtensor_on2nd(tensor, index):
    return tensor[:, :index], tensor[:, index:]

# helper function to break down the hidden states into a list of hidden states and pass them to the planner,then
# process the next hidden states again into the required format
def planner_with_split(planner, observation, hidden_states, cell_states ,model_params):
    # we do not consider batch size here;
    # hidden_states size (2 (layers at each level), 2, start_hidden_size)
    hidden_states = torch.flatten(hidden_states, start_dim=1)
    cell_states = torch.flatten(cell_states, start_dim=1)
    hidden_states_list = []
    cell_states_list = []
    for i in range(model_params['num_levels']):
        hidden_state, hidden_states = split_2dtensor_on2nd(hidden_states, model_params['start_hidden_size']//(2**i))
        cell_state, cell_states = split_2dtensor_on2nd(cell_states, model_params['start_hidden_size']//(2**i))
        hidden_states_list.append(hidden_state)
        cell_states_list.append(cell_state)
    loc, scale, new_hidden_states, new_cell_states = planner(observation, hidden_states_list, cell_states_list)
    # returned hidden_states_lists: (num_levels, 2, start_hidden_size//(2**i))
    # the first rows: shape (2, start_hidden_size)
    new_hidden_states_first_row = new_hidden_states[0]
    new_cell_states_first_row = new_cell_states[0]
    # second rows: shape (2, start_hidden_size-difference)
    new_hidden_states_second_row = torch.cat(new_hidden_states[1:], dim=1)
    new_cell_states_second_row = torch.cat(new_cell_states[1:], dim=1)
    difference = model_params['start_hidden_size'] - new_hidden_states_second_row.shape[1]
    new_hidden_states_second_row = torch.cat((new_hidden_states_second_row, torch.zeros(2, difference)), dim=1)
    new_cell_states_second_row = torch.cat((new_cell_states_second_row, torch.zeros(2, difference)), dim=1)
    new_hidden_states = torch.stack((new_hidden_states_first_row, new_hidden_states_second_row), dim=0)
    new_cell_states = torch.stack((new_cell_states_first_row, new_cell_states_second_row), dim=0)
    return loc, scale, new_hidden_states, new_cell_states

# helper function to get the hidden states specs
def get_hidden_states_specs(model_params):
    return UnboundedContinuousTensorSpec((2, 2, model_params['start_hidden_size']))

pushing_env = CausalWorldEnv(task_params= EnvConstants.TASK_PARAMS['pushing'], task=generate_task(task_generator_id='pushing'), enable_visualization=False,
                             action_mode=action_mode)
pushing_env = TransformedEnv(
    pushing_env,
    Compose(StepCounter(),
            InitTracker(),
            TensorDictPrimer(hidden_states=get_hidden_states_specs(model_params), cell_states=get_hidden_states_specs(model_params)),)
)
lazy_planner = LazyPlanner(num_levels=model_params['num_levels'], start_hidden_size=model_params['start_hidden_size'], task_params=EnvConstants.TASK_PARAMS['pushing'])
# wrapper function to pass to the policy module
def planner_func(observation, hidden_states, cell_states):
    return planner_with_split(lazy_planner, observation, hidden_states, cell_states, model_params)
policy_module = TensorDictModule(
    planner_func, in_keys=['observation', 'hidden_states', 'cell_states'], out_keys=['loc', 'scale', ('next', 'hidden_states'), ('next', 'cell_states')]
)
# Build distribution with the output of the planners
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=pushing_env.action_spec,
    in_keys=['loc', 'scale'],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": pushing_env.action_spec.space.minimum,
        "max": pushing_env.action_spec.space.maximum,
    },
    return_log_prob=True,
    safe=True,
)

# Build value function
value_module = ValueNetwork()
value_module = ValueOperator(
    module=value_module,
    in_keys=['observation'],
)

data_collector = SyncDataCollector(
    pushing_env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=device,
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
    storage=LazyMemmapStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)


optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

# Training
# outer loop to collect batches of data
for i, tensordict_data in enumerate(data_collector):
    # inner loop to perform epoches of training
    for _ in range(num_epochs):
        # calculate the advantages
        with torch.no_grad():
            advantage_module(tensordict_data)
        # flatten the batch_size dimension (note we may have several workers collecting data, resulting in a batch_size
        # of higher dimension)
        data_view=tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        # for sub_batch, calculate the loss and backprop
        for _ in range(frames_per_batch//sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()


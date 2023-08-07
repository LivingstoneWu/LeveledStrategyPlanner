import torch
from causal_world.envs import CausalWorld
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from ..Env.CausalEnv import CausalWorldEnv


from causal_world.task_generators.task import generate_task




device = "cpu" if not torch.has_cuda else "cuda:0"

# env params
env_freq = 250
frame_skip = 5
action_mode = 'joint_torques'
frames_per_batch = 1000

sub_batch_size= 64
num_epochs = 10
clip_epsilon = (0.2)
gamma = 0.99
lam = 0.95
entropy_eps = 1e-3


pushing_env = CausalWorld(task=generate_task(task_generator_id='pushing'), enable_visualization=False, action_mode='joint_torques')
base_env = GymEnv("Pendulum-v0", device=device, frame_skip=frame_skip)
env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(
            in_keys=["observation"],
        ),
        StepCounter(),
    ),
)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print(env.rollout(3))
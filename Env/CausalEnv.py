import torch
from causal_world.envs import CausalWorld
from causal_world.task_generators.task import generate_task
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
import numpy as np
from torch import Tensor
from Models import hierarchical_controller

# this part rewrites the CausalWorld Env to fit into torchRL

class CausalWorldEnv(EnvBase):
    def __init__(self, observation_params, task_params, **kwargs):
        super(CausalWorldEnv, self).__init__()
        self.env = CausalWorld(**kwargs)
        self.action_space = self.env.action_space
        self.task_params = task_params


        self.action_spec = BoundedTensorSpec(self.action_space.low, self.action_space.high, dtype=torch.float64)
        observation_length = observation_params['observation_full_length']
        obs_spec = UnboundedContinuousTensorSpec(shape=observation_length, dtype=torch.float64)
        self.observation_spec = CompositeSpec(observation=obs_spec)
        self.reward_spec = BoundedTensorSpec(np.array([0]), np.array([1]), dtype=torch.float64)


    def _reset(self, tensordict, **kwargs):
        observation = self.env.reset()
        observation = hierarchical_controller.padding_observation(torch.from_numpy(observation), self.task_params)
        out_tensordict = TensorDict({'observation': observation}, batch_size=torch.Size())
        return out_tensordict

    def _step(self, tensordict):
        action = tensordict['action']
        observation, reward, done, info = self.env.step(Tensor.numpy(action))
        observation = hierarchical_controller.padding_observation(torch.from_numpy(observation), self.task_params)
        out_tensordict = TensorDict(
            {
                "next":{
                        "observation": observation,
                        "reward": reward,
                        "done": done,
                }
            },
            batch_size=torch.Size(),
        )
        return out_tensordict

    def _set_seed(self, seed):
        self.env.seed(seed)
import torch
from causal_world.envs import CausalWorld
from causal_world.wrappers import CurriculumWrapper
from causal_world.intervention_actors import GoalInterventionActorPolicy, RandomInterventionActorPolicy

from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
import numpy as np
from torch import Tensor
from torchrl.envs import EnvBase

from Models import hierarchical_controller


# this part rewrites the CausalWorld Env to fit into torchRL
# Note, the environment observation is padded here, no more padding needed to feed into the model

class CausalWorldEnv(EnvBase):
    def __init__(self, task_params, observation_params=hierarchical_controller.observation_params, **kwargs):
        super(CausalWorldEnv, self).__init__()
        self.env = CausalWorld(**kwargs)
        self.action_space = self.env.action_space
        self.task_params = task_params

        self.action_spec = BoundedTensorSpec(self.action_space.low, self.action_space.high, dtype=torch.float32)
        observation_length = 1 + 3 * observation_params['joint_length'] +\
                            task_params['num_objects'] * observation_params['object_length'] +\
                            task_params['num_goals'] * observation_params['goal_length'] +\
                            task_params['num_blocks'] * observation_params['block_length']
        obs_spec = UnboundedContinuousTensorSpec(shape=observation_length, dtype=torch.float32)
        self.observation_spec = CompositeSpec(observation=obs_spec)
        self.reward_spec = BoundedTensorSpec(np.array([0]), np.array([1]), dtype=torch.float32)
        self.done_spec = DiscreteTensorSpec(2, shape=torch.Size((1,)), dtype=torch.bool)


    def _reset(self, tensordict, **kwargs):
        observation = self.env.reset()
        out_tensordict = TensorDict({'observation': torch.from_numpy(observation).float()}, batch_size=torch.Size())
        return out_tensordict


    def _step(self, tensordict):
        action = tensordict['action']
        # need unsqueeze the batch_size dimension here. Check the training code though
        action = action.squeeze(0)
        observation, reward, done, info = self.env.step(Tensor.numpy(action.cpu()))
        out_tensordict = TensorDict(
            {
                "next":{
                        "observation": torch.from_numpy(observation).float(),
                        "reward": torch.from_numpy(np.array([reward])).float(),
                        "done": done,
                }
            },
            batch_size=torch.Size(),
        )
        return out_tensordict

    def _set_seed(self, seed):
        self.env.seed(seed)


class CurriculumEnv(EnvBase):
    """
    CausalWorld Environments that perform interventions on an episodic basis.
    Params:
        task: note the intervention_space must be correctly configured when generating the task.
    """
    def __init__(self, task_params, intervention_actors=(RandomInterventionActorPolicy(), GoalInterventionActorPolicy()), intervention_actives=((0, 1000000000, 1, 0), (0, 1000000000, 1, 0)), observation_params=hierarchical_controller.observation_params, **kwargs):
        self.env = CausalWorld(**kwargs)
        self.env = CurriculumWrapper(self.env, intervention_actors, intervention_actives)
        self.action_space = self.env.action_space
        self.task_params = task_params
        self.action_spec = BoundedTensorSpec(self.action_space.low, self.action_space.high, dtype=torch.float32)
        observation_length = 1 + 3 * observation_params['joint_length'] + \
                             task_params['num_objects'] * observation_params['object_length'] + \
                             task_params['num_goals'] * observation_params['goal_length'] + \
                             task_params['num_blocks'] * observation_params['block_length']
        obs_spec = UnboundedContinuousTensorSpec(shape=observation_length, dtype=torch.float32)
        self.observation_spec = CompositeSpec(observation=obs_spec)
        self.reward_spec = BoundedTensorSpec(np.array([0]), np.array([1]), dtype=torch.float32)
        self.done_spec = DiscreteTensorSpec(2, shape=torch.Size((1,)), dtype=torch.bool)

    def _reset(self, tensordict, **kwargs):
        observation = self.env.reset()
        out_tensordict = TensorDict({'observation': torch.from_numpy(observation).float()}, batch_size=torch.Size())
        return out_tensordict

    def _step(self, tensordict):
        action = tensordict['action']
        # need unsqueeze the batch_size dimension here. Check the training code though
        action = action.squeeze(0)
        observation, reward, done, info = self.env.step(Tensor.numpy(action.cpu()))
        out_tensordict = TensorDict(
            {
                "next": {
                    "observation": torch.from_numpy(observation).float(),
                    "reward": torch.from_numpy(np.array([reward])).float(),
                    "done": done,
                }
            },
            batch_size=torch.Size(),
        )
        return out_tensordict

    def _set_seed(self, seed):
        self.env.seed(seed)



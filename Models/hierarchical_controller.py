import torch
import torch.nn as nn
from .. import env_constants

device = "cpu" if not torch.has_cuda else "cuda:0"

# env params
frame_skip = 5
action_mode = 'joint_torques'
frames_per_batch = 1000


# observation params specifies how many variable objects are there in the environment.
class AttentionSubModule(nn.Module):
    def __init__(self, observation_constants, observation_params, is_hidden=False):
        super(AttentionSubModule, self).__init__()
        self.observation_constants = observation_constants
        self.


# A planner contains attention
class PlannerModule(nn.Module):
    def __init__(self,


class LeveledStrategy(nn.Module):
    def __init__(self, num_levels, num_actions, num_observations):
        super(LeveledStrategy, self).__init__()
        self.num_levels = num_levels
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.level = 0
        self.levels = nn.ModuleList([nn.Sequential(nn.Linear(num_observations, num_actions), nn.Sigmoid()) for _ in range(num_levels)])

    def forward(self, x):
        return self.levels[self.level](x)

# sanity check, for reaching task
class LazyPlannerModule(nn.Module):
    def __init__(self, observation_constants):
        
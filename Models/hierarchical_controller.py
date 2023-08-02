import torch
import torch.nn as nn
from .. import env_constants

device = "cpu" if not torch.has_cuda else "cuda:0"

# env params
env_freq = 250
frame_skip = 5
action_mode = 'joint_torques'
frames_per_batch = 1000
observation_constants = env_constants.EnvConstants.OBSERVATION_INDICES


class PredictorSubModule(nn.Module):
    def __init__(self, time_scale, observation_params, hidden_size):
        super(PredictorSubModule, self).__init__()


# observation params specifies how many variable objects are there in the environment.
# "is_hidden" specifies whether the planner layer is hidden (i.e. whether the planner layer has a predecessor)
class AttentionSubModule(nn.Module):
    def __init__(self, observation_params, observation_constants=observation_constants, is_hidden=False, dropout=0.1, hidden_size=128):
        super(AttentionSubModule, self).__init__()
        self.observation_constants = observation_constants
        self._is_hidden = is_hidden
        self.hidden_size=hidden_size
        self._observation_params = observation_params
        self.num_objects = observation_params['num_objects']
        self.num_blocks = observation_params['num_blocks']
        self.num_goals = observation_params['num_goals']
        # computing sizes of variables
        obs_vector_size = 27
        self.obj_vector_size = 17
        self.block_vector_size = 11
        self.goal_vector_size = 11
        query_key_size = 27
        value_size = 20
        # indices
        self.obj_start_index = observation_constants['end_effector_positions'][1]
        self.goal_start_index = self.obj_start_index + self.num_objects * self.obj_vector_size
        self.block_start_index = self.goal_start_index + self.num_goals * self.goal_vector_size
        self.hidden_start_index = self.block_start_index + self.num_blocks * self.block_vector_size
        # specifying layers, obs being the structured representation; obj being the objects representation
        self.obs_Linear = nn.Linear(obs_vector_size, query_key_size)
        self.obj_Linear = nn.Linear(self.obj_vector_size, query_key_size)
        self.block_Linear = nn.Linear(self.block_vector_size, query_key_size)
        self.goal_Linear = nn.Linear(self.goal_vector_size, query_key_size)
        self.obs_value = nn.Linear(obs_vector_size, value_size)
        self.obj_value = nn.Linear(self.obj_vector_size, value_size)
        self.block_value = nn.Linear(self.block_vector_size, value_size)
        self.goal_value = nn.Linear(self.goal_vector_size, value_size)
        self.dropout = nn.Dropout(dropout)
        # the FC layers before the output
        self.out1 = nn.Linear(value_size, 3 * value_size)
        self.out2 = nn.Linear(3 * value_size, 3 * value_size)
        self.out3 = nn.Linear(3 * value_size, value_size)

    # input is the observation vector, and the hidden vector if is_hidden==True, otherwise padding of 0s
    def forward(self, x):
        # compute the query and key vectors
        obs_key = self.obs_Linear(x[self.observation_constants['joint_positions'][0]:self.observation_constants['end_effector_positions'][1]])
        obj_keys = self.obj_Linear(torch.reshape(x[self.obj_start_index:self.goal_start_index], (self.num_objects, self.obj_vector_size)))
        goal_keys = self.goal_Linear(torch.reshape(x[self.goal_start_index:self.block_start_index], (self.num_goals, self.goal_vector_size)))
        if self.num_blocks!=0:
            block_key = self.block_Linear(torch.reshape(x[self.block_start_index:self.hidden_start_index], (self.num_blocks, self.block_vector_size)))
        else:
            block_key = torch.zeros(1, self.block_vector_size, device=device)




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
#
# # # sanity check, for reaching task
# # class LazyPlannerModule(nn.Module):
#     def __init__(self):
#
#
# class LazyAttentionSubModule(nn.Module):
#     def __init__(self, observation_constants, pos_length=3):
#         super(LazyAttentionSubModule, self).__init__()
#         self.time_left_idx = observation_constants['time_left']
#         self.joint_positions_idx = observation_constants['joint_positions']
#         self.joint_velocities_idx = observation_constants['joint_velocities']
#         self.end_effector_positions_idx = observation_constants['end_effector_positions']
#         self.goal_positions_idx = observation_constants['goal_positions']
#
#     @staticmethod
#     def dotprod_scoring(Q, K):
#         if Q.shape != K.shape:
#             raise ValueError("q and k must have the same shape")
#         return torch.matmul(Q, K.transpose(0, 1))
#
#     def forward(self, x):
#         time_left = x[self.time_left_idx[0]:self.time_left_idx[1]]
#         joint_positions = x[self.joint_positions_idx[0]:self.joint_positions_idx[1]]
#         joint_velocities = x[self.joint_velocities_idx[0]:self.joint_velocities_idx[1]]
#         end_effector_positions = x[self.end_effector_positions_idx[0]:self.end_effector_positions_idx[1]]
#         goal_positions = x[self.goal_positions_idx[0]:self.goal_positions_idx[1]]

import torch
import torch.nn as nn
from config import env_constants
import torch.nn.functional as F
import math

observation_constants = env_constants.EnvConstants.OBSERVATION_INDICES

# model params
# here we define the size for each group of variables, padded if less than the amount defined for every observation.
num_joints = 3
num_objects = 10
num_blocks = 2
num_goals = 10
len_joints = 27
len_object = 17
len_block = 11
len_goal = 11
object_start_index = len_joints
goal_start_index = len_joints + num_objects * len_object
block_start_index = goal_start_index + num_goals * len_goal
block_end_index = block_start_index + num_blocks * len_block
DEFAULT_KEY_VALUE_SIZE = 9

# params of the preprocessed observation vector (joints rearranged and other variables padded)
observation_params = {

    'joint_indices': [0, len_joints],
    'num_joints': num_joints,
    'joint_length': len_joints // num_joints,

    'object_indices': [object_start_index, goal_start_index],
    'num_objects': num_objects,
    'object_length': len_object,

    'goal_indices': [goal_start_index, block_start_index],
    'num_goals': num_goals,
    'goal_length': len_goal,

    'block_indices': [block_start_index, block_end_index],
    'num_blocks': num_blocks,
    'block_length': len_block,

    'num_queries': num_joints + num_objects + num_goals + num_blocks,
    'attention_length': DEFAULT_KEY_VALUE_SIZE * (num_joints + num_objects + num_goals + num_blocks),
    'observation_full_length': len_joints + num_objects * len_object + num_goals * len_goal + num_blocks * len_block,

    # define the output size
    'action_size': 9,
}



# helper function to slice a 2d tensor with a list
def slice_indices(tensor, indices):
    return tensor[:, indices[0]:indices[1]]


def padding_observation(observation, task_params, device, observation_params=observation_params, time_left_length=1):
    """
    Input: observation ndarray, (batch_size, observation_length)
    Output: padded observation tensor
    Params: task_params specify the number of each objects in the task
    """

    # calculate the indices for slicing first
    joint_indices = [x + time_left_length for x in observation_params['joint_indices']]
    object_start_index = joint_indices[1]
    goal_start_index = object_start_index + task_params['num_objects'] * observation_params['object_length']
    block_start_index = goal_start_index + task_params['num_goals'] * observation_params['goal_length']
    block_end_index = block_start_index + task_params['num_blocks'] * observation_params['block_length']

    # slice the observation tensor
    joint_observation = slice_indices(observation, joint_indices)
    object_observation = slice_indices(observation, [object_start_index, goal_start_index])
    goal_observation = slice_indices(observation, [goal_start_index, block_start_index])
    block_observation = slice_indices(observation, [block_start_index, block_end_index])

    # pad the observation tensor
    pad_obj = torch.cat((object_observation, torch.zeros(observation.size()[0], (
                observation_params['num_objects'] - task_params['num_objects']) * observation_params['object_length'], device=device)),
                        dim=1)
    pad_goal = torch.cat((goal_observation, torch.zeros(observation.size()[0],
                                                        (observation_params['num_goals'] - task_params['num_goals']) *
                                                        observation_params['goal_length'], device=device)), dim=1)
    pad_block = torch.cat((block_observation, torch.zeros(observation.size()[0], (
                observation_params['num_blocks'] - task_params['num_blocks']) * observation_params['block_length'], device=device)),
                          dim=1)

    # concatenate the padded observation tensor
    padded_observation = torch.cat((joint_observation, pad_obj, pad_goal, pad_block), dim=1)

    return padded_observation


class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    # queries: (batch_size, num_queries, query_key_size)
    # keys: (batch_size, num_keys, query_key_size)
    # values: (batch_size, num_keys, value_size)
    def forward(self, queries, keys, values):
        # compute the key size
        key_size = queries.size(-1)
        # (batch_size, num_queries, num_keys)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(key_size)
        # (batch_size, num_queries, num_keys)
        attention_weights = torch.softmax(scores, dim=-1)
        # (batch_size, num_queries, value_size)
        res = torch.bmm(self.dropout(attention_weights), values)
        return res


class AttentionSubModule(nn.Module):
    """
    Input: padded observation
    Output: attention pooling results, 25 (num_queries) * 9 (key_value_size) = 225 Note that the output is not flattened as the predictor module may use that.
    Params:
        observation_params: indices of start-end of each group of variables, num_queries

    """

    def __init__(self, observation_params=observation_params, key_value_size=DEFAULT_KEY_VALUE_SIZE, dropout=0.1):
        super(AttentionSubModule, self).__init__()
        # indices, parameters
        self.key_value_size = key_value_size
        self.observation_params = observation_params
        self.num_queries = observation_params['num_queries']
        self.num_joints = observation_params['num_joints']
        self.num_objects = observation_params['num_objects']
        self.num_goals = observation_params['num_goals']
        self.num_blocks = observation_params['num_blocks']
        self.joint_length = observation_params['joint_length']
        self.object_length = observation_params['object_length']
        self.goal_length = observation_params['goal_length']
        self.block_length = observation_params['block_length']

        # key & query layers
        self.joint_Key = nn.Linear(self.observation_params['joint_length'], self.key_value_size)
        self.object_Key = nn.Linear(self.observation_params['object_length'], self.key_value_size)
        self.goal_Key = nn.Linear(self.observation_params['goal_length'], self.key_value_size)
        self.block_Key = nn.Linear(self.observation_params['block_length'], self.key_value_size)

        # value layers
        self.joint_Value = nn.Linear(self.observation_params['joint_length'], self.key_value_size)
        self.object_Value = nn.Linear(self.observation_params['object_length'], self.key_value_size)
        self.goal_Value = nn.Linear(self.observation_params['goal_length'], self.key_value_size)
        self.block_Value = nn.Linear(self.observation_params['block_length'], self.key_value_size)

        # residual layers that transform the input to the same dimension as the output
        self.joint_Residual = nn.Linear(self.observation_params['joint_length'], self.key_value_size)
        self.object_Residual = nn.Linear(self.observation_params['object_length'], self.key_value_size)
        self.goal_Residual = nn.Linear(self.observation_params['goal_length'], self.key_value_size)
        self.block_Residual = nn.Linear(self.observation_params['block_length'], self.key_value_size)

        # attention layers
        self.Attention = DotProductAttention(dropout=dropout)

        # layer normalization
        self.LayerNorm = nn.LayerNorm(self.key_value_size)

    # Input: (batch_size, observation_size)
    def forward(self, x):
        joint_keys = self.joint_Key(
            slice_indices(x, self.observation_params['joint_indices']).view(-1, self.num_joints, self.joint_length))
        object_keys = self.object_Key(
            slice_indices(x, self.observation_params['object_indices']).view(-1, self.num_objects, self.object_length))
        goal_keys = self.goal_Key(
            slice_indices(x, self.observation_params['goal_indices']).view(-1, self.num_goals, self.goal_length))
        block_keys = self.block_Key(
            slice_indices(x, self.observation_params['block_indices']).view(-1, self.num_blocks, self.block_length))

        # queries & keys of the shape (batch_size, num_queries, key_value_size)
        keys = torch.cat([joint_keys, object_keys, goal_keys, block_keys], dim=1)

        joint_values = self.joint_Value(
            slice_indices(x, self.observation_params['joint_indices']).view(-1, self.num_joints, self.joint_length))
        object_values = self.object_Value(
            slice_indices(x, self.observation_params['object_indices']).view(-1, self.num_objects, self.object_length))
        goal_values = self.goal_Value(
            slice_indices(x, self.observation_params['goal_indices']).view(-1, self.num_goals, self.goal_length))
        block_values = self.block_Value(
            slice_indices(x, self.observation_params['block_indices']).view(-1, self.num_blocks, self.block_length))

        # values of the shape (batch_size, num_queries, key_value_size)
        values = torch.cat([joint_values, object_values, goal_values, block_values], dim=1)

        # transform the input to the same dimension as the output
        joint_residuals = self.joint_Residual(
            slice_indices(x, self.observation_params['joint_indices']).view(-1, self.num_joints, self.joint_length))
        object_residuals = self.object_Residual(
            slice_indices(x, self.observation_params['object_indices']).view(-1, self.num_objects, self.object_length))
        goal_residuals = self.goal_Residual(
            slice_indices(x, self.observation_params['goal_indices']).view(-1, self.num_goals, self.goal_length))
        block_residuals = self.block_Residual(
            slice_indices(x, self.observation_params['block_indices']).view(-1, self.num_blocks, self.block_length))

        # residuals of the shape (batch_size, num_queries, key_value_size)
        residuals = torch.cat([joint_residuals, object_residuals, goal_residuals, block_residuals], dim=1)

        # compute attention, size (batch_size, num_queries, key_value_size), and add residual connection
        x = self.Attention(keys, keys, values) + residuals

        # layer normalization
        x = self.LayerNorm(x)
        return x


class LazyPlannerModule(nn.Module):
    """
    LazyPlanner do not have predictor module. this class will always be hidden planners.
    Input: observation + prev_hidden_state (plan), lstm_hidden_state, lstm_cell_state
    Output: hidden_state (plan) for the next level
    """

    def __init__(self, hidden_size, dropout=0.1, observation_params=observation_params):
        super(LazyPlannerModule, self).__init__()
        self.hidden_size = hidden_size
        self.attention = AttentionSubModule(dropout=dropout)
        # last layer's hidden size is 2 * of this layer
        self.lstm = nn.LSTM(input_size=hidden_size * 2 + observation_params['attention_length'], num_layers=2,
                            hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.residual_connection = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.observation_params = observation_params

    def forward(self, x, current_hidden_state, current_cell_state):
        # x: (batch_size, obsevation_size + hidden_size)
        attention = self.attention(x[:, :self.observation_params['observation_full_length']])
        # unsqueeze to add seq_len dimension
        plan = x[:, self.observation_params['observation_full_length']:].unsqueeze(1)
        attention_flatten = torch.flatten(attention, start_dim=1).unsqueeze(1)
        current_hidden_state = current_hidden_state.permute(1, 0, 2).contiguous()
        current_cell_state = current_cell_state.permute(1, 0, 2).contiguous()
        lstm_output, (next_hidden_state, next_cell_state) = self.lstm(torch.cat((plan, attention_flatten), dim=2),
                                                                      (current_hidden_state, current_cell_state))
        next_hidden_state = next_hidden_state.permute(1, 0, 2)
        next_cell_state = next_cell_state.permute(1, 0, 2)
        # residual connection and layer normalization
        x = self.residual_connection(x[:, self.observation_params['observation_full_length']:]) + self.layer_norm(
            lstm_output.squeeze(1))
        return x, next_hidden_state, next_cell_state


class LazyPlannerStarter(nn.Module):
    """
    The first layer of the lazy planner.
    Input: observation, lstm_hidden_state, lstm_cell_state
    Output: hidden_state (plan) for the next level
    """

    def __init__(self, hidden_size, dropout=0.1, observation_params=observation_params):
        super(LazyPlannerStarter, self).__init__()
        self.hidden_size = hidden_size
        self.attention = AttentionSubModule(dropout=dropout)
        # last layer's hidden size is 2 * of this layer
        self.lstm = nn.LSTM(input_size=observation_params['attention_length'], num_layers=2,
                            hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.observation_params = observation_params
        self.residual_connection = nn.Linear(observation_params['observation_full_length'], hidden_size)

    def forward(self, x, current_hidden_state, current_cell_state):
        # x: (batch_size, obsevation_size)
        attention = self.attention(x[:, :self.observation_params['observation_full_length']])
        attention_flatten = torch.flatten(attention, start_dim=1).unsqueeze(1)
        # hidden_state: (batch_size, num_layers, hidden_size)
        current_hidden_state = current_hidden_state.permute(1, 0, 2).contiguous()
        current_cell_state = current_cell_state.permute(1, 0, 2).contiguous()

        lstm_output, (next_hidden_state, next_cell_state) = self.lstm(attention_flatten,
                                                                      (current_hidden_state, current_cell_state))
        next_hidden_state = next_hidden_state.permute(1, 0, 2)
        next_cell_state = next_cell_state.permute(1, 0, 2)

        # residual connection and layer normalization
        x = self.residual_connection(x[:, :self.observation_params['observation_full_length']]) + self.layer_norm(
            lstm_output.squeeze(1))
        return x, next_hidden_state, next_cell_state


class LazyPlanner(nn.Module):
    """
    The LazyPlanner containing multiple LazyPlannerModules.
    Input: padded observation, lstm hidden states
    Output: the actions of size (batch_size, action_size), lstm hidden states
    Params: num_levels, start_hidden_size, dropout
         Note the start hidden size should be divisible by 2^(num_levels-1)
         The num_levels counts the number of planners, excluding the final FC layers.
    """

    def __init__(self, num_levels, start_hidden_size, task_params, device, dropout=0.1,
                 observation_params=observation_params):
        super(LazyPlanner, self).__init__()
        self.num_levels = num_levels
        self.start_hidden_size = start_hidden_size
        self.dropout = dropout
        self.device = device
        self.observation_params = observation_params
        # define layers of planners
        self.startPlannerLayer = LazyPlannerStarter(start_hidden_size, dropout=dropout,
                                                    observation_params=observation_params)
        self.plannerLayers = nn.ModuleList()
        current_hidden_size = start_hidden_size // 2
        for i in range(num_levels - 1):
            self.plannerLayers.append(
                LazyPlannerModule(current_hidden_size, dropout=dropout, observation_params=observation_params))
            current_hidden_size = current_hidden_size // 2
        # final FC layers
        self.fc1 = nn.Linear(current_hidden_size * 2, current_hidden_size * 4)
        self.fc2 = nn.Linear(current_hidden_size * 4, current_hidden_size * 4)
        self.output = nn.Linear(current_hidden_size * 4, observation_params['action_size'] * 2)
        self.task_params = task_params
        self.loc_activate = nn.Tanh()
        self.scale_activate = nn.Softplus()


    def forward(self, observation, hidden_states, cell_states):
        padded_observation = padding_observation(observation, self.task_params, device=self.device)
        new_hidden_states = []
        new_cell_states = []
        # start planner
        initial_plan, new_hidden_state, new_cell_state = self.startPlannerLayer(padded_observation, hidden_states[0],
                                                                                cell_states[0])
        new_hidden_states.append(new_hidden_state)
        new_cell_states.append(new_cell_state)
        # propagate through planners
        current_plan = initial_plan
        for i in range(self.num_levels - 1):
            current_plan, new_hidden_state, new_cell_state = self.plannerLayers[i](
                torch.cat((padded_observation, current_plan), dim=1), hidden_states[i + 1], cell_states[i + 1])
            new_hidden_states.append(new_hidden_state)
            new_cell_states.append(new_cell_state)
        # final FC layers
        x = F.relu(self.fc1(current_plan))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        loc, scale = torch.tensor_split(x, 2, dim=1)
        loc = self.loc_activate(loc)
        scale = self.scale_activate(scale)
        return loc, scale, new_hidden_states, new_cell_states

    def set_task_params(self, task_params):
        self.task_params = task_params


# a simple value network to speed up training. Similar attention module used.
class ValueNetwork(nn.Module):
    def __init__(self, task_params, device, observation_params=observation_params):
        super(ValueNetwork, self).__init__()
        self.task_params = task_params
        self.device = device
        self.attention = AttentionSubModule()
        self.fc1 = nn.Linear(observation_params['attention_length'], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, observation):
        observation = padding_observation(observation, self.task_params, device=self.device)
        attention = self.attention(observation)
        attention_flatten = attention.view(attention.size(0), -1)
        x = F.relu(self.fc1(attention_flatten))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

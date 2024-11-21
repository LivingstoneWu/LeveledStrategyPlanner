import torch

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
    # always add a batch_size dimension here. In the end remove the dimension again if the input has no batch_size
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
    # returned hidden_states_lists: (num_levels, batch_size, 2 (layers), hidden_size_at_level)
    # the first rows: shape (batch_size, 2, start_hidden_size)
    new_hidden_states_first_row = new_hidden_states[0]
    new_cell_states_first_row = new_cell_states[0]
    # second rows: shape (batch_size, 2, start_hidden_size-difference)
    new_hidden_states_second_row = torch.cat(new_hidden_states[1:], dim=2)
    new_cell_states_second_row = torch.cat(new_cell_states[1:], dim=2)
    difference = model_params['start_hidden_size'] - new_hidden_states_second_row.shape[2]
    new_hidden_states_second_row = torch.cat(
        (
            new_hidden_states_second_row,
            torch.zeros(new_hidden_states_second_row.shape[0], 2, difference, device=device)),
        dim=2)
    new_cell_states_second_row = torch.cat(
        (new_cell_states_second_row, torch.zeros(new_cell_states_second_row.shape[0], 2, difference, device=device)),
        dim=2)
    new_hidden_states = torch.stack((new_hidden_states_first_row, new_hidden_states_second_row), dim=2)
    new_cell_states = torch.stack((new_cell_states_first_row, new_cell_states_second_row), dim=2)
    # if batch_size was 1, remove the batch_size dimension
    if new_hidden_states.shape[0] == 1:
        new_hidden_states = new_hidden_states.squeeze(0)
        new_cell_states = new_cell_states.squeeze(0)
        loc = loc.squeeze(0)
        scale = scale.squeeze(0)
    return loc, scale, new_hidden_states, new_cell_states

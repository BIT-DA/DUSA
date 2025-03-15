import torch


def index_select_plus(inputs, index):
    assert inputs.dim() >= index.dim() and inputs.shape[:index.dim() - 1] == index.shape[:index.dim() - 1]
    index_list = []
    for i, sz in enumerate(index.shape[:-1]):
        ind_i = unsqueeze_n_times_at_dim(torch.arange(sz, device=index.device), n=index.dim() - i - 1, dim=-1)
        index_list.append(ind_i)
    index_list.append(index)
    return inputs[index_list]


def unsqueeze_n_times_at_dim(inputs, n, dim):
    if dim < 0:
        # https://pytorch.org/docs/master/generated/torch.unsqueeze.html#torch.unsqueeze
        # Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.
        dim = dim + inputs.dim() + 1
    return inputs[(slice(None),) * dim + (None,) * n]

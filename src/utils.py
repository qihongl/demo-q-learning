import torch


def to_torch(np_array):
    return torch.tensor(np_array).type(torch.FloatTensor)

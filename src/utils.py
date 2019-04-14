import torch


def to_torch(np_array):
    return torch.tensor(np_array).type(torch.FloatTensor)


def to_numpy(torch_tensor):
    return torch_tensor.data.numpy()

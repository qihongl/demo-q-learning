import torch.nn as nn


class MLP(nn.Module):
    """A agent with a hidden layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        out = self.fc1(x).sigmoid()
        out = self.fc2(out)
        return out

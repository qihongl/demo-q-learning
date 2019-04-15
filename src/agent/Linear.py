import torch.nn as nn


class Linear(nn.Module):
    """A linear agent.
    """

    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

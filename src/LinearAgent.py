import torch.nn as nn


class LinearAgent(nn.Module):
    """A linear agent.

    Parameters
    ----------
    input_dim : type
        Description of parameter `input_dim`.
    output_dim : type
        Description of parameter `output_dim`.

    Attributes
    ----------
    linear : type
        Description of attribute `linear`.

    """

    def __init__(self, input_dim, output_dim):
        super(LinearAgent, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

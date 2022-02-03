import torch
from torch import nn


class ABCModel(nn.Module):
    """
    Abstract class for generative language models.
    """
    def forward(self, x):
        ...

    def loss(self, x, y):
        ...

    def optimizer(self) -> torch.optim.Optimizer:
        ...

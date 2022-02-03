from abc import abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module):
    """
    Abstract class for generative language models.
    """
    def forward(self, x):
        ...

    def loss(self, x, y):
        ...

    def get_optimizer(self) -> torch.optim.Optimizer:
        ...

    def get_config(self):
        ...

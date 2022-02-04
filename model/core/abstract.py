import torch
from torch import nn


class BaseModel(nn.Module):
    """
    Abstract class for generative language models.
    """
    def forward(self, x):
        """
        Forward method
        :param x: input ids
        :return: logits
        """
        ...

    def loss(self, x, y):
        """
        Loss function
        :param x: logits
        :param y: labels (original input ids)
        :return: scalar
        """
        ...

    def get_optimizer(self) -> torch.optim.Optimizer:
        ...

    def get_config(self):
        ...

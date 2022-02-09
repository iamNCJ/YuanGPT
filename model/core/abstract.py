from abc import abstractmethod

import torch
from torch import nn

from config import LMConfig


class BaseModel(nn.Module):
    """
    Abstract class for generative language models.
    """

    @property
    @abstractmethod
    def config(self) -> LMConfig:
        ...

    @config.setter
    @abstractmethod
    def config(self, val):
        self.config = val

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

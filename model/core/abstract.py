from abc import abstractmethod

import torch
from torch import nn

from config import LMConfig


class BaseModel(nn.Module):
    """
    Abstract class for generative language models.
    """
    _config: LMConfig

    @property
    @abstractmethod
    def config(self) -> LMConfig:
        return self._config

    @config.setter
    @abstractmethod
    def config(self, config: LMConfig):
        self._config = config

    def forward(self, input_ids, attention_mask=None, *args):
        """
        Forward method
        :param input_ids: tensor
        :param attention_mask: tensor
        :return: logits, tensor
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

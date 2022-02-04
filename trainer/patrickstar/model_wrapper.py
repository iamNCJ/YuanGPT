from torch import nn

from model import BaseModel


class PStarModel(nn.Module):
    """
    Wrapper `BaseModel` class to make `forward` method return `loss` directly
    """
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass of the model that returns the loss
        :param x: input ids
        :return: scalar loss
        """
        return self.model.loss(self.model(x), x)

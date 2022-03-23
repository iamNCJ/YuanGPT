from torch import nn

from model import BaseModel


class ColAIModel(nn.Module):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*args)

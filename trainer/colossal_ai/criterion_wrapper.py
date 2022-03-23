from torch import nn

from model import BaseModel

class ColAICriterion(nn.Module):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, logits, labels):
        return self.model.loss(logits, labels)

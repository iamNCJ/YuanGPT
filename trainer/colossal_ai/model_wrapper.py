from torch import nn

from model import BaseModel
from colossalai.nn.optimizer import HybridAdam

class ColAIModel(nn.Module):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, *args):
        return self.model(*args)

    def loss(self, logits, labels):
        return self.model.loss(logits, labels)

    def get_optimizer(self) -> torch.optim.Optimizer:
        return HybridAdam(self.model.parameters(), lr=self.config.learning_rate)

import math

from models.config import LMConfig
from models.core.abstract import BaseModel

import torch
from torchtyping import TensorType
from colossalai.nn.optimizer import HybridAdam

from titans.model.gpt import GPT
from torch import Tensor
from torch import nn

class GenerativeLM(BaseModel):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.model = GPT(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.seq_length,
            hidden_size=config.hidden_size,
            num_heads=config.attention_heads,
            depth=config.layer_num,
            activation=nn.functional.relu,
            bias=False,
            fuse_scale_mask_softmax=False,
						checkpoint=True
        )
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, input_ids: TensorType["batch_size", "seq_length"],
                attention_masks: TensorType["batch_size", "seq_length"] = None,
                *args) \
            -> TensorType["batch_size", "seq_length", "vocab_size"]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_masks
        )

    def loss(
            self,
            logits: TensorType["batch_size", "seq_length", "vocab_size"],
            labels: TensorType["batch_size", "seq_length"]
    ) -> TensorType:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        res = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return res

    def get_optimizer(self) -> torch.optim.Optimizer:
        return HybridAdam(self.parameters(), lr=self.config.learning_rate)


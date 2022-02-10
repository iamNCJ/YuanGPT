import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from config import LMConfig
from model.core.abstract import BaseModel
from model.position_encoder import SinCosEncoding
from model.transformer_block import StandardTransformerBlock

patch_typeguard()


class GenerativeLM(BaseModel):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            SinCosEncoding(d_model=config.hidden_size, max_len=config.seq_length),
            *[StandardTransformerBlock(config.hidden_size, config.attention_heads) for _ in range(config.layer_num)],
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        )
        self.loss_fct = nn.CrossEntropyLoss()

    @typechecked
    def forward(self, input_ids: TensorType["batch_size", "seq_length"],
                attention_masks: TensorType["batch_size", "seq_length"], *args) \
            -> TensorType["batch_size", "seq_length", "vocab_size"]:
        return self.model(input_ids)

    @typechecked
    def loss(
            self,
            logits: TensorType["batch_size", "seq_length", "vocab_size"],
            labels: TensorType["batch_size", "seq_length"]
    ) -> TensorType:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


if __name__ == '__main__':
    mock_config = LMConfig(
        vocab_size=10,
        hidden_size=512,
        layer_num=6,
        attention_heads=8,
        seq_length=128,
        learning_rate=0.001,
        batch_size=32
    )
    model = GenerativeLM(mock_config)
    mock_input = torch.randint(0, 10, (4, 128))
    mock_label = torch.randint(0, 10, (4, 128))
    lm_logits = model(mock_input)
    loss = model.loss(lm_logits, mock_label)
    print(loss)

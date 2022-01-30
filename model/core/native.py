import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from model.config import LMConfig
from model.position_encoders import SinCosEncoding
from model.transformer_block import StandardTransformerBlock

patch_typeguard()


class GenerativeLM(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            SinCosEncoding(d_model=config.hidden_size, max_len=config.seq_length),
            *[StandardTransformerBlock(config.hidden_size, config.attention_heads) for _ in range(config.layer_num)],
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        )

    @typechecked
    def forward(self, input_ids: TensorType["batch_size", "seq_length"]) \
            -> TensorType["batch_size", "seq_length", "vocab_size"]:
        return self.model(input_ids)


if __name__ == '__main__':
    from model.config import LMConfig

    mock_config = LMConfig(
        vocab_size=10,
        hidden_size=512,
        layer_num=6,
        attention_heads=8,
        seq_length=128,
    )
    model = GenerativeLM(mock_config)
    mock_input = torch.randint(0, 10, (1, 128))
    print(model(mock_input).shape)

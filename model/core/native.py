import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from model.config import LMConfig
from model.position_encoder import SinCosEncoding
from model.transformer_block import StandardTransformerBlock

patch_typeguard()


class GenerativeLM(nn.Module):
    # TODO: add mask
    def __init__(self, config: LMConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            SinCosEncoding(d_model=config.hidden_size, max_len=config.seq_length),
            *[StandardTransformerBlock(config.hidden_size, config.attention_heads) for _ in range(config.layer_num)],
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        )
        self.loss_fct = nn.CrossEntropyLoss()

    @typechecked
    def forward(self, input_ids: TensorType["batch_size", "seq_length"]) \
            -> TensorType["batch_size", "seq_length", "vocab_size"]:
        return self.model(input_ids)

    @typechecked
    def loss(
            self,
            logits: TensorType["batch_size", "seq_length", "hidden_size"],
            labels: TensorType["batch_size", "seq_length"]
    ) -> TensorType:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


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
    mock_label = torch.randint(0, 10, (1, 128))
    lm_logits = model(mock_input)
    loss = model.loss(lm_logits, mock_label)
    print(loss)

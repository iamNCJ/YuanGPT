import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from flash_attn import FlashMHA

patch_typeguard()


class StandardTransformerBlock(nn.Module):
    # TODO: add mask
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        # self.MHA = nn.MultiheadAttention(
        #     embed_dim=hidden_size,
        #     num_heads=num_heads,
        #     batch_first=True,
        # )
        self.MHA = FlashMHA(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    @typechecked
    def forward(self, x: TensorType["batch_size", "seq_len", "hidden_size"]) \
            -> TensorType["batch_size", "seq_len", "hidden_size"]:
        residual_1 = self.layer_norm_1(x)
        residual_1, _ = self.MHA(residual_1, need_weights=False)
        x_1 = residual_1 + x

        residual_2 = self.layer_norm_2(x_1)
        residual_2 = self.FFN(residual_2)
        return x_1 + residual_2


if __name__ == '__main__':
    mock_input = torch.rand(2, 3, 10)
    model = StandardTransformerBlock(10, 2)
    y = model(mock_input)
    print(y.shape)

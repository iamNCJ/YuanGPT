import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


class StandardTransformerBlock(nn.Module):
    # TODO: add mask
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
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
        residual = self.layer_norm_1(x)
        residual, _ = self.MHA(residual, residual, residual, need_weights=False)
        x += residual
        residual = self.layer_norm_2(x)
        residual = self.FFN(residual)
        x += residual
        return x


if __name__ == '__main__':
    mock_input = torch.rand(2, 3, 10)
    model = StandardTransformerBlock(10, 2)
    y = model(mock_input)
    print(y.shape)

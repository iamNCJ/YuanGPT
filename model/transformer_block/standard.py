from torch import nn


class StandardTransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
    ):
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # TODO: Implement forward pass
        pass

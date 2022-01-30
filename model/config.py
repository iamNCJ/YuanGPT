from dataclasses import dataclass


@dataclass
class LMConfig:
    hidden_size: int
    attention_heads: int
    layer_num: int
    seq_length: int

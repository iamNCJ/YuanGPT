from dataclasses import dataclass


@dataclass
class LMConfig:
    hidden_size: int
    attention_heads: int
    layer_num: int
    seq_length: int
    vocab_size: int
    learning_rate: float
    batch_size: int
    accumulate_grad_batches: int = 1
    seed: int = 42

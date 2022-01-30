import pytorch_lightning as pl
import torch.nn as nn
from torchtyping import TensorType

from .config import LMConfig
from .position_encoders import SinCosEncoding


class GenerativeLM(pl.LightningModule):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = SinCosEncoding(d_model=config.hidden_size, max_len=config.seq_length)

    def forward(self,
                input_ids: TensorType["batch_size", "seq_length"]) \
            -> TensorType["batch_size", "seq_length", "hidden_size"]:
        E = self.word_embedding(input_ids)
        H = self.position_embedding(E)



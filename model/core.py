import pytorch_lightning as pl
import torch.nn as nn

from .config import LMConfig


class GenerativeLM(pl.LightningModule):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.embedding = nn.Embedding()

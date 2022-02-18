from dataclasses import asdict

import pytorch_lightning as pl
import torch

from model import BaseModel
from trainer.lightning.strategy import DistributedStrategy
from util import MemTracker


class LitModel(pl.LightningModule):
    """
    Pytorch Lightning Trainer Wrapper
    """

    def __init__(self, model: BaseModel, strategy: DistributedStrategy, profile_mem: bool = False):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.profile = profile_mem
        if self.profile:
            self.gpu_mem_tracker = MemTracker()
        self.save_hyperparameters(asdict(model.config))

    def forward(self, *args):
        # input_ids, attention_masks(optional)
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        # batch = [input_ids, attention_masks(optional)]
        if self.profile:
            self.gpu_mem_tracker.track()
        y_hat = self.forward(*batch)
        if self.profile:
            self.gpu_mem_tracker.track()
        loss = self.model.loss(y_hat, batch[0])
        if self.profile:
            self.gpu_mem_tracker.track()
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # batch = [input_ids, attention_masks(optional)]
        y_hat = self.forward(*batch)
        loss = self.model.loss(y_hat, batch[0])
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        # Do Adam on CPU when offloading
        if self.strategy.use_offload:
            # from torch.optim import Adam
            from torch.optim._multi_tensor import Adam
            return Adam(self.model.parameters(), lr=self.model.config.learning_rate)
            # from deepspeed.ops.adam import DeepSpeedCPUAdam
            # return DeepSpeedCPUAdam(self.parameters(), lr=self.model.config.learning_rate)
        # Use FusedAdam when ZeRO is on and offload is not used, which reduces optimizer state
        elif self.strategy.use_deepspeed_zero:
            from deepspeed.ops.adam import FusedAdam
            return FusedAdam(self.parameters(), lr=self.model.config.learning_rate)
        return self.model.get_optimizer()

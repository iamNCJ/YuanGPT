from dataclasses import asdict

import pytorch_lightning as pl
import torch

from model import BaseModel
from trainer.lightning.strategy import DistributedStrategy


class LitModel(pl.LightningModule):
    """
    Pytorch Lightning Trainer Wrapper
    """

    def __init__(self, model: BaseModel, strategy: DistributedStrategy):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.save_hyperparameters(asdict(model.get_config()))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.model.loss(y_hat, batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.model.loss(y_hat, batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        # Do Adam on CPU when offloading
        if self.strategy.use_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            return DeepSpeedCPUAdam(self.parameters(), lr=self.model.get_config().learning_rate)
        # Use FusedAdam when ZeRO is on and offload is not used, which reduces optimizer state
        elif self.strategy.use_deepspeed_zero:
            from deepspeed.ops.adam import FusedAdam
            return FusedAdam(self.parameters(), lr=self.model.get_config().learning_rate)
        return self.model.get_optimizer()

from dataclasses import asdict

import pytorch_lightning as pl
import torch

from model import BaseModel


class LitModel(pl.LightningModule):
    """
    Pytorch Lightning Trainer Wrapper
    """

    def __init__(self, model: BaseModel, use_deepspeed: bool = False):
        super().__init__()
        self.model = model
        self.use_deepspeed = use_deepspeed
        self.save_hyperparameters(asdict(model.get_config()))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.model.loss(y_hat, batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.model.loss(y_hat, batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        if self.use_deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            return DeepSpeedCPUAdam(self.parameters())
        return self.model.get_optimizer()

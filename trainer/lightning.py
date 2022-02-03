import pytorch_lightning as pl
import torch

from model import ABCModel


class LitModel(pl.LightningModule):
    """
    Pytorch Lightning Trainer Wrapper
    """

    def __init__(self, model: ABCModel):
        super().__init__()
        self.model = model

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
        return self.model.optimizer()


if __name__ == '__main__':
    from model import NativeModel, LMConfig
    from data import MockDataModule

    mock_config = LMConfig(
        vocab_size=51200,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=0.001
    )
    core_model = NativeModel(mock_config)
    wrapper_model = LitModel(core_model)
    dm = MockDataModule(vocab_size=mock_config.vocab_size, seq_length=mock_config.seq_length)
    trainer = pl.Trainer()
    trainer.fit(wrapper_model, dm)

import torch

from config import LMConfig
from data import YuanDataModule
from model import HFModel
import pytorch_lightning as pl
from trainer.lightning import pl_train, DistributedStrategy
from trainer.lightning.model_wrapper import LitModel

if __name__ == '__main__':

    config = LMConfig(
        vocab_size=53005,
        hidden_size=128,
        layer_num=8,
        attention_heads=8,
        seq_length=256,
        learning_rate=5e-5,
        batch_size=2,
    )
    core_model = HFModel(config)
    dm = YuanDataModule(
        batch_size=config.batch_size,
        processed_data_path='./data/yuan/processed_data.npz'
    )
    # pl_train(
    #     core_model, dm, "model_name",
    #     use_distributed=DistributedStrategy.CUSTOM,
    #     accelerator='gpu',
    #     devices=-1,
    #     precision=16,
    #     max_epochs=1,
    #     # accumulate_grad_batches=16,
    #     seed=config.seed
    # )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='fsdp_native',
    )
    trainer.fit(LitModel(core_model, strategy=DistributedStrategy.FSDP), dm)

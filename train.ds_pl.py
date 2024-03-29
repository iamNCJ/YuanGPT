import torch

from config import LMConfig
from data import YuanDataModule
from model import HFModel
from trainer.lightning import pl_train, DistributedStrategy

if __name__ == '__main__':

    config = LMConfig(
        vocab_size=53005,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=5e-5,
        batch_size=16,
    )
    core_model = HFModel(config)
    dm = YuanDataModule(
        batch_size=config.batch_size,
        processed_data_path='./data/yuan/processed_data.npz'
    )
    # dm = MockDataModule(
    #     vocab_size=config.vocab_size,
    #     seq_length=config.seq_length,
    #     batch_size=config.batch_size,
    #     mock_data_size=128
    # )
    pl_train(
        core_model, dm, "model_name",
        use_distributed=DistributedStrategy.CUSTOM,
        accelerator='gpu',
        devices=-1,
        precision=16,
        max_epochs=1,
        # accumulate_grad_batches=16,
        seed=config.seed
    )

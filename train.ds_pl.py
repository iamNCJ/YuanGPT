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
        learning_rate=0.0008,
        batch_size=16,
    )
    core_model = HFModel(config)
    dm = YuanDataModule(
        batch_size=config.batch_size,
        processed_data_path='./data/yuan/processed_data.npz'
    )
    pl_train(
        core_model, dm,
        use_distributed=DistributedStrategy.CUSTOM,
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16,
        max_epochs=1,
        seed=config.seed
    )

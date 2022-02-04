import torch

from config import LMConfig
from data import MockDataModule
from model import HFModel
from trainer.lightning import pl_train

if __name__ == '__main__':
    mock_config = LMConfig(
        vocab_size=53228,
        hidden_size=128,
        layer_num=4,
        attention_heads=24,
        seq_length=256,
        learning_rate=0.0001,
        batch_size=4,
    )
    core_model = HFModel(mock_config)
    dm = MockDataModule(
        vocab_size=mock_config.vocab_size,
        seq_length=mock_config.seq_length,
        batch_size=mock_config.batch_size,
        mock_data_size=100000
    )
    pl_train(
        core_model, dm,
        gpus=4,
        precision=16,
        max_epochs=1,
        strategy='ddp',
        num_nodes=2
    )

import torch

from config import LMConfig
from model import NativeModel
from data import MockDataModule
from trainer.lightning import pl_train

if __name__ == '__main__':
    mock_config = LMConfig(
        vocab_size=53228,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=0.0001,
        batch_size=2,
    )
    core_model = NativeModel(mock_config)
    dm = MockDataModule(
        vocab_size=mock_config.vocab_size,
        seq_length=mock_config.seq_length,
        batch_size=mock_config.batch_size,
        mock_data_size=100000
    )
    pl_train(
        core_model, dm,
        gpus=-1 if torch.cuda.is_available() else None,
        amp_backend='native',
        precision=16,
        max_epochs=1,
        strategy='ddp_sharded',
    )

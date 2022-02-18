import torch

from config import LMConfig
from data import YuanDataModule, MockDataModule
from model import HFModel
from trainer.lightning import pl_train, DistributedStrategy

if __name__ == '__main__':

    mock_config = LMConfig(
        vocab_size=53228,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=0.0001,
        batch_size=18,
    )
    core_model = HFModel(mock_config)
    dm = YuanDataModule(
        batch_size=mock_config.batch_size,
        processed_data_path='./data/yuan/processed_data.npz'
    )
    # dm = MockDataModule(
    #     batch_size=mock_config.batch_size,
    #     vocab_size=mock_config.vocab_size,
    #     seq_length=mock_config.seq_length,
    #     mock_data_size=15 * 4 * 18
    # )
    pl_train(
        core_model, dm,
        use_distributed=DistributedStrategy.DEEPSPEED_STAGE_3_OFFLOAD,
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16,
        # profiler="pytorch",
        max_epochs=1,
    )

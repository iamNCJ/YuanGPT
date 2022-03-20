from config import LMConfig
from data import MockDataModule, YuanDataModule
from trainer.deepspeed import ds_train

if __name__ == '__main__':
    config = LMConfig(
        vocab_size=53005,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=5e-5,
        batch_size=8,
    )
    dm = YuanDataModule(
        batch_size=16,
        processed_data_path='./data/yuan/processed_data.npz'
    )
    ds_train(config, dm)

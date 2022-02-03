import torch
from pytorch_lightning.plugins import DeepSpeedPlugin

from config import LMConfig
from data import MockDataModule
from model import NativeModel
from trainer.lightning import pl_train

if __name__ == '__main__':
    deepspeed_config = {
        # "zero_allow_untested_optimizer": True,
        "train_batch_size": 2,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            },
        },
        "zero_optimization": {
            "stage": 3,
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True,
            #     "buffer_count": 4,
            #     # "fast_init": False
            #   },
        },
    }

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
        precision=16,
        max_epochs=1,
        strategy='deepspeed_stage_2_offload'
        # DeepSpeedPlugin(config=deepspeed_config, logging_batch_size_per_gpu=mock_config.batch_size),
    )

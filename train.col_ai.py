from models.config import LMConfig
# from model import ColAIModel
from data import YuanDataModule, MockDataModule
from trainer.colossal_ai import col_ai_train
import torch.distributed as dist

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
    # dist.init_process_group('nccl', init_method='tcp://172.25.2.104:29500', rank=0, world_size=2)
    # dist.init_process_group('nccl')
    # config = LMConfig(
    #     vocab_size=2000,
    #     hidden_size=768,
    #     layer_num=12,
    #     attention_heads=12,
    #     seq_length=1024,
    #     learning_rate=5e-5,
    #     batch_size=16,
    # )
    # config = LMConfig(
    #     vocab_size=53228,
    #     hidden_size=3072,
    #     layer_num=40,
    #     attention_heads=24,
    #     seq_length=2048,
    #     learning_rate=0.0001,
    #     batch_size=1,
    # )
    dm = YuanDataModule(
        batch_size=config.batch_size,
        processed_data_path='./data/yuan/processed_data.npz',
        use_distributed_sampler=True
    )
    # dm = MockDataModule(
    #     vocab_size=config.vocab_size,
    #     seq_length=config.seq_length,
    #     batch_size=config.batch_size,
    #     mock_data_size=2048,
    #     use_distributed_sampler=True
    # )
    col_ai_train(
        config, dm,
        num_epochs=1,
        warmup_steps=5,
        seed=config.seed
    )
# OMP_NUM_THREADS=32 torchrun --standalone --nnodes=1 --nproc_per_node 1 train.col_ai.py --config=trainer/colossal_ai/strategy.py
# OMP_NUM_THREADS=32 torchrun --standalone --nnodes=1 --nproc_per_node 2 train.col_ai.py --config=trainer/colossal_ai/strategy.py --from_torch
# OMP_NUM_THREADS=32 torchrun --rdzv_endpoint=172.25.2.104:29400 --nnodes=2 --node_rank=0 --nproc_per_node=2 train.col_ai.py --config=trainer/colossal_ai/strategy.py --from_torch

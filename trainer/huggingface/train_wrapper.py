import sys

sys.path.append('../../')
import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import BaseModel
from torchtyping import TensorType
from model import HFModel
from config import LMConfig
from data import MockDataModule, YuanDataModule
from transformers import Trainer, TrainingArguments


class HFTrainer(Trainer):
    def __init__(self,
                 model: BaseModel,
                 data_module: pl.LightningDataModule,
                 args: TrainingArguments,
                 ):
        super().__init__(model, args,
                         train_dataset=data_module.train_dataloader(),
                         eval_dataset=data_module.val_dataloader())
        self.data_module = data_module
        self.loss_fct = nn.CrossEntropyLoss()
        self.is_model_parallel = True

    def get_train_dataloader(self):
        return self.data_module.train_dataloader()

    def get_eval_dataloader(self):
        return self.data_module.val_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False) -> TensorType:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # logits = model(**inputs)
        logits = model(inputs[0] if len(inputs) == 1 else inputs)
        loss = self.model.loss(logits, inputs[0])
        return (loss, logits) if return_outputs else loss

    def create_optimizer(self) -> torch.optim.Optimizer:
        return self.model.get_optimizer()


def train(
        model: BaseModel,
        data_module: pl.LightningDataModule,
        args: TrainingArguments,
) -> None:
    trainer = HFTrainer(
        model=model,
        data_module=data_module,
        args=args,
    )
    trainer.train()


custom_deepspeed_config = {
    # Batch Size
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 16,
    # Do not set `gradient_accumulation_steps` in the DeepSpeed config as this will be set
    # with the `accumulate_grad_batches` argument passed via the Lightning Trainer.

    # amp
    "bf16": {
        "enabled": True
    },

    # ZeRO
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "stage": 2,
        "offload_parameters": False,  # Enable Offloading parameters to the host CPU
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        # "offload_param": {
        #     "device": "cpu",
        #     "pin_memory": False
        # },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
    },

    # Logging
    # "steps_per_print": 1,
    # "wall_clock_breakdown": False,
    # "tensorboard": {
    #     "enabled": False,
    #     "output_path": "logs/ds_logs/",
    #     "job_name": f"train_gpt2_yuan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # },

    # Profiling
    "flops_profiler": {
        "enabled": False,
        "profile_step": 2,
        "module_depth": -1,
        "top_modules": 10,
        "detailed": True,
        "output_file": "./flops_profiler.txt",
    }
}

if __name__ == '__main__':
    mock_config = LMConfig(
        vocab_size=53005,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=5e-5,
        batch_size=16,
    )
    mock_data = False
    if mock_data:
        dm = MockDataModule(
            vocab_size=mock_config.vocab_size,
            seq_length=mock_config.seq_length,
            batch_size=mock_config.batch_size
        )
    else:
        dm = YuanDataModule(
            batch_size=mock_config.batch_size,
            # processed_data_path='/shared/YuanDataset/processed_data.npz',
            processed_data_path='/workspace/shared/YuanDataset/processed_data.npz',
            use_distributed_sampler=True,
        )
    dm.setup()
    hf_model = HFModel(config=mock_config)
    training_args = TrainingArguments(output_dir='test_trainer',
                                      do_train=True,
                                      logging_steps=1,
                                      num_train_epochs=1.0,
                                      per_device_train_batch_size=custom_deepspeed_config[
                                          "train_micro_batch_size_per_gpu"],
                                      gradient_accumulation_steps=custom_deepspeed_config[
                                          "gradient_accumulation_steps"],
                                      deepspeed=custom_deepspeed_config
                                      )
    train(model=hf_model, data_module=dm, args=training_args)

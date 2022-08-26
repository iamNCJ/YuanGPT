from transformers.deepspeed import HfDeepSpeedConfig
import pytorch_lightning as pl
import deepspeed
import torch
from tqdm import tqdm

from config import LMConfig
from model import BaseModel
from model.core import HFModel
from trainer.deepspeed.model_wrapper import DeepSpeedModel 

ds_config = {
    # Batch Size
    # "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 1,
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
        # "offload_parameters": True,  # Enable Offloading parameters to the host CPU
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        # "offload_param": {
        #     "device": "cpu",
        #     "pin_memory": True
        # },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
    },
    # "activation_checkpointing": {
    #     "partition_activations": True,
    #     "cpu_checkpointing": True,
    #     "contiguous_memory_optimization": True,
    # },

    # Logging
    # "steps_per_print": 1,
    # "wall_clock_breakdown": False,
    # "tensorboard": {
    #     "enabled": False,
    #     "output_path": "logs/ds_logs/",
    #     "job_name": f"train_gpt2_yuan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    # Profiling
    # "flops_profiler": {
    #     "enabled": False,
    #     "profile_step": 2,
    #     "module_depth": -1,
    #     "top_modules": 10,
    #     "detailed": True,
    #     "output_file": "./flops_profiler.txt",
    # }
}

dschf = HfDeepSpeedConfig(ds_config) # keep this object alive


def train(config: LMConfig, data_module: pl.LightningDataModule) -> None:
    model = DeepSpeedModel(HFModel(config))
    
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    data_module.setup()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.train()
    losses = []
    for data in tqdm(data_module.train_dataloader(), desc="Training"):
        # Move the tensors to device
        data = [d.to(device) for d in data]
        # Forward pass
        loss = model(*data)
        # Backward pass
        model.backward(loss)
        # Optimizer Step
        model.step()
        losses.append(loss.item())
        tqdm.write(f"loss: {loss.item()}")
    

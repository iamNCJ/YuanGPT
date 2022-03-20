import os
import datetime
import json
import pathlib
import re
import string
import logging
import loguru
import random
import pytz
import sh
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from transformers.deepspeed import HfDeepSpeedConfig
import pytorch_lightning as pl
import deepspeed
import torch
from tqdm import tqdm

from config import LMConfig
from model import BaseModel
from model.core import HFModel
from trainer.deepspeed.model_wrapper import DeepSpeedModel 

######################################################################
########### Experiment Management Related Functions ##################
######################################################################


def get_unique_identifier(length: int = 8) -> str:
    """Create a unique identifier by choosing `length`
    random characters from list of ascii characters and numbers
    """
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix]
                   for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(checkpoint_dir: pathlib.Path,
                          all_arguments: Dict[str, Any]) -> pathlib.Path:
    """Create an experiment directory and save all arguments in it.
    Additionally, also store the githash and gitdiff. Finally create
    a directory for `Tensorboard` logs. The structure would look something
    like
        checkpoint_dir
            `-experiment-name
                |- hparams.json
                |- githash.log
                |- gitdiff.log
                `- tb_dir/

    Args:
        checkpoint_dir (pathlib.Path):
            The base checkpoint directory
        all_arguments (Dict[str, Any]):
            The arguments to save

    Returns:
        pathlib.Path: The experiment directory
    """
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
    expname = "yuan_deepspeed.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    if not is_rank_0():
        return exp_dir
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_128:
        log_dist(
            "Seems like the code is not running from"
            " within a git repo, so hash will"
            " not be stored. However, it"
            " is strongly advised to use"
            " version control.",
            ranks=[0],
            level=logging.INFO)
    # And the git diff
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_129:
        log_dist(
            "Seems like the code is not running from"
            " within a git repo, so diff will"
            " not be stored. However, it"
            " is strongly advised to use"
            " version control.",
            ranks=[0],
            level=logging.INFO)
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir(exist_ok=False)
    return exp_dir


######################################################################
################ Checkpoint Related Functions ########################
######################################################################


def load_model_checkpoint(
    load_checkpoint_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """Loads the optimizer state dict and model state dict from the load_checkpoint_dir
    into the passed model and optimizer. Searches for the most recent checkpoint to
    load from

    Args:
        load_checkpoint_dir (pathlib.Path):
            The base checkpoint directory to load from
        model (torch.nn.Module):
            The model to load the checkpoint weights into
        optimizer (torch.optim.Optimizer):
            The optimizer to load the checkpoint weigths into

    Returns:
        Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
            The checkpoint step, model with state_dict loaded and
            optimizer with state_dict loaded

    """
    log_dist(
        f"Loading model and optimizer checkpoint from {load_checkpoint_dir}",
        ranks=[0],
        level=logging.INFO)
    checkpoint_files = list(
        filter(
            lambda path: re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name) is
            not None,
            load_checkpoint_dir.glob("*.pt"),
        ))
    assert len(checkpoint_files) > 0, "No checkpoints found in directory"
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda path: int(
            re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name).group("iter_no")
        ),
    )
    latest_checkpoint_path = checkpoint_files[-1]
    checkpoint_step = int(
        re.search(r"iter_(?P<iter_no>\d+)\.pt",
                  latest_checkpoint_path.name).group("iter_no"))

    state_dict = torch.load(latest_checkpoint_path)
    model.load_state_dict(state_dict["model"], strict=True)
    optimizer.load_state_dict(state_dict["optimizer"])
    log_dist(
        f"Loading model and optimizer checkpoints done. Loaded from {latest_checkpoint_path}",
        ranks=[0],
        level=logging.INFO)
    return checkpoint_step, model, optimizer


ds_config = {
    # Batch Size
    # "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
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
        # if step % log_every == 0:
        #     log_dist("Loss: {0:.4f}".format(np.mean(losses)),
        #              ranks=[0],
        #              level=logging.INFO)
        #     if is_rank_0():
        #         summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
        # if step % checkpoint_every == 0:
        #     model.save_checkpoint(save_dir=exp_dir,
        #                           client_state={'checkpoint_step': step})
        #     log_dist("Saved model to {0}".format(exp_dir),
        #              ranks=[0],
        #              level=logging.INFO)
    # Save the last checkpoint if not saved yet
    # if step % checkpoint_every != 0:
    #     model.save_checkpoint(save_dir=exp_dir,
    #                           client_state={'checkpoint_step': step})
    #     log_dist("Saved model to {0}".format(exp_dir),
    #              ranks=[0],
    #              level=logging.INFO)
    # deepspeed.init_distributed()
    

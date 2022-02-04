import pytorch_lightning as pl
import torch
from tqdm import tqdm

from config import LMConfig
from patrickstar.runtime import initialize_engine
from patrickstar.utils import get_rank

from model.core import HFModel

pstar_config = {
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0,
            "use_hybrid_adam": True,
        },
    },
    "fp16": {  # loss scaler params
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 2 ** 3,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "default_chunk_size": 256 * 1024 * 1024,
    "release_after_init": True,
    "use_cpu_embedding": False
}


def train(config: LMConfig, data_module: pl.LightningDataModule):
    def model_func():
        return HFModel(config)

    model, optimizer = initialize_engine(
        model_func=model_func,
        local_rank=get_rank(),
        config=pstar_config
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_module.setup()
    for data in tqdm(data_module.train_dataloader(), desc="Training"):
        optimizer.zero_grad()
        data = data.to(device)
        logits = model(data)
        loss = model.loss(logits, data)
        model.backward(loss)
        optimizer.step()
        tqdm.write(f"loss: {loss}")

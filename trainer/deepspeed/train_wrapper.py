import pytorch_lightning as pl

import deepspeed
from model import BaseModel


def train(
        model: BaseModel,
        data_module: pl.LightningDataModule
) -> None:
    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=model,
                                                         model_parameters=params)

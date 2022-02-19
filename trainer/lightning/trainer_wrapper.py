# from dataclasses import asdict

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelSummary

from model import BaseModel
from trainer.lightning.model_wrapper import LitModel
# from trainer.lightning.named_logger import NamedLogger
from trainer.lightning.strategy import DistributedStrategy


def train(
        model: BaseModel,
        data_module: pl.LightningDataModule,
        use_distributed: DistributedStrategy = DistributedStrategy.NONE,
        seed: int = 42,
        profile_mem: bool = False,
        **kwargs
) -> None:
    """
    Do train using Pytorch-Lightning trainer
    :param model: `BaseModel` instance
    :param data_module: `pl.LightningDataModule` instance
    :param use_distributed: `DistributedStrategy` enum
    :param seed: random seed
    :param profile_mem: whether to use gpu mem tracer to profile gpu memory usage
    :param kwargs: other kwargs for `pl.Trainer`
    """

    # set seed
    pl.seed_everything(seed)

    # do train
    wrapper_model = LitModel(model, strategy=use_distributed, profile_mem=profile_mem)
    # logger = NamedLogger(asdict(model.config))
    trainer = pl.Trainer(
        logger=False,
        log_every_n_steps=1,
        # callbacks=[ModelSummary(max_depth=3)],
        strategy=use_distributed.pl_strategy,
        **kwargs
    )
    trainer.fit(wrapper_model, data_module)


if __name__ == '__main__':
    from config import LMConfig
    from model import NativeModel
    from data import MockDataModule

    mock_config = LMConfig(
        vocab_size=53228,
        hidden_size=3072,
        layer_num=40,
        attention_heads=24,
        seq_length=2048,
        learning_rate=0.0001,
        batch_size=32,
    )
    core_model = NativeModel(mock_config)
    dm = MockDataModule(
        vocab_size=mock_config.vocab_size,
        seq_length=mock_config.seq_length,
        batch_size=mock_config.batch_size
    )
    train(core_model, dm)

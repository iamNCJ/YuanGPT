import logging
import sys
import os
from dataclasses import asdict
from datetime import datetime

import pytorch_lightning as pl
# from pytorch_lightning.callbacks.progress.tqdm_progress import reset

from config import LMConfig
from model import BaseModel
from trainer.lightning.model_wrapper import LitModel
from trainer.lightning.named_logger import NamedLogger
from trainer.lightning.strategy import DistributedStrategy
from trainer.lightning.stream_logger import StreamToLogger

from pytorch_lightning.callbacks import Callback, TQDMProgressBar


class TimerCallback(Callback):
    def __init__(self, log_name):
        super().__init__()
        self.log_name = log_name

    def on_train_start(self, trainer, pl_module):
        print(f"Training has started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")

    def on_train_end(self, trainer, pl_module):
        print(f"Training has finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")

    # def on_exception(self, trainer, pl_module, exception):
    #     print("Saving model...")
    #     trainer.save_checkpoint(f"./ckpt/{self.log_name}")
    #     print("Model saved!")


# class LitProgressBar(TQDMProgressBar):
#     def __init__(self, config: LMConfig):
#         super().__init__()
#         self.lr = config.learning_rate
#         self.batch_size = config.batch_size

#     def on_train_epoch_start(self, trainer, pl_module):
#         super().on_train_epoch_start(trainer, pl_module)
#         total_train_batches = self.total_train_batches
#         total_val_batches = self.total_val_batches
#         if total_train_batches != float("inf") and total_val_batches != float("inf"):
#             # val can be checked multiple times per epoch
#             val_checks_per_epoch = total_train_batches // trainer.val_check_batch
#             total_val_batches = total_val_batches * val_checks_per_epoch
#         total_batches = total_train_batches + total_val_batches
#         reset(self.main_progress_bar, total=total_batches, current=self.train_batch_idx)
#         self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch} lr={self.lr} bs={self.batch_size}")


def train(
        model: BaseModel,
        data_module: pl.LightningDataModule,
        log_name: str = None,
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

    # configure logging at the root level of lightning
    logger = logging.getLogger("pytorch_lightning")
    fh = logging.FileHandler(f"log/train_gpt2_yuan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # redirect stdout & stderr to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # set seed
    pl.seed_everything(seed)

    # do train
    wrapper_model = LitModel(model, strategy=use_distributed, profile_mem=profile_mem)
    board_logger = NamedLogger(asdict(model.config), log_name)
    trainer = pl.Trainer(
        logger=board_logger,
        log_every_n_steps=1,
        enable_model_summary=False,
        strategy=use_distributed.pl_strategy,
        num_sanity_val_steps=0,
        callbacks=[TimerCallback(log_name)],
        **kwargs
    )

    if os.path.exists(f"./ckpt/{log_name}"):
        trainer.fit(wrapper_model, data_module, ckpt_path=f"./ckpt/{log_name}")
    else:
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

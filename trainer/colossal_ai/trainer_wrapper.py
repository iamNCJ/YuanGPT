from config import LMConfig
import contextlib
import torch
import pytorch_lightning as pl
from model import BaseModel
from trainer.colossal_ai.criterion_wrapper import ColAICriterion
# from trainer.colossal_ai.model_wrapper import ColAIModel
from model import ColAIModel, HFModel, NativeModel

import colossalai
import colossalai.utils as utils
from colossalai.context.parallel_mode import ParallelMode
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc
from colossalai.nn import LinearWarmupLR
from colossalai.utils.timer import MultiTimer
from colossalai.trainer import hooks, Trainer
from colossalai.zero.init_ctx import ZeroInitContext

def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device

def train(config: LMConfig,
          data_module: pl.LightningDataModule,
          num_epochs: int = 1,
          warmup_steps: int = 5,
          seed: int = 42) -> None:
    """
    Do train using ColossalAI trainer
    :param model: `BaseModel` instance
    :param data_module: `pl.LightningDataModule`
    :param seed: random seed
    """
    logger = get_dist_logger()

    parser = colossalai.get_default_parser()

    args = parser.parse_args()
    disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config, seed=seed)
    logger.info('ColAI launched', ranks=[0])

    use_zero = hasattr(gpc.config, 'zero')
    ctx = contextlib.nullcontext()
    if use_zero:
        ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                              shard_strategy=gpc.config.zero.model_config.shard_strategy,
                              shard_param=True)
    with ctx:
        model = NativeModel(config)

    if use_zero:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    logger.info(f'Numel: {numel}', ranks=[0])

    criterion = ColAICriterion(model)
    optimizer = model.get_optimizer()

    data_module.setup(has_labels=True)
    # train_dataloader = data_module.train_dataloader()
    train_dataloader = utils.get_dataloader(data_module.dataset.train_dataset,
                                            seed=seed,
                                            batch_size=config.batch_size,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=8)
    logger.info('dataloader built', ranks=[0])

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=num_epochs, warmup_steps=warmup_steps)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    logger.info('ColAI initialized', ranks=[0])
    global_batch_size = config.batch_size * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    timier = MultiTimer()

    trainer = Trainer(
        engine=engine,
        logger=logger,
        # schedule=schedule,
        timer=timier
    )

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        # hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False
    )

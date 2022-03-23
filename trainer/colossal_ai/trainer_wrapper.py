import pytorch_lightning as pl
from model import BaseModel
from trainer.colossal_ai.criterion_wrapper import ColAICriterion
from trainer.colossal_ai.model_wrapper import ColAIModel

import colossalai
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.nn import LinearWarmupLR
from colossalai.utils.timer import MultiTimer
from colossalai.trainer import hooks, Trainer

def train(model: BaseModel,
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
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config,
                                     seed=seed)
    else:
        colossalai.launch_from_slurm(config=args.config,
                                     host=args.host,
                                     port=29500,
                                     seed=seed)
    logger.info('ColAI launched', ranks=[0])

    optimizer = model.get_optimizer()
    criterion = ColAICriterion(model)
    data_module.setup(has_labels=True)
    train_dataloader = data_module.train_dataloader()
    logger.info('dataloader built', ranks=[0])

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=num_epochs, warmup_steps=warmup_steps)

    model = ColAIModel(model);
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    logger.info('ColAI initialized', ranks=[0])

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
        hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.LogMemoryByEpochHook(logger),
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

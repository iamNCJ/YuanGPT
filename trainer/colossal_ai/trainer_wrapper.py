from config import LMConfig
import contextlib
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pytorch_lightning as pl
from model import BaseModel
from trainer.colossal_ai.criterion_wrapper import ColAICriterion
from model import HFModel, NativeModel, ColAIModel, PPModel
from model.core.pipeline.hf_pp import PipelineGPT2Model

from torch.optim import SGD

import colossalai
import colossalai.utils as utils
from colossalai.context.parallel_mode import ParallelMode
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc
from colossalai.nn import LinearWarmupLR
from colossalai.utils.timer import MultiTimer
from colossalai.trainer import hooks, Trainer
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.utils import is_using_pp
from colossalai.utils import get_current_device
from transformers import GPT2Config

from colossalai.nn.optimizer import HybridAdam
from colossalai.pipeline.pipeline_process_group import ppg

from colossalai.nn.parallel import GeminiDDP

from tqdm import tqdm

import os

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
    # os.environ['MASTER_ADDR'] = '172.25.1.105'
    # os.environ['MASTER_PORT'] = '29501'
    logger = get_dist_logger()

    parser = colossalai.get_default_parser()

    

    args = parser.parse_args()
    print(args)
    disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config, seed=seed)
    logger.info('ColAI launched', ranks=[0])

    use_zero = hasattr(gpc.config, 'zero')
    use_pipeline = is_using_pp()
    num_chunks = getattr(gpc.config.model, 'num_chunks', 1)

    # ppg.set_global_info(rank=args.rank,
    #                     world_size=args.world_size,
    #                     dp_degree=2)

    if not use_pipeline:
        ctx = contextlib.nullcontext()
        if use_zero:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                shard_param=True)
        with ctx:
            model = HFModel(config)
        # with ColoInitContext(device=get_current_device()):
        #     model = HFModel(config)
        # model = GeminiDDP(model,
        #                   device=get_current_device(),
        #                   placement_policy='cuda',
        #                   pin_memory=True,
        #                   search_range_mb=32)
    else:
        pipelinable = PipelinableContext()
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.seq_length,
            n_embd=config.hidden_size,
            n_layer=config.layer_num,
            n_head=config.attention_heads,
            activation_function='relu',
            n_inner=4 * config.hidden_size,
            use_cache=False
        )
        with pipelinable:
            model = PipelineGPT2Model(gpt2_config, config.batch_size)
        exec_seq = ['embedding']
        for i in range(config.layer_num):
            exec_seq.append('blocks.{}'.format(i))
        # exec_seq.append('blocks')
        exec_seq.append('norm')
        exec_seq.append('head')
        
        pipelinable.to_layer_list(exec_seq)
        ctx = contextlib.nullcontext()
        if use_zero:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                shard_param=True)
        with ctx:
            model = pipelinable.partition(num_chunks, gpc.pipeline_parallel_size,
                                          gpc.get_local_rank(ParallelMode.PIPELINE))
    if use_zero:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    # numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Numel: {numel}', ranks=[0])

    criterion = ColAICriterion(model)
    # optimizer = model.get_optimizer()
    optimizer = SGD(model.parameters(), lr=config.learning_rate)

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
    # with profile(
    #     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('/workspace/log/yuan_profile'),
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True, 
    #     profile_memory=True, 
    #     with_stack=True
    # ) as prof:
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False
    )
    # losses = []
    # model.train()
    # for data in tqdm(train_dataloader, desc="Training"):
    #     # Move the tensors to device
    #     # print(data)
    #     print(get_current_device())
    #     data = data[get_current_device().index].cuda()
    #     print(data.shape)
    #     optimizer.zero_grad()
    #     print('Forward')
    #     outputs = model(data)
    #     print(outputs.shape)
    #     loss = criterion(outputs, data)
    #     # Forward pass
    #     print('Backward')
    #     # Backward pass
    #     optimizer.backward(loss)
    #     # Optimizer Step
    #     print('Optimizer')
    #     optimizer.step()
    #     losses.append(loss.item())
    #     tqdm.write(f"loss: {loss.item()}")
        # engine.train()
        # for data, label in train_dataloader:
        #     data = data.cuda()
        #     label = label.cuda()
        #     engine.zero_grad()
        #     output = engine(data)
        #     loss = engine.criterion(output, label)
        #     engine.backward(loss)
        #     engine.step()
        #     prof.step()
    # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
    # prof.export_chrome_trace('trace.json')
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    

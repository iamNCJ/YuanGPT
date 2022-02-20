# from datetime import datetime
from enum import Enum
from pytorch_lightning.plugins import DeepSpeedPlugin

custom_deepspeed_config = {
    # Batch Size
    "train_micro_batch_size_per_gpu": 16,
    # "gradient_accumulation_steps": 1,
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
            "pin_memory": False,
        },
        "round_robin_gradients": True,  # Stage 2 optimization for CPU offloading that parallelizes gradient copying
        # "reduce_scatter": False  # Use allReduce
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


class DistributedStrategy(str, Enum):
    """
    Enum for the different distributed strategies.
    """
    NONE = 0
    DDP = 1
    DDP_SHARDED = 2
    DEEPSPEED_STAGE_1 = 3
    DEEPSPEED_STAGE_2 = 4
    DEEPSPEED_STAGE_3 = 5
    DEEPSPEED_STAGE_2_OFFLOAD = 6
    DEEPSPEED_STAGE_3_OFFLOAD = 7
    CUSTOM = 8

    @property
    def use_offload(self):
        """
        Check whether the strategy uses ZeRO offload.
        :return: bool
        """
        return self in [DistributedStrategy.DEEPSPEED_STAGE_2_OFFLOAD,
                        DistributedStrategy.DEEPSPEED_STAGE_3_OFFLOAD]

    @property
    def use_deepspeed_zero(self):
        """
        Check whether the strategy uses DeepSpeed ZeRO (no offload).
        :return: bool
        """
        return self in [DistributedStrategy.DEEPSPEED_STAGE_1,
                        DistributedStrategy.DEEPSPEED_STAGE_2,
                        DistributedStrategy.DEEPSPEED_STAGE_3]

    @property
    def use_custom(self):
        """
        Check whether the strategy uses custom DeepSpeed config json.
        :return: bool
        """
        return self == DistributedStrategy.CUSTOM

    @property
    def pl_strategy(self) -> str:
        mapping = {
            DistributedStrategy.NONE: None,
            DistributedStrategy.DDP: 'ddp',
            DistributedStrategy.DDP_SHARDED: 'ddp_sharded',
            DistributedStrategy.DEEPSPEED_STAGE_1: 'deepspeed_stage_1',
            DistributedStrategy.DEEPSPEED_STAGE_2: 'deepspeed_stage_2',
            DistributedStrategy.DEEPSPEED_STAGE_3: 'deepspeed_stage_3',
            DistributedStrategy.DEEPSPEED_STAGE_2_OFFLOAD: 'deepspeed_stage_2_offload',
            DistributedStrategy.DEEPSPEED_STAGE_3_OFFLOAD: 'deepspeed_stage_3_offload',
            DistributedStrategy.CUSTOM: DeepSpeedPlugin(config=custom_deepspeed_config)
        }
        return mapping[self]

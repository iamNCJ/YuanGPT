from enum import Enum
from pytorch_lightning.plugins import DeepSpeedPlugin

custom_deepspeed_config = {
    # Batch Size
    "train_micro_batch_size_per_gpu": 18,
    "gradient_accumulation_steps": 1,
    # Precision
    "bf16": {
        "enabled": True
    },
    # ZeRO
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "stage": 3,
        # "offload_optimizer": True,  # Enable Offloading optimizer state/calculation to the host CPU
        "offload_parameters": True,  # Enable Offloading parameters to the host CPU
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        }
    },
    # Logging
    "logging": {
        "steps_per_print": 1,
        "wall_clock_breakdown": True,
        "dump_state": True,
    },
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
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

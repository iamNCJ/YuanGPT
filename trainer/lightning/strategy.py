from enum import Enum


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
    def pl_strategy(self) -> str:
        mapping = {
            DistributedStrategy.NONE: None,
            DistributedStrategy.DDP: 'ddp',
            DistributedStrategy.DDP_SHARDED: 'ddp_sharded',
            DistributedStrategy.DEEPSPEED_STAGE_1: 'deepspeed_stage_1',
            DistributedStrategy.DEEPSPEED_STAGE_2: 'deepspeed_stage_2',
            DistributedStrategy.DEEPSPEED_STAGE_3: 'deepspeed_stage_3',
            DistributedStrategy.DEEPSPEED_STAGE_2_OFFLOAD: 'deepspeed_stage_2_offload',
            DistributedStrategy.DEEPSPEED_STAGE_3_OFFLOAD: 'deepspeed_stage_3_offload'
        }
        return mapping[self]

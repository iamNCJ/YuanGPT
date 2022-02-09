from enum import Enum


class DistributedStrategy(str, Enum):
    """
    Enum for the different distributed strategies.
    """
    NONE = None
    DDP = 'ddp'
    DDP_SHARDED = 'ddp_sharded'
    DEEPSPEED_STAGE_1 = 'deepspeed_stage_1'
    DEEPSPEED_STAGE_2 = 'deepspeed_stage_2'
    DEEPSPEED_STAGE_3 = 'deepspeed_stage_3'
    DEEPSPEED_STAGE_2_OFFLOAD = 'deepspeed_stage_2_offload'
    DEEPSPEED_STAGE_3_OFFLOAD = 'deepspeed_stage_3_offload'

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

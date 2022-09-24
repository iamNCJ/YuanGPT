from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.amp import AMP_TYPE


zero = dict(
    # model_config=dict(
    #     tensor_placement_policy='auto',
    #     shard_strategy=TensorShardStrategy()
    # ),
    # optimizer_config=dict()
    model_config=dict(
        shard_strategy=TensorShardStrategy(),
        reduce_scatter_bucket_size_mb=25,
        fp32_reduce_scatter=False,
        tensor_placement_policy="cuda",
        gradient_predivide_factor=1.0,
        reuse_fp16_shard=False
    ),
    optimizer_config=dict(
        gpu_margin_mem_ratio=0.8,
        initial_scale=2**5,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2,
        max_scale=2**32
    )
)

# fp16 = dict(
#     mode=AMP_TYPE.TORCH,
# )

# gradient_accumulation = 16

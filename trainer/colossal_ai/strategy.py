from colossalai.zero.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
from colossalai.amp import AMP_TYPE

BATCH_SIZE = 16
NUM_MICRO_BATCHES = 2
SEQ_LEN = 2048
HIDDEN_SIZE = 3072

TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

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
model = dict(
    num_chunks=1
)

gradient_accumulation = 16

parallel = dict(
    pipeline=2
)

from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.amp import AMP_TYPE


zero = dict(
    model_config=dict(
        tensor_placement_policy='auto',
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict()
)


gradient_accumulation = 2

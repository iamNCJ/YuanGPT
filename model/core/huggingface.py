import torch
from torch import nn
from torch.distributed.fsdp.wrap import wrap
from torchtyping import TensorType
from transformers import GPT2Config
from transformers.modeling_utils import no_init_weights

from config import LMConfig
from model.core.abstract import BaseModel
from model.core.hf_backbone import GPT2LMHeadHackedModel

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


class GenerativeLM(BaseModel):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.model = None
        self.config = config
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
        with no_init_weights(_enable=False):
            self.model = GPT2LMHeadHackedModel(gpt2_config)
            # self.model.lm_head = torch.nn.functional.linear
        # self.model.gradient_checkpointing_enable()
        self.loss_fct = nn.CrossEntropyLoss()

    def configure_sharded_model(self):
        for i, layer in enumerate(self.model.transformer.h):
            self.model.transformer.h[i] = wrap(layer)
        self.model.lm_head = wrap(self.model.lm_head)

    def forward(self, input_ids: TensorType["batch_size", "seq_length"],
                attention_masks: TensorType["batch_size", "seq_length"] = None,
                *args) \
            -> TensorType["batch_size", "seq_length", "vocab_size"]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_masks
        ).logits

    def loss(
            self,
            logits: TensorType["batch_size", "seq_length", "vocab_size"],
            labels: TensorType["batch_size", "seq_length"]
    ) -> TensorType:
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # half = self.config.batch_size // 2
        # shift_logits0 = shift_logits[:half, ...]
        # shift_logits1 = shift_logits[half:, ...]
        # shift_labels0 = shift_labels[:half, ...]
        # shift_labels1 = shift_labels[half:, ...]
        # print(shift_logits0.shape, shift_labels0.shape)
        # res0 = self.loss_fct(shift_logits0.view(-1, shift_logits0.size(-1)), shift_labels0.view(-1))
        # res1 = self.loss_fct(shift_logits1.view(-1, shift_logits1.size(-1)), shift_labels1.view(-1))
        res = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # print(res0.shape, res1.shape)
        # print(res0)quit()
        # assert torch.isnan(shift_logits).sum() == 0
        # assert torch.isnan(shift_labels).sum() == 0
        # assert torch.isnan(res).sum() == 0
        # return (res0 * half + res1 * (self.config.batch_size - half)) / self.config.batch_size
        return res

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

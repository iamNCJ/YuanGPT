import torch
from torch import nn
from torchtyping import TensorType
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_utils import no_init_weights

from config import LMConfig
from model.core.abstract import BaseModel


class GenerativeLM(BaseModel):
    def __init__(self, config: LMConfig):
        super().__init__()
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
        with no_init_weights(_enable=True):
            self.model = GPT2LMHeadModel(gpt2_config)
            self.model.gradient_checkpointing_enable()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids: TensorType["batch_size", "seq_length"],
                attention_masks: TensorType["batch_size", "seq_length"] = None) \
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
        return self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

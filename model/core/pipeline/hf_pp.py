from typing import Optional, Tuple, Union
from colossalai.context.parallel_mode import ParallelMode
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.pipeline.utils import partition_uniform

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from colossalai.nn.layer.base_layer import ParallelLayer
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
# from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from transformers.pytorch_utils import Conv1D

# from flash_attn.flash_attention import FlashMHA
from .flash_attention import FlashHackedMHA
from .embed import VocabParallelEmbedding, VocabParallelGPTLMHead

class GPT2HackedMLP(ParallelLayer):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class FlashGPT2Block(ParallelLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.index = layer_idx

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.attn = FlashHackedMHA(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
            causal=True  # for auto-regressive modeling in GPT
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2HackedMLP(inner_dim, config)

    def _forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor],
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states
        )
        attn_output = attn_outputs[0]  # output_attn: attn, weights (None)
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states  # hidden_states

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        print('index: ', self.index)
        if self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward),
                hidden_states,
                attention_mask,
            )
        else:
            return self._forward(
                hidden_states,
                attention_mask
            )


class GPTLMHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 seq_len: int,
                 embedding_layer) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.dense = col_nn.Classifier(hidden_size, vocab_size, embedding_layer.weight)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        print('LMHead')
        # the size of x before dense is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # the size of x after dense is (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        x = x.view(-1, self.seq_len, self.hidden_size)
        x = self.dense(x)
        print(x.size())
        return x



class PipelineGPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, batch_size):
        super().__init__(config)

        self.input_shape = (batch_size, config.n_positions)
        
        # self.embed_dim = config.n_embd

        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # self.drop = nn.Dropout(config.embd_pdrop)
        self.embedding = VocabParallelEmbedding(
            config.n_embd, 
            config.vocab_size, 
            config.n_positions, 
            config.embd_pdrop
        )

        self.blocks = nn.ModuleList(
            [FlashGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # self.lm_head = VocabParallelGPTLMHead(vocab_size=config.vocab_size, embed_dim=config.n_embd)
        self.head = GPTLMHead(config.n_embd, config.vocab_size, self.config.n_positions, self.embedding.word_embeddings)

        # self.transformer = FlashGPT2Model(config)
        # self.lm_head = GPTLMHead(
        #     hidden_size=config.hidden_size,
        #     vocab_size=config.vocab_size,
        #     embedding_layer=self.wte
        # )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        # hidden_states: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """


        # device = input_ids.device

        # position_ids = torch.arange(0, self.input_shape[-1], dtype=torch.long, device=device)
        # position_ids = position_ids.unsqueeze(0).view(-1, self.input_shape[-1])

        # inputs_embeds = self.wte(input_ids)
        # position_embeds = self.wpe(position_ids)
        # hidden_states = inputs_embeds + position_embeds

        # hidden_states = self.drop(hidden_states)
        hidden_states = self.embedding(input_ids=input_ids)


            

        # GPT2Attention mask.
        # if attention_mask is not None:

        #     attention_mask = attention_mask.view(self.config.batch_size, -1)
        #     attention_mask = attention_mask[:, None, None, :]
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min


        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)
        # output_shape = self.input_shape + (hidden_states.size(-1),)
        # hidden_states = hidden_states.view(output_shape)
        lm_logits = self.head(hidden_states)
        return lm_logits





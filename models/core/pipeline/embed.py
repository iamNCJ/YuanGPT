
import torch
from torch import nn as nn, Tensor, distributed as dist
from torch.nn import functional as F
import torch.nn.init as init
from colossalai.context import seed, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from colossalai.nn.layer.base_layer import ParallelLayer
from torch.nn.parameter import Parameter
from colossalai.registry import LAYERS, LOSSES, MODELS
from colossalai.nn.layer.utils import divide
from colossalai.nn.layer.parallel_1d._utils import gather_forward_split_backward, reduce_grad, reduce_input
from colossalai.nn.layer.parallel_1d.layers import Linear1D_Row


class VocabParallelEmbedding(torch.nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 dtype=torch.float):
        super(VocabParallelEmbedding, self).__init__()

        self.hidden_size = hidden_size

        # Word embeddings (parallel).
        # self.word_embeddings = VocabParallelEmbedding1D(
        #     vocab_size, self.hidden_size, dtype=dtype)
        self.word_embeddings = torch.nn.Embedding(
            vocab_size, self.hidden_size, dtype=dtype)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, self.hidden_size, dtype=dtype)
        self._position_embeddings_key = 'position_embeddings'


        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)


    def forward(self, input_ids, position_ids=None):
        # Embeddings.
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        words_embeddings = self.word_embeddings(input_ids)

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long, device=get_current_device())
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings

        # Dropout.
        with seed(ParallelMode.TENSOR):
            embeddings = self.embedding_dropout(embeddings)
        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] \
            = self.position_embeddings.state_dict(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] \
                        = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)



class VocabParallelGPTLMHead(ParallelLayer):
    """
    Language model head that shares the same parameters with the embedding matrix.
    """

    def __init__(self,
                 embed=None,
                 vocab_size=None,
                 dtype=None,
                 embed_dim=None
                 ):
        super().__init__()
        if embed is not None:
            self.head = embed
        else:
            self.head = torch.nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.head.weight)
        return x


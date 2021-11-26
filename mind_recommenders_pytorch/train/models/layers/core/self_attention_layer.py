import numpy as np
import torch
from torch.nn import Linear

from .masked_softmax import MaskedSoftmax


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.Query = self._init_linear(input_dim, output_dim)
        self.Key = self._init_linear(input_dim, output_dim)
        self.Value = self._init_linear(input_dim, output_dim)

        self.masked_softmax = MaskedSoftmax()

    @staticmethod
    def _init_linear(input_dim, output_dim):
        linear = Linear(input_dim, output_dim, bias=False)
        torch.nn.init.xavier_uniform_(linear.weight)
        return linear

    def forward(self, batch_sequence, sequence_mask):
        query = self.Query(batch_sequence)
        key = self.Key(batch_sequence)
        value = self.Value(batch_sequence)

        multihead_query = self._split_heads(query)
        multihead_key = self._split_heads(key)
        multihead_value = self._split_heads(value)

        attention_logit = torch.matmul(multihead_query, multihead_key.transpose(3, 2))  # dot(query, key.T)
        attention_logit /= np.sqrt(self.output_dim)  # scaling for dot product

        attention_weight = self.masked_softmax(attention_logit, self._calc_mask_matrix(sequence_mask))

        multihead_output = torch.matmul(attention_weight, multihead_value)
        attention_output = self._concat_heads(multihead_output)

        return attention_output

    def _split_heads(self, matrix):
        return torch.stack(matrix.split(self.num_heads, dim=-1))

    def _concat_heads(self, multihead_output):
        return torch.cat(tuple(multihead_output), dim=-1)

    def _calc_mask_matrix(self, sequence_mask):
        return sequence_mask.unsqueeze(-1) * sequence_mask.unsqueeze(-2)

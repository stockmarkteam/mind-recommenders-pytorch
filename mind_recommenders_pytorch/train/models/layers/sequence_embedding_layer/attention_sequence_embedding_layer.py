import torch

from ..core.attention_layer import AttentionLayer
from ..core.self_attention_layer import SelfAttentionLayer


class AttentionSequenceEmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=16,
        attn_hidden_dim=200,
        p_dropout=0.2,
    ):
        super().__init__()
        self.intermediate_layer = SelfAttentionLayer(input_dim, output_dim, num_heads)
        self.attention_layer = AttentionLayer(
            input_dim=output_dim,
            hidden_dim=attn_hidden_dim,
        )
        self.dropout = torch.nn.Dropout(p_dropout)
        
    def forward(self, embedded_sequence, sequence_mask):
        token_intermediates = self.intermediate_layer(embedded_sequence, sequence_mask)
        token_intermediates = self.dropout(token_intermediates)

        token_weights = self.attention_layer(
            token_intermediates,
            sequence_mask,
        )

        news_embeddings = (token_intermediates * token_weights.unsqueeze(-1)).sum(axis=1)

        return news_embeddings

import torch

from ..core.attention_layer import AttentionLayer


class CnnSequenceEmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        attn_hidden_dim=200,
        kernel_size=3,
        p_dropout=0.2,
    ):
        super().__init__()
        self.intermediate_layer = CnnIntermediateLayer(input_dim, output_dim, kernel_size)
        self.attention_layer = AttentionLayer(
            input_dim=output_dim,
            hidden_dim=attn_hidden_dim,
        )
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, embedded_sequence, sequence_mask):
        token_intermediates = self.intermediate_layer(embedded_sequence, sequence_mask)
        token_intermediates = self.dropout(token_intermediates)

        token_weights = self.attention_layer(token_intermediates, sequence_mask)
        news_embeddings = (token_intermediates * token_weights.unsqueeze(-1)).sum(axis=1)

        return news_embeddings


class CnnIntermediateLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            padding="same",
        )
        self.activation = torch.nn.ReLU()

    def forward(self, embedded_sequence, mask):
        hidden_tokens = self.conv(embedded_sequence.transpose(2, 1))
        hidden_tokens = self.activation(hidden_tokens.transpose(1, 2))
        return hidden_tokens * mask.unsqueeze(-1)

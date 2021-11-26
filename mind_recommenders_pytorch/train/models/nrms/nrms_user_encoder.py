from typing import List

import torch

from ..layers.sequence_embedding_layer.attention_sequence_embedding_layer import AttentionSequenceEmbeddingLayer
from ...data.objects import ArticleInput


class NrmsUserEncoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        attn_hidden_dim=200,
        p_dropout=0.2,
        num_heads=16,
        max_history_length=50,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length

        self.sequence_embedding_layer = AttentionSequenceEmbeddingLayer(
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            attn_hidden_dim=attn_hidden_dim,
            num_heads=num_heads,
            p_dropout=p_dropout,
        )

    def forward(self, histories: List[ArticleInput], news_encoder):
        encoded_histories = [news_encoder(history) for history in histories]
        encoded_histories = torch.stack([self._pad_history(history) for history in encoded_histories])
        history_mask = self._build_sequence_mask(histories, device=encoded_histories.device)

        user_embeddings = self.sequence_embedding_layer(encoded_histories, history_mask)

        return user_embeddings

    def _pad_history(self, encoded_history):
        zero_pad = torch.zeros(
            self.max_history_length - encoded_history.shape[0],
            encoded_history.shape[1],
        ).to(encoded_history.device)
        return torch.vstack([encoded_history, zero_pad])

    def _build_sequence_mask(self, histories, device):
        return torch.vstack(
            [
                torch.cat(
                    [
                        torch.ones(history.n_articles),
                        torch.zeros(self.max_history_length - history.n_articles),
                    ]
                )
                for history in histories
            ]
        ).to(device)

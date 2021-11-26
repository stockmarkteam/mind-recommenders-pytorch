import numpy as np
import torch
from torch.nn import Dropout, Embedding


class WordEmbeddingBasedTokenEmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_path,
        p_dropout=0.2,
    ):
        super().__init__()
        self.embedding_layer = self._init_embedding(embedding_path)
        self.dropout = Dropout(p_dropout)

    def _init_embedding(self, embedding_path):
        weights = np.load(embedding_path)
        weights = np.vstack([np.zeros(weights.shape[-1]), weights])
        return Embedding.from_pretrained(
            torch.Tensor(weights),
            freeze=False,
            sparse=True,
        )

    def forward(self, input_ids):
        token_embeddings = self.embedding_layer(input_ids)
        token_embeddings = self.dropout(token_embeddings)
        return token_embeddings

    @property
    def embedding_dim(self):
        return self.embedding_layer.embedding_dim

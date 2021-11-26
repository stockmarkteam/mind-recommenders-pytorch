import torch
from torch.nn import Dropout
from transformers import AutoModel


class TransformerBasedTokenEmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        model_path,
        p_dropout=0.2,
        n_trainable_layers=2,
    ):
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_path)
        self.dropout = Dropout(p_dropout)
        self.n_trainable_layers = n_trainable_layers
        self._disable_grad()

    def _disable_grad(self):
        for emb_param in self.transformer_model.embeddings.parameters():
            emb_param.requires_grad = False
        for layer in self.transformer_model.encoder.layer[: -self.n_trainable_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.transformer_model.pooler.parameters():
            param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        output last token embeddings
        """
        out = self.transformer_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        token_embeddings = out.hidden_states[-1]
        token_embeddings = self.dropout(token_embeddings)
        return token_embeddings

    @property
    def embedding_dim(self):
        return self.transformer_model.embeddings.word_embeddings.embedding_dim

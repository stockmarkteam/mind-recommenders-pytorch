import torch

from ..layers.category_encoder import CategoryEncoder
from ..layers.core.attention_layer import AttentionLayer
from ..layers.sequence_embedding_layer.cnn_sequence_embedding_layer import CnnSequenceEmbeddingLayer
from ..layers.sequence_encoder import SequenceEncoder


class NamlNewsEncoder(torch.nn.Module):
    def __init__(
        self,
        token_embedding_layer,
        article_attributes,
        n_categories=None,
        n_subcategories=None,
        output_dim=400,
        attn_hidden_dim=200,
        category_embedding_dim=100,
        kernel_size=3,
        attention_dropout=0.2,
    ):
        super().__init__()
        self.article_attributes = article_attributes
        self.token_embedding_layer = token_embedding_layer
        self.embedding_dim = self.token_embedding_layer.embedding_dim

        if "title" in article_attributes:
            self.title_encoder = SequenceEncoder(
                self.token_embedding_layer,
                CnnSequenceEmbeddingLayer(
                    input_dim=self.embedding_dim,
                    output_dim=output_dim,
                    kernel_size=kernel_size,
                ),
            )

        if "body" in article_attributes:
            self.body_encoder = SequenceEncoder(
                self.token_embedding_layer,
                CnnSequenceEmbeddingLayer(
                    input_dim=self.embedding_dim,
                    output_dim=output_dim,
                    kernel_size=kernel_size,
                ),
            )

        if "category" in article_attributes:
            self.category_encoder = CategoryEncoder(
                n_categories=n_categories,
                embedding_dim=category_embedding_dim,
                output_dim=output_dim,
            )

        if "subcategory" in article_attributes:
            self.subcategory_encoder = CategoryEncoder(
                n_categories=n_subcategories,
                embedding_dim=category_embedding_dim,
                output_dim=output_dim,
            )

        self.attention_layer = AttentionLayer(
            input_dim=output_dim,
            hidden_dim=attn_hidden_dim,
            dropout=attention_dropout,
        )

    def _encode_title(self, news_input):
        return self.title_encoder(**news_input.title._asdict())

    def _encode_body(self, news_input):
        return self.body_encoder(**news_input.body._asdict())

    def _encode_category(self, news_input):
        return self.category_encoder(news_input.category)

    def _encode_subcategory(self, news_input):
        return self.subcategory_encoder(news_input.subcategory)

    def forward(self, news_input):
        encoded_attributes = [getattr(self, f"_encode_{attr}")(news_input) for attr in self.article_attributes]
        stacked_embeddings = torch.stack(encoded_attributes).transpose(1, 0)

        weights = self.attention_layer(stacked_embeddings)

        weigted_averaged_embeddings = (stacked_embeddings * weights.unsqueeze(-1)).sum(axis=1)

        return weigted_averaged_embeddings

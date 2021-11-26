from torch.nn import CELU, Embedding, Linear, Module, Sequential

class CategoryEncoder(Module):
    def __init__(
        self,
        n_categories,
        embedding_dim,
        output_dim,
    ):
        """
        (category+1) * embedding_dim次元のembedding layerを持つCategoryEncoder
        index=0は未知カテゴリのために用意している
        ただし、未知カテゴリ（idx=0）が入力に与えられたときの出力のmasking（例：ゼロ乗算する）等については考慮できていない（実験設定では未知カテゴリが出現しないため）
        """
        super().__init__()
        self.emb_layer = Embedding(
            num_embeddings=n_categories+1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.dense = Sequential(
            Linear(
                in_features=embedding_dim,
                out_features=output_dim,
            ),
            CELU(),
        )

    def forward(self, sequences):
        embeddings = self.emb_layer(sequences)
        embeddings = self.dense(embeddings)
        return embeddings

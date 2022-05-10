import torch

from perceiver_io.decoders import PerceiverDecoder
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io import PerceiverIO

from ..layers.fourier_encoder import fourier_encode

import pandas as pd

class PerceiverIONewsEncoder(torch.nn.Module):
    def __init__(
        self,
        token_embedding_layer,
        article_attributes,
        n_categories=None,
        n_subcategories=None,
        num_heads=8,
        num_latents = 256,
        latent_dim = 64,
        num_self_attn_per_block = 12,
        num_blocks = 4,
        decoder_query_dim = 200,
        max_history_length=50,
        max_title_length = 30,
        max_body_length = 128,
        word_pos_emb = True,
        seq_pos_emb = True,
        feat_type_emb = True
    ):
        super().__init__()
        self.article_attributes = article_attributes
        self.token_embedding_layer = token_embedding_layer
        self.embedding_dim = self.token_embedding_layer.embedding_dim

        self.output_dim = decoder_query_dim

        self.word_pos_emb = word_pos_emb
        self.seq_pos_emb = seq_pos_emb
        self.feat_type_emb = feat_type_emb

        self.max_title_length = max_title_length
        self.max_body_length = max_body_length
        self.max_history_length = max_history_length

        max_attr_len = 1
        if "title" in article_attributes or "title_avg" in article_attributes:
            self.title_encoder = self.token_embedding_layer
            if self.max_title_length > max_attr_len: max_attr_len = self.max_title_length

        if "body" in article_attributes or "body_avg" in article_attributes:
            self.body_encoder = self.token_embedding_layer
            if self.max_body_length > max_attr_len: max_attr_len = self.max_body_length

        if "category" in article_attributes:
            self.category_encoder = torch.nn.Embedding(
                num_embeddings=n_categories+1,
                embedding_dim=self.embedding_dim,
            )

        if "subcategory" in article_attributes:
            self.subcategory_encoder = torch.nn.Embedding(
                num_embeddings=n_subcategories+1,
                embedding_dim=self.embedding_dim,
            )

        input_enc_dims = 0
        if self.word_pos_emb: input_enc_dims += 8
        if self.seq_pos_emb: input_enc_dims += 9
        if self.feat_type_emb: input_enc_dims += 9

        self.encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=self.embedding_dim+input_enc_dims,
            num_self_attn_per_block=num_self_attn_per_block,
            num_blocks=num_blocks,
            num_self_attn_heads=num_heads
        )
        self.decoder = PerceiverDecoder(
            latent_dim=latent_dim,
            query_dim=decoder_query_dim
        )
        self.perceiver = PerceiverIO(self.encoder, self.decoder)

        self.query_output = torch.nn.Parameter(torch.randn(1, 1, decoder_query_dim))

        with torch.no_grad():
            self.feat_type_embedding = fourier_encode(torch.tensor(range(len(self.article_attributes))), 16)
            self.position_embedding = fourier_encode(
                    torch.tensor(range(max(max_attr_len, max_history_length) + 1)), 16)


    def _encode_title(self, news_input, i):
        title_emb = self.title_encoder(news_input.title.input_ids)

        if self.feat_type_emb:
            title_emb = torch.cat((title_emb,
                               self.feat_type_embedding[i, :].
                               to(title_emb.device).unsqueeze(0).
                               repeat([title_emb.shape[0], title_emb.shape[1], 1])),
                              dim=2)
        if self.word_pos_emb:
            title_emb = torch.cat((title_emb,
            self.position_embedding[:title_emb.shape[1], :-1].
                       to(title_emb.device).unsqueeze(0).
                       repeat([title_emb.shape[0], 1, 1])),
                      dim=2)

        return title_emb, news_input.title.attention_mask > 0


    def _encode_body(self, news_input, i):
        body_emb = self.body_encoder(news_input.body.input_ids)

        if self.feat_type_emb:
            body_emb = torch.cat((body_emb,
                                  self.feat_type_embedding[i, :].
                                  to(body_emb.device).unsqueeze(0).
                                  repeat([body_emb.shape[0], body_emb.shape[1], 1])),
                                 dim=2)

        if self.word_pos_emb:
            body_emb = torch.cat((body_emb,
                                 self.position_embedding[:body_emb.shape[1], :-1].
                                 to(body_emb.device).unsqueeze(0).
                                 repeat([body_emb.shape[0], 1, 1])),
                             dim=2)

        return body_emb, news_input.body.attention_mask > 0

    def _encode_category(self, news_input, i):
        emb = self.category_encoder(news_input.category)
        emb=emb.reshape(emb.shape[0], 1,-1)
        if self.feat_type_emb:
            emb = torch.cat((emb,
                                   self.feat_type_embedding[i, :].
                                   to(emb.device).unsqueeze(0).
                                   repeat([emb.shape[0], emb.shape[1], 1])),
                                  dim=2)
        if self.word_pos_emb:
            emb = torch.cat(
                    (
                        emb,
                        torch.zeros((emb.shape[0], emb.shape[1], 8), device=emb.device)
                    ), dim=2
                )

        return emb, torch.ones((emb.shape[0],emb.shape[1]), device=emb.device) == 1

    def _encode_subcategory(self, news_input, i):
        emb = self.subcategory_encoder(news_input.subcategory)
        emb=emb.reshape(emb.shape[0], 1,-1)
        if self.feat_type_emb:
            emb = torch.cat((emb,
                                   self.feat_type_embedding[i, :].
                                   to(emb.device).unsqueeze(0).
                                   repeat([emb.shape[0], emb.shape[1], 1])),
                                  dim=2)
        if self.word_pos_emb:
            emb = torch.cat(
                    (
                        emb,
                        torch.zeros((emb.shape[0], emb.shape[1], 8), device=emb.device)
                    ), dim=2
                )

        return emb, torch.ones((emb.shape[0],emb.shape[1]), device=emb.device) == 1

    def _apply_encoding_and_position_embedding(self, news_input, user_encoder=False):
        encoded_attributes = []
        attention_masks = []
        i = 0
        for attr in self.article_attributes:
            enc_att, att_mask = getattr(self, f"_encode_{attr}")(news_input, i)
            i+=1

            encoded_attributes.append(enc_att)
            attention_masks.append(att_mask)

        attention_masks = torch.cat(attention_masks, 1)
        stacked_embeddings = torch.cat(encoded_attributes, dim=1)

        if self.seq_pos_emb:

            if user_encoder:
                # behavior sequence
                stacked_embeddings = torch.cat((stacked_embeddings,
                                   self.position_embedding[-stacked_embeddings.shape[0]:, :].
                                   to(stacked_embeddings.device).unsqueeze(1).
                                   repeat([1, stacked_embeddings.shape[1], 1])),
                                  dim=2)
            else:
                # candidate
                stacked_embeddings = torch.cat((stacked_embeddings,
                                    self.position_embedding[self.max_history_length, :].
                                    to(stacked_embeddings.device).unsqueeze(0).
                                    repeat([stacked_embeddings.shape[0], stacked_embeddings.shape[1], 1])),
                                   dim=2)
        return stacked_embeddings, attention_masks

    def forward(self, news_input):

        # candidates
        if type(news_input) is not list:
            embeddings, attention_masks = self._apply_encoding_and_position_embedding(news_input)
            return self.perceiver(embeddings, self.query_output, attention_masks).squeeze()

        # user encoder
        else:
            embeddings = []
            # for each user
            for news in news_input:
                emb, attention_masks = self._apply_encoding_and_position_embedding(news, True)
                if emb == None or len(emb) == 0:
                    emb = torch.tensor([], device=news.vector.device)
                    attention_masks = torch.tensor([], dtype=bool, device=news.vector.device)
                else:
                    emb = emb.reshape(1, emb.shape[0] * emb.shape[1], emb.shape[2])
                    attention_masks = attention_masks.reshape(1, emb.shape[0] * emb.shape[1])

                # can call perceiver once for a batch if all behaviors have the same length
                embeddings.append(self.perceiver(emb, self.query_output, attention_masks).squeeze())

        return torch.stack(embeddings)

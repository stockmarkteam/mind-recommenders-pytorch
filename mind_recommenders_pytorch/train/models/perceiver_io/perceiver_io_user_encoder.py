# from .perceiver_io_news_encoder import PerceiverIONewsEncoder
from typing import List
from ...data.objects import ArticleInput
import torch

class PerceiverIOUserEncoder(torch.nn.Module):

    def forward(self, histories: List[ArticleInput], news_encoder):
        encoded_histories = [history for history in histories]

        return news_encoder(encoded_histories)

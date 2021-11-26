from collections import namedtuple
from copy import copy

import numpy as np
import torch

SequenceInput = namedtuple(
    "SequenceInput",
    (
        "input_ids",
        "token_type_ids",
        "attention_mask",
    ),
    defaults=(
        torch.Tensor([[]]),
        torch.Tensor([[]]),
        torch.Tensor([[]]),
    ),
)

ArticleInput = namedtuple(
    "ArticleInput",
    (
        "title",
        "body",
        "category",
        "subcategory",
        "n_articles",
    ),
    defaults=(
        SequenceInput(),
        SequenceInput(),
        torch.Tensor([]),
        torch.Tensor([]),
        0,
    ),
)


class Impressions:
    def __init__(self, impressions):
        self.positive_impressions = MonauralImpressions([x[0] for x in impressions if x[1] == 1])
        self.negative_impressions = MonauralImpressions([x[0] for x in impressions if x[1] == 0])

    def sample(self, n_negatives=5):
        return self.positive_impressions.sample(1), self.negative_impressions.sample(n_negatives)


class MonauralImpressions:
    def __init__(self, article_ids):
        self.original_ids = article_ids
        self.store = []
        self._fill_store()

    def _fill_store(self):
        self.store = np.hstack([self.store, np.random.permutation(copy(self.original_ids))]).astype(np.uint32)

    def sample(self, n_samples):
        while len(self.store) < n_samples:
            self._fill_store()

        sampled = self.store[:n_samples]
        self.store = self.store[n_samples:]

        return sampled

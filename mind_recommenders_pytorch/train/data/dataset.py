import numpy as np
import torch
from torch.utils.data import Dataset
from .objects import Impressions


class MindDatasetBase(Dataset):
    def __init__(
        self,
        samples,
        articles,
        article_attributes,
        max_history_length, 
        article_input_converter,
    ):
        super().__init__()
        self.samples = samples
        self.articles = articles
        self.article_attributes = article_attributes

        self.max_history_length = max_history_length

        self.article_converter = article_input_converter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self):
        raise NotImplementedError

    def _article_ids_to_input(self, article_ids):
        articles = self.articles.loc[article_ids]
        return self.article_converter.run(articles, self.article_attributes)


class MindTrainDataset(MindDatasetBase):
    def __init__(
        self,
        samples,
        articles,
        article_attributes,
        article_input_converter,
        max_history_length=50,
        n_negatives=4,
    ):
        super().__init__(
            samples,
            articles,
            article_attributes,
            max_history_length=max_history_length,
            article_input_converter=article_input_converter,
        )
        self.n_negatives = n_negatives
        self.samples["impression_pool"] = self.samples.impressions.map(lambda imp: Impressions(imp))

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        labels = torch.Tensor([1] + [0] * self.n_negatives)

        impression_pool = sample.impression_pool
        candidate_ids = np.hstack(impression_pool.sample(self.n_negatives))
        candidates = self._article_ids_to_input(candidate_ids)
        history = self._article_ids_to_input(sample.history[: self.max_history_length])

        return candidates, labels, history



class MindValidationDataset(MindDatasetBase):
    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        impressions = sample.impressions

        labels = torch.Tensor([x[1] for x in impressions])

        candidate_ids = [x[0] for x in impressions]
        candidates = self._article_ids_to_input(candidate_ids)

        history = self._article_ids_to_input(sample.history[: self.max_history_length])


        return candidates, labels, history

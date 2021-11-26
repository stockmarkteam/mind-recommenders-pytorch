import torch

from .utils import stack_article_inputs

class InputCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        candidates = stack_article_inputs([sample[0] for sample in batch])
        n_candidates_per_sample = tuple([len(sample[1]) for sample in batch])
        labels = torch.hstack([sample[1] for sample in batch])
        histories = tuple([sample[2] for sample in batch])

        return candidates, n_candidates_per_sample, labels, histories


from dataclasses import dataclass

import torch


@dataclass
class EvaluationResult:
    targets: torch.Tensor
    predictions: torch.tensor
    auc: float
    mrr: float
    ndcg_5: float
    ndcg_10: float

    def to_numpy(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.cpu().numpy())
        return self

    def __getitem__(self, key):
        return getattr(self, key)

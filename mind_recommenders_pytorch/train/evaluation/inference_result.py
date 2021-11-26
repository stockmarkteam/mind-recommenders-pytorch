from dataclasses import dataclass

import torch

@dataclass
class InferenceResult:
    predictions: torch.Tensor
    targets: torch.Tensor
    n_candidates_per_sample: tuple
    

    def to_dict(self):
        return self.__dict__

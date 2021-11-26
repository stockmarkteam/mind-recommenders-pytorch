import torch
import numpy as np

from .inference_result import InferenceResult


class InferenceResultAggregator:
    AGGREGATE_RULES = {
        "predictions": torch.hstack,
        "targets": torch.hstack,
        "n_candidates_per_sample": lambda results: tuple(np.hstack(results))
    }

    @classmethod
    def run(cls, outputs):
        results = dict()
        for key, agg_func in cls.AGGREGATE_RULES.items():
            values = [out[key] for out in outputs]
            results[key] = agg_func(values)
        return InferenceResult(**results)

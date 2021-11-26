from .inference_result import InferenceResult
import torch

class Inferencer:
    @staticmethod
    def run(model, batch):
        candidates, n_candidates_per_sample, labels, histories = batch
        with torch.no_grad():
            preds = model.forward(candidates, histories, n_candidates_per_sample)

        return InferenceResult(
            predictions = preds.cpu(),
            targets=labels.cpu(),
            n_candidates_per_sample=n_candidates_per_sample,
        ).to_dict()

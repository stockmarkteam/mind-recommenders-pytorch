import numpy as np
import torch
from sklearn import metrics

from .evaluation_result import EvaluationResult
from .inference_result import InferenceResult

# these metrics are borrowed from here: https://github.com/msnews/MIND/blob/master/evaluate.py 
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

class Evaluator:
    @classmethod
    def run(
        cls,
        result: InferenceResult,
    ) -> EvaluationResult:

        predictions = result.predictions.numpy()
        targets = result.targets.numpy()

        auc = metrics.roc_auc_score(targets, predictions)

        samplewise_predictions = [x.numpy() for x in result.predictions.split(result.n_candidates_per_sample)]
        samplewise_targets = [x.numpy() for x in result.targets.split(result.n_candidates_per_sample)]

        mrr = np.mean([mrr_score(y_true, y_score) for y_true, y_score in zip(samplewise_targets, samplewise_predictions)])
        
        ndcg_5 = np.mean(
            [ndcg_score(y_true, y_score, k=5) for y_true, y_score in zip(samplewise_targets, samplewise_predictions)]
        )
        ndcg_10 = np.mean(
            [ndcg_score(y_true, y_score, k=10) for y_true, y_score in zip(samplewise_targets, samplewise_predictions)]
        )

        return EvaluationResult(
            auc=auc,
            mrr=mrr,
            ndcg_5=ndcg_5,
            ndcg_10=ndcg_10,
            targets=targets,
            predictions=predictions,
        )

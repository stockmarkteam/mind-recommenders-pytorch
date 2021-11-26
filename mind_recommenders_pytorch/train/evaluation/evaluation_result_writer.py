from .evaluation_result import EvaluationResult


class EvaluationResultWriter:
    def __init__(self, model, phase_name):
        self.model = model
        self.phase_name = phase_name

    def run(self, result: EvaluationResult):
        self.log(result, "auc")
        self.log(result, "mrr")
        self.log(result, "ndcg_5")
        self.log(result, "ndcg_10")

    def log(self, result, key):
        self.model.log(f"{self.phase_name}_{key}", result[key])

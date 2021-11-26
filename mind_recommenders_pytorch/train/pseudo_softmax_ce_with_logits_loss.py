import torch

class PseudoSoftmaxCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def forward(self, preds, targets, n_candidates_per_sample):
        pseudo_class_preds = self.softmax(torch.vstack(preds.split(n_candidates_per_sample)))
        pseudo_class_targets = torch.vstack(targets.split(n_candidates_per_sample)).argmax(axis=1)
        loss = self.loss_func(pseudo_class_preds, pseudo_class_targets)
        return loss

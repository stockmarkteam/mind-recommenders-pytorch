import torch
from torch.nn import Linear

from .masked_softmax import MaskedSoftmax


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            Linear(hidden_dim, 1),
        )
        self.masked_softmax = MaskedSoftmax()

    def forward(self, batch_sequence, mask=None):
        outs = self.dense(batch_sequence).squeeze(-1)
        if mask is None:
            mask = torch.ones(outs.shape).to(outs.device)
        weights = self.masked_softmax(outs, mask)
        return weights

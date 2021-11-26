import torch


class MaskedSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_sequence, mask):
        # https://github.com/huggingface/transformers/issues/542
        mask = (1 - mask).type(torch.float) * (-100000)
        batch_sequence = batch_sequence + mask
        return torch.nn.Softmax(dim=-1)(batch_sequence)

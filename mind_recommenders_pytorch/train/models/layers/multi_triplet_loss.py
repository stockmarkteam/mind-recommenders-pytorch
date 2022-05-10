import torch

class MultiTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.loss_func = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, users_cands_pair, targets, n_candidates_per_sample):
        user_emb = users_cands_pair[0]
        cand_embs = users_cands_pair[1]

        cand_embs = torch.vstack(cand_embs)

        cands_pos = cand_embs[(targets == 1).nonzero(as_tuple=True)]
        cands_neg = cand_embs[(targets == 0).nonzero(as_tuple=True)].reshape(cands_pos.shape[0], -1, cands_pos.shape[1])

        loss = 0
        for i in range(cands_neg.shape[1]):
            loss += self.loss_func(user_emb, cands_pos, cands_neg[:,i,:])
        return loss


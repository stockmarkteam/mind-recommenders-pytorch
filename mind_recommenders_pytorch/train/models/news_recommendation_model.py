import torch

class NewsRecommendationModel(torch.nn.Module):
    def __init__(self, news_encoder, user_encoder):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
    
    def forward(self, candidates, histories, n_candidates_per_user):
        
        candidate_embeddings = self.news_encoder(candidates)
        user_embeddings = self.user_encoder(histories, self.news_encoder)

        userwise_candidates = candidate_embeddings.split(n_candidates_per_user)

        return torch.hstack(
            [
                torch.matmul(user, candidates.T)
                for user, candidates in zip(user_embeddings.split(1), userwise_candidates)
            ]
        ).flatten()

import torch
from torch import nn 
from video_idm.config import action_dim, tau_embedding_dim


class AdaLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_proj = nn.Linear(action_dim, action_dim)
        self.beta_proj = nn.Linear(action_dim, action_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=action_dim)

    def forward(self, X, tau_embedding):
        # X: (B, length, action_dim)
        gamma = self.gamma_proj(tau_embedding).unsqueeze(1)
        beta = self.beta_proj(tau_embedding).unsqueeze(1)        
        # gamma: (B, action_dim) -> (B, 1, action_dim)
        return self.layer_norm(X) * (1 + gamma) + beta 
    

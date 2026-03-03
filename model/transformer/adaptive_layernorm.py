import torch
import math 
from torch import nn 
from config import * 


class AdaLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_proj = nn.Linear(action_dim, action_dim)
        self.beta_proj = nn.Linear(action_dim, action_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=action_dim)

    def forward(self, X, tau):
        # X: (B, length, action_dim)
        sin_embed_tau = self.sinusoidal_embedding(tau)
        gamma = self.gamma_proj(sin_embed_tau).unsqueeze(1)
        beta = self.beta_proj(sin_embed_tau).unsqueeze(1)        
        # gamma: (B, action_dim) -> (B, 1, action_dim)
        return self.layer_norm(X) * (1 + gamma) + beta 
    
    def sinusoidal_embedding(self, tau):
        # tau: (B, 1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)  # (B, 1)
        
        half_dim = action_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=tau.device) / half_dim)
        # freqs: (half_dim, )
        angles = tau * freqs 
        
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # emb: (B, action_dim)
        return emb 
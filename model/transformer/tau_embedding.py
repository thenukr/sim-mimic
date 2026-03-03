import torch
from torch import nn 
from config import * 
import math 

class TauEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim * 4),
            nn.SiLU(), 
            nn.Linear(action_dim * 4, action_dim)
        )

    def forward(self, tau):
        x = self.sinusoidal_embedding(tau)
        return self.mlp(x)
    
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
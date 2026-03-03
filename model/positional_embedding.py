import torch
from torch import nn

from .config import action_dim, max_seq_len

class PositionalEmbeddings(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.max_seq_len = max_seq_len
        self.dim = action_dim

        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, action_dim)) * 0.02 

        
    def forward(self, tokens):
        # tokens: (B, length + 1, action_dim)
        seq_len = tokens.shape[1]
        pos = self.pos_embed[:seq_len, :]
    
        return tokens + pos 

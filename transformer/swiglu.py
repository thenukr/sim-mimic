import torch
from torch import nn 
from video_idm.config import D_model, hidden_dim, action_dim

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.D_model = action_dim
        self.hidden_dim = hidden_dim 

        self.W_gate = nn.Linear(self.D_model, self.hidden_dim, bias = False) 
        self.W_up   = nn.Linear(self.D_model, self.hidden_dim, bias=False)

        # Projection back down
        self.W_down = nn.Linear(self.hidden_dim, self.D_model, bias=False)

        # Activation
        self.act = nn.SiLU()

    def forward(self, x):

        gate = self.W_gate(x)        # (B, T, hidden_dim)
        up   = self.W_up(x)          # (B, T, hidden_dim)

        gated = self.act(gate) * up  # elementwise multiply

        out = self.W_down(gated)   # (B, T, D_model)

        return out
    

import torch
from torch import nn 
from config import * 
from adaptive_layernorm import AdaLN
from cross_attention import MultiHeadCrossAttention
from self_attention import MultiHeadAttention
from swiglu import SwiGLU

class DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.AdaLN1 = AdaLN()
        self.AdaLN2 = AdaLN()
        self.AdaLN3 = AdaLN()

        self.cross_attention = MultiHeadCrossAttention()
        self.self_attention = MultiHeadAttention()
        self.SwiGLU = SwiGLU()

    def forward(self, X, Z, tau_embedding):
        # X: (B, length + 1, action_dim)
        # Z: (B, num_patches, action_dim)

        X = X + self.self_attention(self.AdaLN1(X, tau_embedding))
        X = X + self.cross_attention(self.AdaLN2(X, tau_embedding), Z)
        X = X + self.SwiGLU(self.AdaLN3(X, tau_embedding))

        return X 


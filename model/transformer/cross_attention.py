import torch 
from torch import nn  
from config import *  

class MultiHeadCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = n_heads
        self.D_model = action_dim 
        self.D_k = self.D_model // self.H 

        self.W_q = nn.Linear(D_model, D_model, bias=False)
        self.W_k = nn.Linear(D_model, D_model, bias=False)
        self.W_v = nn.Linear(D_model, D_model, bias=False)
        self.W_o = nn.Linear(D_model, D_model, bias=False)

    def forward(self, X, Z): 
        B, T, D_model = X.shape 
        B, T_z, D_model = Z.shape 
        # X: (B, chunk_length + 1, action_dim)
        # Z: (B, num_patches, action_dim)
        Q_full = self.W_q(X)
        K_full = self.W_k(Z)
        V_full = self.W_v(Z)

        #reshape (B,T,D_model) -> (B,T,H,D_k) -> (B,H,T,D_k) tranpose 1,2 just swaps them around 
        Q_parallel = Q_full.view(B, T, self.H, self.D_k).transpose(1,2)
        K_parallel = K_full.view(B, T_z, self.H, self.D_k).transpose(1, 2)
        V_parallel = V_full.view(B, T_z, self.H, self.D_k).transpose(1, 2) 

        attn_out = self.ScaledDotProductAttention(Q_parallel, K_parallel, V_parallel) # (B, H, T, D_k)
        # (B, H, T, D_k) needs to be # (B, T, D_model) 
        attn_out = attn_out.transpose(1,2).contiguous().view(B,T,D_model)
        
        output = self.W_o(attn_out)

        return output 
    

    def ScaledDotProductAttention(self, Q, K, V): 
        # Q_x: (B, chunk_length+1, action_dim)
        # Z_k: (B, num_patches, action_dim)
        # Z_v: (B, num_patches, action_dim)

        B, H, T, D_k = Q.shape  
        S = (Q @ K.transpose(2,3)) * pow(D_k, -0.5)

        attention_weights = torch.softmax(S, dim=-1) #softmax is applied to all the keys in each query row
        output = attention_weights @ V #(chunk + 1, action_dim) 

        return output 
import torch 
from torch import nn  
from video_idm.config import *  



class MultiHeadAttention(nn.Module): 
    def __init__(self):
        super().__init__()
        self.H = n_heads
        self.D_model = D_model 
        self.D_k = D_model // self.H 

        self.W_q = nn.Linear(D_model, D_model, bias=False)
        self.W_k = nn.Linear(D_model, D_model, bias=False)
        self.W_v = nn.Linear(D_model, D_model, bias=False)
        self.W_o = nn.Linear(D_model, D_model, bias=False)


    def forward(self, X):
        B, T, D_model = X.shape 
        
        Q_full = self.W_q(X)
        K_full = self.W_k(X)
        V_full = self.W_v(X)

        #reshape (B,T,D_model) -> (B,T,H,D_k) -> (B,H,T,D_k) tranpose 1,2 just swaps them around 
        Q_parallel = Q_full.view(B, T, self.H, self.D_k).transpose(1,2)
        K_parallel = K_full.view(B, T, self.H, self.D_k).transpose(1, 2)
        V_parallel = V_full.view(B, T, self.H, self.D_k).transpose(1, 2) 

        attn_out = self.ScaledDotProductAttention(Q_parallel, K_parallel, V_parallel) # (B, H, T, D_k)
        # (B, H, T, D_k) needs to be # (B, T, D_model) 
        attn_out = attn_out.transpose(1,2).contiguous().view(B,T,D_model)
        
        output = self.W_o(attn_out)

        return output 
        

    
    def ScaledDotProductAttention(self, Q, K, V): 
        B, H, T, D_k = Q.shape  
        S = (Q @ K.transpose(2,3)) * pow(D_k, -0.5)
        mask = torch.triu(torch.full(size=(T,T),fill_value=float('-inf')), diagonal=1)
        mask = mask.to(device)
        S_masked = S + mask 

        attention_weights = torch.softmax(S_masked, dim=-1) #softmax is applied to all the keys in each query row
        output = attention_weights @ V # (B, H, T, D_k) 

        return output 
        

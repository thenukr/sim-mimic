import torch 
from config import *

#z latent -> (B, 16, T, H, W) -> (B, T, H, W, 16) -> (B, T * H * W , 16)
# -> (B, num_patches, 16) -> *(16, action_dim) -> (B, num_patches, action_dim)

class VAELatentProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_proj = nn.Linear(16, action_dim)
    
    def forward(self, VAE_latent):
        B, C, T, H, W = VAE_latent.shape
        VAE_latent = VAE_latent.permute(0,2,3,4,1) 
        VAE_latent = VAE_latent.reshape(B, T * H * W, C) 
        
        return self.latent_proj(VAE_latent)
    



    

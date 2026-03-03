import torch
from torch import nn

from .action_encoder import ActionChunkEncoder, ProprioceptionEncoder
from .config import action_dim, chunk_length, dof, n_layers
from .positional_embedding import PositionalEmbedding
from .transformer.tau_embedding import TauEmbedding
from .transformer.dit_block import DiTBlock
from .conditioning_projector import VAELatentProjection

class InverseDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = ActionChunkEncoder()
        self.proprio_encoder = ProprioceptionEncoder()
        self.positional_embedding = PositionalEmbedding()
        self.tau_embedding = TauEmbedding()
        self.layers = nn.ModuleList([DiTBlock() for _ in range(n_layers)])
        self.vae_proj = VAELatentProjection()
        self.action_decoder = nn.Linear(action_dim, dof)

    def forward(self, noised_action_chunk, proprioception, VAE_latent, tau):
        # action chunk: B, length, dof -> enc -> B, length, action_dim
        # VAE latent proj: B, num_patches, action_dim 

        enc_action_chunk = self.action_encoder(noised_action_chunk)
        enc_proprio = self.proprio_encoder(proprioception)
        enc_action_toks = torch.cat([enc_proprio, enc_action_chunk], dim=1)
        X = self.positional_embedding(enc_action_toks)
        Z = self.vae_proj(VAE_latent)
        tau_embedding = self.tau_embedding(tau)

        for layer in self.layers: 
            X = layer(X, Z, tau_embedding)
        
        X = X[:,1:,:] # get rid of proprio     
        
        return self.action_decoder(X)
        


        




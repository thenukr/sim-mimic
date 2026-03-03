import torch
from model.idm import InverseDynamicsModel
from model.config import *

model = InverseDynamicsModel().to(device)

B = 2
noised_actions = torch.randn(B, chunk_length, dof).to(device)
proprio = torch.randn(B, dof).to(device)
vae_latent = torch.randn(B, 16, 2, 8, 8).to(device)
tau = torch.rand(B).to(device)

output = model(noised_actions, proprio, vae_latent, tau)
print(output.shape)  # should be (2, 8, 23)

loss = output.mean()
loss.backward()
print("backward pass ok")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")


import torch
from copy import deepcopy
from cosmos_tokenizer.networks import TokenizerConfigs, TokenizerModels

# 1) Build model
cfg = deepcopy(TokenizerConfigs["CV"].value)
cfg["temporal_compression"] = 4
cfg["spatial_compression"] = 8
vae = TokenizerModels[cfg["name"]].value(**cfg).float().eval()  # nn.Module

# 2) Load weights (from downloaded CV4x8x8 model.pt)
ckpt = torch.load(
    "checkpoints/nvidia/Cosmos-0.1-Tokenizer-CV4x8x8/model.pt",
    map_location="cpu",
    weights_only=False,
)["model"]
state_dict = {
    k.replace("network.", "", 1): v.float()
    for k, v in ckpt.items()
    if k.startswith("network.")
}
vae.load_state_dict(state_dict, strict=False)

# 3) Encode tensor -> latent
x = torch.randn(1, 3, 9, 256, 256).clamp(-1, 1)  # already in [-1, 1]
def vae_encode(x):
    with torch.no_grad():
        z, _posterior = vae.encode(x)   # z is latent
    return z # e.g. [1, 16, 3, 32, 32]: (B, C, time, height, width)


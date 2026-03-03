import torch
from torch import nn

from .config import action_dim, dof


class ActionChunkEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dof = dof
        self.dim = action_dim
        self.action_proj = nn.Linear(dof, action_dim)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        # (B, chunk_len, dof) -> (B, chunk_len, action_dim)
        return self.action_proj(action_chunk)


class ProprioceptionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dof = dof
        self.dim = action_dim
        self.proprio_proj = nn.Linear(dof, action_dim)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        # (B, dof) -> (B, 1, action_dim)
        embed = self.proprio_proj(proprio)
        return embed.unsqueeze(1)


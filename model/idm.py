import torch
from torch import nn

from .action_encoder import ActionChunkEncoder, ProprioceptionEncoder
from .config import action_dim, chunk_length, dof


class InverseDynamicsModel(nn.Module):
    s
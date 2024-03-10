import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

import torch
from torch import nn


class GeneralDetector(nn.Module):

  pool_size: int
  # Receptive Field size
  size: int

  def __init__(self, pool_size: int, size: int) -> None:
    super().__init__()


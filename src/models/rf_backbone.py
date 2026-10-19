from __future__ import annotations

import math

import torch
import torch.nn as nn


def choose_groups(channels: int, requested_groups: int) -> int:
  groups = min(int(requested_groups), int(channels))
  while groups > 1 and channels % groups != 0:
    groups -= 1
  return max(groups, 1)


class SinusoidalTimeEmbedding(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.dim = int(dim)

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    half = self.dim // 2
    if half < 1:
      return t[:, None]
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -scale)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < self.dim:
      emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResBlock3D(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, time_dim: int,
               norm_groups: int = 8, dropout: float = 0.0):
    super().__init__()
    self.norm1 = nn.GroupNorm(choose_groups(in_channels, norm_groups), in_channels)
    self.act1 = nn.SiLU(inplace=False)
    self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    self.time_proj = nn.Sequential(
      nn.SiLU(inplace=False),
      nn.Linear(time_dim, out_channels),
    )
    self.norm2 = nn.GroupNorm(choose_groups(out_channels, norm_groups), out_channels)
    self.act2 = nn.SiLU(inplace=False)
    self.dropout = nn.Dropout3d(float(dropout)) if dropout > 0 else nn.Identity()
    self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
    self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) \
      if in_channels != out_channels else nn.Identity()

  def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
    h = self.conv1(self.act1(self.norm1(x)))
    time_bias = self.time_proj(time_emb).to(dtype=h.dtype)
    h = h + time_bias[:, :, None, None, None]
    h = self.conv2(self.dropout(self.act2(self.norm2(h))))
    return h + self.skip(x)


class Downsample3D(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.op = nn.Conv3d(channels, channels, kernel_size=4, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.op(x)


class Upsample3D(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.op = nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.op(x)


class RFBackbone(nn.Module):
  """Time-conditioned 3D U-Net for rectified-flow velocity prediction."""

  def __init__(
      self,
      input_channels: int = 1,
      output_channels: int = 1,
      base_channels: int = 32,
      time_embedding_dim: int = 128,
      norm_groups: int = 8,
      dropout: float = 0.0):
    super().__init__()
    c = int(base_channels)
    tdim = int(time_embedding_dim)
    self.time_mlp = nn.Sequential(
      SinusoidalTimeEmbedding(tdim),
      nn.Linear(tdim, tdim),
      nn.SiLU(inplace=False),
      nn.Linear(tdim, tdim),
    )

    self.input = nn.Conv3d(input_channels, c, kernel_size=3, padding=1)
    self.enc1a = ResBlock3D(c, c, tdim, norm_groups, dropout)
    self.enc1b = ResBlock3D(c, c, tdim, norm_groups, dropout)
    self.down1 = Downsample3D(c)

    self.enc2a = ResBlock3D(c, c * 2, tdim, norm_groups, dropout)
    self.enc2b = ResBlock3D(c * 2, c * 2, tdim, norm_groups, dropout)
    self.down2 = Downsample3D(c * 2)

    self.mid1 = ResBlock3D(c * 2, c * 4, tdim, norm_groups, dropout)
    self.mid2 = ResBlock3D(c * 4, c * 4, tdim, norm_groups, dropout)

    self.up2 = Upsample3D(c * 4)
    self.dec2a = ResBlock3D(c * 4 + c * 2, c * 2, tdim, norm_groups, dropout)
    self.dec2b = ResBlock3D(c * 2, c * 2, tdim, norm_groups, dropout)

    self.up1 = Upsample3D(c * 2)
    self.dec1a = ResBlock3D(c * 2 + c, c, tdim, norm_groups, dropout)
    self.dec1b = ResBlock3D(c, c, tdim, norm_groups, dropout)

    self.out = nn.Sequential(
      nn.GroupNorm(choose_groups(c, norm_groups), c),
      nn.SiLU(inplace=False),
      nn.Conv3d(c, output_channels, kernel_size=3, padding=1),
    )

  @staticmethod
  def _match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] == target.shape[2:]:
      return source
    return torch.nn.functional.interpolate(
      source, size=target.shape[2:], mode="trilinear", align_corners=False)

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    time_emb = self.time_mlp(t)
    x1 = self.input(x)
    x1 = self.enc1b(self.enc1a(x1, time_emb), time_emb)
    x2 = self.down1(x1)
    x2 = self.enc2b(self.enc2a(x2, time_emb), time_emb)
    xm = self.down2(x2)
    xm = self.mid2(self.mid1(xm, time_emb), time_emb)

    y = self._match(self.up2(xm), x2)
    y = torch.cat([y, x2], dim=1)
    y = self.dec2b(self.dec2a(y, time_emb), time_emb)
    y = self._match(self.up1(y), x1)
    y = torch.cat([y, x1], dim=1)
    y = self.dec1b(self.dec1a(y, time_emb), time_emb)
    return self.out(y)

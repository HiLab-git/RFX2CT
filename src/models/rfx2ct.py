from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from src.models.rf_backbone import RFBackbone
from src.models.projectnet import ProjectNet


ARCHITECTURE = "rfx2ct_direct_alignment_v1"
LEVEL_NAMES = ("enc1", "enc2", "mid")


class DynamicGate(nn.Module):
  """Per-scale sigmoid weights computed only from the backbone time embedding."""

  def __init__(self, embedding_dim: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(embedding_dim, embedding_dim),
      nn.SiLU(),
      nn.Linear(embedding_dim, len(LEVEL_NAMES)),
    )
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, time_embedding: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(self.net(time_embedding))


class RFX2CT(nn.Module):
  """X-ray-conditioned RF reconstruction with three-scale direct injection."""

  def __init__(
      self,
      view_keys: Sequence[str],
      volume_size: int = 128,
      base_channels: int = 32,
      time_embedding_dim: int = 128,
      norm_groups: int = 8,
      dropout: float = 0.0,
      projectnet_channels: Sequence[int] = (4, 8, 16),
      view_angles: Optional[Mapping[str, float]] = None,
      view_detector_flips: Optional[Mapping[str, float]] = None,
  ):
    super().__init__()
    self.volume_size = int(volume_size)
    self.backbone = RFBackbone(
      input_channels=1, output_channels=1, base_channels=base_channels,
      time_embedding_dim=time_embedding_dim, norm_groups=norm_groups,
      dropout=dropout,
    )
    self.projectnet = ProjectNet(
      view_keys=view_keys,
      feature_channels=projectnet_channels, volume_size=volume_size,
      view_angles=view_angles, view_detector_flips=view_detector_flips,
    )
    self.adapters = nn.ModuleDict({
      level: nn.Conv3d(in_channels, base_channels * (2 ** index), 1)
      for index, (level, in_channels) in enumerate(
        zip(LEVEL_NAMES, self.projectnet.output_channels))
    })
    for adapter in self.adapters.values():
      nn.init.zeros_(adapter.weight)
      nn.init.zeros_(adapter.bias)
    self.gate = DynamicGate(time_embedding_dim)

  def gate_weights(self, t: torch.Tensor) -> torch.Tensor:
    return self.gate(self.backbone.time_mlp(t))

  def forward(
      self, x: torch.Tensor, t: torch.Tensor, xrays: torch.Tensor,
      spacing_xyz: torch.Tensor, ori_size: torch.Tensor,
  ) -> torch.Tensor:
    spatial_shapes = {
      level: tuple(int(size) // (2 ** index) for size in x.shape[-3:])
      for index, level in enumerate(LEVEL_NAMES)
    }
    condition = self.projectnet(xrays, spatial_shapes)
    time_embedding = self.backbone.time_mlp(t)
    gates = self.gate(time_embedding)

    def inject(feature: torch.Tensor, level: str, index: int) -> torch.Tensor:
      residual = self.adapters[level](condition[level])
      weight = gates[:, index].reshape(-1, 1, 1, 1, 1)
      return feature + weight * residual

    b = self.backbone
    x1 = b.enc1b(b.enc1a(b.input(x), time_embedding), time_embedding)
    x1 = inject(x1, "enc1", 0)
    x2 = b.enc2b(b.enc2a(b.down1(x1), time_embedding), time_embedding)
    x2 = inject(x2, "enc2", 1)
    xm = b.mid2(b.mid1(b.down2(x2), time_embedding), time_embedding)
    xm = inject(xm, "mid", 2)
    y = b._match(b.up2(xm), x2)
    y = b.dec2b(b.dec2a(torch.cat([y, x2], dim=1), time_embedding), time_embedding)
    y = b._match(b.up1(y), x1)
    y = b.dec1b(b.dec1a(torch.cat([y, x1], dim=1), time_embedding), time_embedding)
    return b.out(y)

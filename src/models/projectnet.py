from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_VIEW_ANGLES = {"xray1": 0.0, "xray2": 90.0}
# Stored lateral detector columns follow +y; alignment uses -y at 90°.
DEFAULT_DETECTOR_FLIPS = {"xray1": 1.0, "xray2": -1.0}


def _resize_2d(features: torch.Tensor, size: int) -> torch.Tensor:
  if tuple(features.shape[-2:]) == (size, size):
    return features
  return F.interpolate(
    features, size=(size, size), mode="bilinear", align_corners=False)


def align_to_ct_axes(
    volume: torch.Tensor,
    angle_degrees: torch.Tensor,
    detector_flip: torch.Tensor | None = None,
) -> torch.Tensor:
  """Align [ray, z, detector] features to CT axes [z, y, x]."""

  if volume.ndim != 5:
    raise ValueError("Expected volume [B,C,R,Z,U], got {}".format(
      tuple(volume.shape)))
  batch, _, ray_size, z_size, u_size = volume.shape
  if len({ray_size, z_size, u_size}) != 1:
    raise ValueError("Alignment expects cubic ray-depth volumes, "
                     "got {}".format(tuple(volume.shape[-3:])))
  device = volume.device
  dtype = volume.dtype
  angle_degrees = angle_degrees.to(device=device, dtype=dtype).reshape(-1)
  if angle_degrees.numel() == 1:
    angle_degrees = angle_degrees.expand(batch)
  if angle_degrees.numel() != batch:
    raise ValueError("Expected {} angle values, got {}".format(
      batch, angle_degrees.numel()))
  if detector_flip is None:
    detector_flip = torch.ones_like(angle_degrees)
  else:
    detector_flip = detector_flip.to(device=device, dtype=dtype).reshape(-1)
    if detector_flip.numel() == 1:
      detector_flip = detector_flip.expand(batch)
    if detector_flip.numel() != batch:
      raise ValueError("Expected {} detector flips, got {}".format(
        batch, detector_flip.numel()))

  size = int(ray_size)
  d = _normalized_axis(size, device, dtype)
  h = _normalized_axis(size, device, dtype)
  w = _normalized_axis(size, device, dtype)
  zz, yy, xx = torch.meshgrid(d, h, w, indexing="ij")
  theta = angle_degrees * (math.pi / 180.0)
  sin_t = torch.sin(theta).view(batch, 1, 1, 1)
  cos_t = torch.cos(theta).view(batch, 1, 1, 1)
  flip = detector_flip.view(batch, 1, 1, 1)

  ray = xx.unsqueeze(0) * sin_t + yy.unsqueeze(0) * cos_t
  detector_u = flip * (xx.unsqueeze(0) * cos_t - yy.unsqueeze(0) * sin_t)
  detector_z = zz.unsqueeze(0).expand(batch, -1, -1, -1)
  grid = torch.stack([detector_u, detector_z, ray], dim=-1)
  return F.grid_sample(
    volume,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
  )


def _normalized_axis(size: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
  return torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)


class XRayFeaturePyramid2D(nn.Module):
  def __init__(self, channels: Sequence[int], norm_groups: int = 4):
    super().__init__()
    if len(channels) != 3:
      raise ValueError("Expected three feature channel levels.")
    c1, c2, c3 = (int(value) for value in channels)

    def block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
      groups = min(int(norm_groups), out_channels)
      while groups > 1 and out_channels % groups != 0:
        groups -= 1
      return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(max(groups, 1), out_channels),
        nn.SiLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(max(groups, 1), out_channels),
        nn.SiLU(inplace=False),
      )

    self.level1 = block(1, c1, 1)
    self.level2 = block(c1, c2, 2)
    self.level3 = block(c2, c3, 2)

  def forward(self, xray: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    level1 = self.level1(xray)
    level2 = self.level2(level1)
    level3 = self.level3(level2)
    return level1, level2, level3


class ProgressiveRayDepthProjector(nn.Module):
  """Learn a content-dependent feature distribution along each X-ray ray."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      ray_depth: int,
      norm_groups: int = 4):
    super().__init__()
    self.out_channels = int(out_channels)
    self.ray_depth = int(ray_depth)
    if self.out_channels <= 0 or self.ray_depth <= 0:
      raise ValueError("out_channels and ray_depth must be positive")

    depths = []
    for candidate in (
        self.ray_depth // 8,
        self.ray_depth // 4,
        self.ray_depth // 2,
        self.ray_depth,
    ):
      candidate = max(int(candidate), 1)
      if not depths or candidate != depths[-1]:
        depths.append(candidate)

    def block(input_channels: int, output_channels: int) -> nn.Sequential:
      groups = min(int(norm_groups), output_channels)
      while groups > 1 and output_channels % groups != 0:
        groups -= 1
      return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.GroupNorm(max(groups, 1), output_channels),
        nn.SiLU(inplace=False),
      )

    self.progression = nn.ModuleList([
      block(input_channels, output_channels)
      for input_channels, output_channels in zip(
        (int(in_channels),) + tuple(depths[:-1]), depths)
    ])
    self.to_volume = nn.Conv2d(
      depths[-1], self.out_channels * self.ray_depth, 3, padding=1)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    for block in self.progression:
      features = block(features)
    features = self.to_volume(features)
    batch, _, detector_0, detector_1 = features.shape
    return features.reshape(
      batch, self.out_channels, self.ray_depth, detector_0, detector_1)


def _make_projectors(
    feature_channels: Sequence[int],
    volume_size: int,
) -> nn.ModuleList:
  sizes = (int(volume_size), int(volume_size) // 2, int(volume_size) // 4)
  if any(size <= 0 for size in sizes):
    raise ValueError("volume_size must be at least 4, got {}".format(volume_size))
  return nn.ModuleList([
    ProgressiveRayDepthProjector(channels, channels, size)
    for channels, size in zip(feature_channels, sizes)
  ])


class ProjectNet(nn.Module):
  """Shared X-ray encoder and depth lifting, followed by angle alignment."""

  def __init__(
      self,
      view_keys: Sequence[str],
      feature_channels: Sequence[int] = (4, 8, 16),
      volume_size: int = 128,
      view_angles: Mapping[str, float] | None = None,
      view_detector_flips: Mapping[str, float] | None = None):
    super().__init__()
    self.view_keys = tuple(str(key) for key in view_keys)
    self.feature_channels = tuple(int(value) for value in feature_channels)
    self.volume_size = int(volume_size)
    self.view_angles = dict(DEFAULT_VIEW_ANGLES)
    if view_angles:
      self.view_angles.update({str(key): float(value)
                               for key, value in view_angles.items()})
    self.view_detector_flips = dict(DEFAULT_DETECTOR_FLIPS)
    if view_detector_flips:
      self.view_detector_flips.update({
        str(key): float(value) for key, value in view_detector_flips.items()
      })
    if not self.view_keys:
      raise ValueError("At least one X-ray view key is required")
    if len(set(self.view_keys)) != len(self.view_keys):
      raise ValueError("Duplicate view keys are not allowed")
    missing_angles = sorted(set(self.view_keys) - set(self.view_angles))
    if missing_angles:
      raise ValueError("Missing view angles for {}".format(missing_angles))
    self.encoder = XRayFeaturePyramid2D(self.feature_channels)
    self.projectors = _make_projectors(
      self.feature_channels, self.volume_size)

  @property
  def output_channels(self) -> Tuple[int, int, int]:
    return tuple(len(self.view_keys) * channels
                 for channels in self.feature_channels)

  def forward(
      self,
      xrays: torch.Tensor,
      spatial_shapes: Mapping[str, Sequence[int]],
  ) -> Dict[str, torch.Tensor]:
    if xrays.ndim != 4 or xrays.shape[1] != len(self.view_keys):
      raise ValueError("Expected xrays [B,{},H,W], got {}".format(
        len(self.view_keys), tuple(xrays.shape)))
    pyramids = {
      view_key: self.encoder(xrays[:, index:index + 1])
      for index, view_key in enumerate(self.view_keys)
    }
    conditions = {}
    for level_index, level_name in enumerate(("enc1", "enc2", "mid")):
      shape = tuple(int(value) for value in spatial_shapes[level_name])
      expected_size = self.volume_size // (2 ** level_index)
      if shape != (expected_size,) * 3:
        raise ValueError("ProjectNet expects {} at {}, got {}".format(
          (expected_size,) * 3, level_name, shape))
      lifted_views = []
      for view_key in self.view_keys:
        features = _resize_2d(pyramids[view_key][level_index], expected_size)
        lifted = self.projectors[level_index](features)
        angles = features.new_full(
          (features.shape[0],), self.view_angles[view_key])
        detector_flip = features.new_full(
          (features.shape[0],), self.view_detector_flips.get(view_key, 1.0))
        lifted_views.append(align_to_ct_axes(lifted, angles, detector_flip))
      conditions[level_name] = torch.cat(lifted_views, dim=1)
    return conditions

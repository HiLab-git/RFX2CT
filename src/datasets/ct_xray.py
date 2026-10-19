from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_split_file(path: str | Path) -> list[str]:
  with Path(path).open("r", encoding="utf-8") as handle:
    return [line.strip() for line in handle if line.strip()]


def read_h5_float32(hdf5: h5py.File, key: str) -> np.ndarray:
  if key not in hdf5:
    raise KeyError("Missing HDF5 key: {}".format(key))
  dataset = hdf5[key]
  array = np.empty(dataset.shape, dtype=np.float32)
  try:
    dataset.read_direct(array)
  except Exception:
    array = np.asarray(dataset.astype("float32")[()], dtype=np.float32)
  return array


def normalize_ct_zero_one(ct: np.ndarray, ct_min: float, ct_max: float) -> np.ndarray:
  ct = np.clip(ct.astype(np.float32, copy=False), ct_min, ct_max)
  return (ct - ct_min) / max(ct_max - ct_min, 1e-6)


def resize_ct_tensor(ct: torch.Tensor, size: int) -> torch.Tensor:
  if tuple(ct.shape[-3:]) == (size, size, size):
    return ct
  return F.interpolate(
    ct.unsqueeze(0), size=(size, size, size), mode="trilinear",
    align_corners=False).squeeze(0)


class XRayCTDataset(Dataset):
  """Paired CT volumes and X-rays stored in one HDF5 file per case.

  CT tensors use [C,D,H,W] = [C,z,y,x]. X-rays keep their stored detector
  orientation; ProjectNet aligns each view in volume space.
  """

  def __init__(
      self,
      dataroot: str | Path,
      split_file: str | Path,
      xray_keys: Sequence[str],
      ct_key: str = "ct",
      volume_size: int = 128,
      ct_min: float = 0.0,
      ct_max: float = 2500.0):
    self.dataroot = Path(dataroot)
    self.samples = read_split_file(split_file)
    self.xray_keys = tuple(str(key) for key in xray_keys)
    self.ct_key = str(ct_key)
    self.volume_size = int(volume_size)
    self.ct_min = float(ct_min)
    self.ct_max = float(ct_max)
    if not self.samples:
      raise RuntimeError("Empty split file: {}".format(split_file))
    if self.volume_size <= 0:
      raise ValueError("volume_size must be positive, got {}".format(
        self.volume_size))
    if self.ct_max <= self.ct_min:
      raise ValueError("ct_max must be greater than ct_min")
    if not self.xray_keys:
      raise ValueError("At least one X-ray key is required.")
    if len(set(self.xray_keys)) != len(self.xray_keys):
      raise ValueError("Duplicate X-ray keys are not allowed: {}".format(
        self.xray_keys))

  def __len__(self) -> int:
    return len(self.samples)

  def h5_path(self, sample_id: str) -> Path:
    return self.dataroot / sample_id / "ct_xray_data.h5"

  def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
    sample_id = self.samples[index]
    path = self.h5_path(sample_id)
    if not path.is_file():
      raise FileNotFoundError(str(path))

    with h5py.File(path, "r") as hdf5:
      extra_views = [key for key in hdf5 if key.startswith("xray")
                     and key[4:].isdigit() and key not in ("xray1", "xray2")]
      if "xray2" in self.xray_keys and extra_views:
        raise ValueError(
          "{}: expected a two-view HDF5 file with xray1=frontal and "
          "xray2=lateral; an angle-indexed multi-view file cannot be used "
          "as an orthogonal pair".format(path))
      ct = read_h5_float32(hdf5, self.ct_key)
      if ct.ndim != 3:
        raise ValueError("{}: expected {} to be 3D, got {}".format(
          path, self.ct_key, ct.shape))
      xrays = [read_h5_float32(hdf5, key) for key in self.xray_keys]
      spacing = read_h5_float32(hdf5, "spacing") \
        if "spacing" in hdf5 else np.ones(3, dtype=np.float32)
      ori_size_array = read_h5_float32(hdf5, "ori_size") \
        if "ori_size" in hdf5 else np.asarray(
          ct.shape[0], dtype=np.float32)

    if tuple(spacing.shape) != (3,):
      raise ValueError("{}: expected spacing shape (3,), got {}".format(
        path, spacing.shape))
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
      raise ValueError("{}: spacing must contain three finite positive values".format(
        path))
    for key, xray in zip(self.xray_keys, xrays):
      if xray.ndim != 2:
        raise ValueError("{}: expected {} to be 2D, got {}".format(
          path, key, xray.shape))
      if xray.shape[0] != xray.shape[1]:
        raise ValueError("{}: expected {} to be square, got {}".format(
          path, key, xray.shape))

    xray_tensor = torch.stack([
      torch.from_numpy(np.clip(xray, 0.0, 255.0) / 255.0)
      for xray in xrays
    ], dim=0)
    ori_size_flat = np.asarray(ori_size_array).reshape(-1)
    if ori_size_flat.size != 1:
      raise ValueError("{}: expected scalar ori_size, got {}".format(
        path, ori_size_array.shape))
    ori_size = float(ori_size_flat[0])
    if not np.isfinite(ori_size) or ori_size <= 0:
      raise ValueError("{}: invalid ori_size={}".format(path, ori_size))

    real = normalize_ct_zero_one(ct, self.ct_min, self.ct_max)
    return {
      "real_ct": resize_ct_tensor(
        torch.from_numpy(real).unsqueeze(0), self.volume_size).contiguous(),
      "xrays": xray_tensor.contiguous(),
      "spacing": torch.from_numpy(spacing.astype(np.float32, copy=False)),
      "ori_size": torch.tensor(ori_size, dtype=torch.float32),
      "sample_id": sample_id,
      "h5_path": str(path),
      "xray_keys": ",".join(self.xray_keys),
    }

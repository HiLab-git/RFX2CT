from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEPTH_ORDER_ARRAY = "array"
DEPTH_ORDER_X2CT = "x2ct"
VALID_DEPTH_ORDERS = {DEPTH_ORDER_ARRAY, DEPTH_ORDER_X2CT}


def validate_depth_order(depth_order: str) -> str:
  value = str(depth_order or DEPTH_ORDER_ARRAY).lower()
  if value not in VALID_DEPTH_ORDERS:
    raise ValueError("Unsupported MHA depth order {!r}; expected one of {}".format(
      depth_order, sorted(VALID_DEPTH_ORDERS)))
  return value


def apply_mha_depth_order_for_export(volume: np.ndarray, depth_order: str) -> np.ndarray:
  depth_order = validate_depth_order(depth_order)
  if depth_order == DEPTH_ORDER_X2CT:
    return np.ascontiguousarray(volume[::-1, :, :])
  return np.ascontiguousarray(volume)


def write_mha_hu(volume_hu: np.ndarray, path: str | Path,
                 depth_order: str = DEPTH_ORDER_ARRAY) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  export_volume = apply_mha_depth_order_for_export(volume_hu, depth_order)
  image = sitk.GetImageFromArray(export_volume.astype(np.int32, copy=False))
  image.SetSpacing((1.0, 1.0, 1.0))
  image.SetOrigin((0.0, 0.0, 0.0))
  sitk.WriteImage(image, str(path))


def normalized_zero_one_to_hu(volume: np.ndarray, ct_min: float = 0.0,
                              ct_max: float = 2500.0) -> np.ndarray:
  volume = np.clip(volume.astype(np.float32, copy=False), 0.0, 1.0)
  return (volume * (float(ct_max) - float(ct_min)) + float(ct_min) - 1024.0)

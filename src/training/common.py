"""Training state, EMA and volume I/O shared by RFX2CT train and export."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.models.rfx2ct import ARCHITECTURE
from src.utils.volume_io import normalized_zero_one_to_hu, write_mha_hu

TIME_CONVENTION = "noise_to_ct"


def seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed % (2 ** 32))
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
  return model.module if isinstance(model, DistributedDataParallel) else model


class EMA:
  def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
    if not 0.0 <= decay < 1.0:
      raise ValueError("EMA decay must be in [0, 1)")
    self.decay = float(decay)
    self.shadow = {
      key: value.detach().cpu().clone()
      for key, value in unwrap_model(model).state_dict().items()
    }

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    for key, value in unwrap_model(model).state_dict().items():
      value_cpu = value.detach().cpu()
      if torch.is_floating_point(self.shadow[key]):
        self.shadow[key].mul_(self.decay).add_(value_cpu, alpha=1.0 - self.decay)
      else:
        self.shadow[key].copy_(value_cpu)

  def state_dict(self) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in self.shadow.items()}

  def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
    if set(state) != set(self.shadow):
      raise RuntimeError("EMA keys do not match the model")
    for key, value in state.items():
      if value.shape != self.shadow[key].shape:
        raise RuntimeError("EMA tensor shape mismatch: " + key)
    self.shadow = {key: value.detach().cpu().clone() for key, value in state.items()}


def capture_rng_state() -> Dict:
  state = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state()}
  if torch.cuda.is_available():
    state["cuda"] = torch.cuda.get_rng_state_all()
  return state


def restore_rng_state(state: Dict) -> None:
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  if "cuda" in state and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["cuda"])


def validate_checkpoint(checkpoint: Dict) -> None:
  if checkpoint.get("architecture") != ARCHITECTURE:
    raise RuntimeError(
      "Checkpoint does not match the RFX2CT architecture (alignment-only ProjectNet). "
      "Use a checkpoint trained with this RFX2CT implementation.")
  if checkpoint.get("time_convention") != TIME_CONVENTION:
    raise RuntimeError("Checkpoint time convention is not noise_to_ct")


def load_checkpoint(path: str | Path) -> Dict:
  # Full resume state includes Python/NumPy objects, not just tensor weights.
  try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  except TypeError:  # PyTorch 1.13 has no weights_only argument.
    checkpoint = torch.load(path, map_location="cpu")
  validate_checkpoint(checkpoint)
  return checkpoint


def resume_signature(cfg: Dict) -> Dict:
  """Changes to training semantics need a new run instead of a silent resume."""
  result = copy.deepcopy(cfg)
  result.pop("output", None)
  result.pop("validation", None)
  train = result.get("train", {})
  for key in ("epochs", "save_epoch_freq", "log_freq", "num_workers"):
    train.pop(key, None)
  return result


def save_checkpoint(path: Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scaler,
                    epoch: int, global_step: int, cfg: Dict,
                    ema: Optional[EMA]) -> None:
  """All ranks call this; rank zero writes a new immutable epoch checkpoint."""
  distributed = dist.is_available() and dist.is_initialized()
  rank = dist.get_rank() if distributed else 0
  world_size = dist.get_world_size() if distributed else 1
  rng_states = [None] * world_size
  own_state = capture_rng_state()
  if distributed:
    dist.all_gather_object(rng_states, own_state)
  else:
    rng_states[0] = own_state
  if rank != 0:
    return
  if path.exists():
    raise FileExistsError("Refusing to overwrite checkpoint: " + str(path))
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "architecture": ARCHITECTURE, "time_convention": TIME_CONVENTION,
    "epoch": int(epoch), "global_step": int(global_step),
    "model": unwrap_model(model).state_dict(),
    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
    "config": copy.deepcopy(cfg), "rng_states": rng_states,
    "world_size": world_size,
  }
  if ema is not None:
    payload["ema_model"] = ema.state_dict()
    payload["ema_decay"] = ema.decay
  temporary = path.with_suffix(".tmp")
  torch.save(payload, temporary)
  os.replace(temporary, path)
  print("saved {}".format(path), flush=True)


def save_mha(volume: np.ndarray, path: Path, cfg: Dict) -> None:
  data = cfg["data"]
  hu = normalized_zero_one_to_hu(
    volume, ct_min=float(data.get("ct_min", 0.0)),
    ct_max=float(data.get("ct_max", 2500.0)))
  write_mha_hu(hu, path, depth_order=cfg.get("validation", {}).get(
    "mha_depth_order", "x2ct"))

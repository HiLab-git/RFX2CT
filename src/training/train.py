"""Train RFX2CT with endpoint-consistent Self-Rollout."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.ct_xray import XRayCTDataset, read_split_file
from src.models.rfx2ct import RFX2CT
from src.training.common import (
  ARCHITECTURE, TIME_CONVENTION, EMA, load_checkpoint, resume_signature,
  restore_rng_state, save_checkpoint, save_mha, seed_everything,
)
from src.training.flow import (
  euler_solve, sample_rf_batch, sample_self_rollout_batch,
)
from src.utils.config import load_config, save_config, write_json


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", required=True)
  parser.add_argument("--dataroot", default="")
  parser.add_argument("--train-split", default="")
  parser.add_argument("--val-split", default="")
  parser.add_argument("--out-dir", default="")
  parser.add_argument("--tag", default="")
  parser.add_argument("--resume", default="")
  parser.add_argument("--device", default="")
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--batch-size", type=int)
  parser.add_argument("--grad-accum-steps", type=int)
  parser.add_argument("--no-sample", action="store_true")
  return parser.parse_args(argv)


def apply_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
  cfg = copy.deepcopy(cfg)
  for name in ("dataroot", "train_split", "val_split"):
    if getattr(args, name):
      cfg.setdefault("data", {})[name] = getattr(args, name)
  for name in ("out_dir", "tag"):
    if getattr(args, name):
      cfg.setdefault("output", {})[name] = getattr(args, name)
  for name in ("epochs", "batch_size", "grad_accum_steps"):
    if getattr(args, name) is not None:
      cfg.setdefault("train", {})[name] = getattr(args, name)
  if args.no_sample:
    cfg.setdefault("validation", {})["sample_every_epochs"] = 0
  return cfg


def validate_config(cfg: Dict) -> None:
  if cfg.get("model", {}).get("architecture") != ARCHITECTURE:
    raise ValueError("model.architecture must be " + ARCHITECTURE)
  if cfg.get("rectified_flow", {}).get("time_convention") != TIME_CONVENTION:
    raise ValueError("rectified_flow.time_convention must be noise_to_ct")
  train_cfg = cfg.get("train", {})
  for key in ("epochs", "batch_size", "grad_accum_steps"):
    if int(train_cfg.get(key, 1)) <= 0:
      raise ValueError("train.{} must be positive".format(key))
  if int(train_cfg.get("log_freq", 20)) <= 0:
    raise ValueError("train.log_freq must be positive")
  rollout = cfg.get("self_rollout", {})
  if not 0.0 <= float(rollout.get("probability", 0.1)) <= 1.0:
    raise ValueError("self_rollout.probability must be in [0, 1]")
  if int(rollout.get("warmup_epochs", 20)) < 0:
    raise ValueError("self_rollout.warmup_epochs must be nonnegative")
  if int(rollout.get("steps", 2)) <= 0:
    raise ValueError("self_rollout.steps must be positive")
  if not 0.0 <= float(rollout.get("min_t", 0.2)) < float(
      rollout.get("max_t", 0.9)) < 1.0:
    raise ValueError("Self-Rollout requires 0 <= min_t < max_t < 1")


def validate_splits(cfg: Dict) -> Dict[str, str]:
  """Reject leakage and fingerprint the three explicit fixed split files."""
  sets = {}
  fingerprints = {}
  for key in ("train_split", "val_split", "test_split"):
    path = cfg["data"].get(key)
    if not path:
      raise ValueError("Missing data.{}; independent splits are required".format(key))
    values = read_split_file(path)
    if not values or len(values) != len(set(values)):
      raise ValueError("{} is empty or contains duplicate case IDs".format(key))
    sets[key] = set(values)
    fingerprints[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
  for first, second in (("train_split", "val_split"), ("train_split", "test_split"),
                         ("val_split", "test_split")):
    overlap = sets[first] & sets[second]
    if overlap:
      raise ValueError("Overlapping {} / {}: {}".format(first, second, sorted(overlap)[:3]))
  return fingerprints


def make_model(cfg: Dict) -> RFX2CT:
  model = cfg["model"]
  projectnet = cfg["projectnet"]
  return RFX2CT(
    view_keys=cfg["data"]["xray_keys"],
    volume_size=int(cfg["data"].get("volume_size", 128)),
    base_channels=int(model.get("base_channels", 32)),
    time_embedding_dim=int(model.get("time_embedding_dim", 128)),
    norm_groups=int(model.get("norm_groups", 8)),
    dropout=float(model.get("dropout", 0.0)),
    projectnet_channels=projectnet.get("feature_channels", [4, 8, 16]),
    view_angles=projectnet["view_angles"],
    view_detector_flips=projectnet["view_detector_flips"],
  )


def make_dataset(cfg: Dict, split_key: str) -> XRayCTDataset:
  data = cfg["data"]
  return XRayCTDataset(
    dataroot=data["dataroot"], split_file=data[split_key],
    xray_keys=data["xray_keys"], ct_key=data.get("ct_key", "ct"),
    volume_size=int(data.get("volume_size", 128)),
    ct_min=float(data.get("ct_min", 0.0)), ct_max=float(data.get("ct_max", 2500.0)),
  )


def move_batch(batch: Dict, device: torch.device):
  return tuple(batch[key].to(device, non_blocking=True)
               for key in ("real_ct", "xrays", "spacing", "ori_size"))


@torch.no_grad()
def sample_ct(model: torch.nn.Module, xrays: torch.Tensor,
                        spacing: torch.Tensor, ori_size: torch.Tensor,
                        volume_size: int, steps: int = 25,
                        clamp_sample: bool = True, amp_enabled: bool = False,
                        noise: Optional[torch.Tensor] = None,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
  if noise is None:
    noise = torch.randn((xrays.shape[0], 1, volume_size, volume_size, volume_size),
                        device=xrays.device, dtype=xrays.dtype, generator=generator)
  def velocity(state, t):
    with autocast(enabled=amp_enabled and state.device.type == "cuda"):
      return model(state, t, xrays, spacing, ori_size)
  sample = euler_solve(velocity, noise, steps=steps)
  return sample.clamp(0.0, 1.0) if clamp_sample else sample


def _append_csv(path: Path, row: Dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  exists = path.exists()
  with path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    if not exists:
      writer.writeheader()
    writer.writerow(row)


def check_run_directory(run_root: Path, resume: str, checkpoint) -> None:
  if not run_root.exists() or not any(run_root.iterdir()):
    return
  if checkpoint is None:
    raise FileExistsError("Output already contains files; use a new tag or --resume: " + str(run_root))
  if Path(resume).resolve().parent != (run_root / "checkpoints").resolve():
    raise FileExistsError("Resume target belongs to another run; use an empty output directory")
  epoch = int(checkpoint["epoch"])
  for folder, pattern in (("checkpoints", "epoch_*.pt"), ("validation", "epoch_*")):
    for path in (run_root / folder).glob(pattern):
      suffix = path.stem.removeprefix("epoch_")
      if suffix.isdigit() and int(suffix) > epoch:
        raise FileExistsError("Output contains epochs after the resume checkpoint; use a new tag")
  log_path = run_root / "logs" / "train.csv"
  if log_path.exists():
    with log_path.open(newline="", encoding="utf-8") as handle:
      if any(int(row["epoch"]) > epoch for row in csv.DictReader(handle)):
        raise FileExistsError("Training log extends past the resume checkpoint; use a new tag")


@torch.no_grad()
def export_validation_samples(model, ema, cfg, run_root, epoch, device) -> None:
  validation = cfg.get("validation", {})
  dataset = make_dataset(cfg, "val_split")
  count = min(int(validation.get("sample_count", 5)), len(dataset))
  if count <= 0:
    return
  use_ema = bool(validation.get("use_ema", True)) and ema is not None
  saved = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
  was_training = model.training
  try:
    if use_ema:
      model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    for index in range(count):
      item = dataset[index]
      xrays = item["xrays"].unsqueeze(0).to(device)
      spacing = item["spacing"].unsqueeze(0).to(device)
      ori_size = item["ori_size"].unsqueeze(0).to(device)
      generator = torch.Generator(device=device).manual_seed(
        int(validation.get("seed", cfg.get("experiment", {}).get("seed", 20260629))) + index)
      fake = sample_ct(
        model, xrays, spacing, ori_size, int(cfg["data"].get("volume_size", 128)),
        int(validation.get("sample_steps", 25)),
        bool(cfg.get("rectified_flow", {}).get("clamp_sample", True)),
        bool(cfg.get("train", {}).get("amp", True)), generator=generator)
      folder = run_root / "validation" / "epoch_{:04d}".format(epoch) / str(item["sample_id"])
      save_mha(fake[0, 0].cpu().numpy(), folder / "fake_ct.mha", cfg)
      save_mha(item["real_ct"][0].numpy(), folder / "real_ct.mha", cfg)
  finally:
    model.load_state_dict(saved, strict=True)
    model.train(was_training)


def train(cfg: Dict, resume: str = "", device_override: str = "") -> Dict:
  cfg = copy.deepcopy(cfg)
  validate_config(cfg)
  cfg["data"]["split_sha256"] = validate_splits(cfg)
  distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
  rank = int(os.environ.get("RANK", "0")) if distributed else 0
  local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
  world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
  requested = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
  if requested.type == "cuda":
    device = torch.device("cuda", local_rank if distributed else (requested.index or 0))
    torch.cuda.set_device(device)
  else:
    device = requested
  if distributed:
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
  try:
    return _train(cfg, resume, device, distributed, rank, world_size)
  finally:
    if distributed and dist.is_initialized():
      dist.destroy_process_group()


def _train(cfg, resume, device, distributed, rank, world_size):
  seed = int(cfg.get("experiment", {}).get("seed", 20260629))
  seed_everything(seed + rank)
  train_cfg = cfg.get("train", {})
  run_root = Path(cfg.get("output", {}).get("out_dir", "checkpoints")) / cfg.get(
    "output", {}).get("tag", "rfx2ct")
  checkpoint = load_checkpoint(resume) if resume else None
  epochs = int(train_cfg.get("epochs", 200))
  check_run_directory(run_root, resume, checkpoint)
  if checkpoint is not None:
    if resume_signature(checkpoint["config"]) != resume_signature(cfg):
      raise ValueError("Resume configuration or split hashes changed; start a new run")
    if int(checkpoint.get("world_size", 1)) != world_size:
      raise ValueError("Resume requires the original world size for rank-specific RNG state")
    if int(checkpoint["epoch"]) >= epochs:
      raise ValueError("Checkpoint already reached requested epoch count")
  model = make_model(cfg).to(device)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=float(train_cfg.get("lr", 1e-4)),
    betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
    weight_decay=float(train_cfg.get("weight_decay", 0.0)))
  amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
  scaler = GradScaler(enabled=amp_enabled)
  ema = EMA(model, float(train_cfg.get("ema_decay", 0.9999))) if train_cfg.get("use_ema", True) else None
  start_epoch, global_step = 1, 0
  if checkpoint is not None:
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None:
      if "ema_model" not in checkpoint:
        raise ValueError("Resume checkpoint is missing EMA state")
      ema.load_state_dict(checkpoint["ema_model"])
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])
    restore_rng_state(checkpoint["rng_states"][rank])
  dataset = make_dataset(cfg, "train_split")
  sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if distributed else None
  loader_generator = torch.Generator()
  loader = DataLoader(
    dataset, batch_size=int(train_cfg.get("batch_size", 1)), sampler=sampler,
    shuffle=sampler is None, num_workers=int(train_cfg.get("num_workers", 4)),
    pin_memory=device.type == "cuda", drop_last=False, generator=loader_generator)
  raw_model = model
  if distributed:
    model = DistributedDataParallel(
      model, device_ids=[device.index] if device.type == "cuda" else None,
      broadcast_buffers=False)
  if rank == 0:
    run_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_root / "config_resolved.yml")
    write_json({"architecture": ARCHITECTURE, "time_convention": TIME_CONVENTION,
                "device": str(device), "world_size": world_size,
                "train_cases": len(dataset), "resume": resume,
                "torch": torch.__version__}, run_root / "run_info.json")
  if distributed:
    dist.barrier()
  accumulation = int(train_cfg.get("grad_accum_steps", 1))
  batches = len(loader)
  rollout_cfg = cfg.get("self_rollout", {})
  for epoch in range(start_epoch, epochs + 1):
    model.train()
    loader_generator.manual_seed(seed + epoch * world_size + rank)
    if sampler is not None:
      sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    epoch_loss, sample_count, rollout_batches = 0.0, 0, 0
    started = time.perf_counter()
    for index, batch in enumerate(loader):
      real, xrays, spacing, ori_size = move_batch(batch, device)
      use_rollout = torch.zeros((), device=device)
      if epoch > int(rollout_cfg.get("warmup_epochs", 20)):
        if rank == 0:
          use_rollout.copy_((torch.rand((), device=device) < float(
            rollout_cfg.get("probability", 0.1))).float())
        if distributed:
          dist.broadcast(use_rollout, src=0)
      if bool(use_rollout.item()):
        rollout_batches += 1
        def velocity(state, t):
          with autocast(enabled=amp_enabled):
            return raw_model(state, t, xrays, spacing, ori_size)
        state, t, target = sample_self_rollout_batch(
          real, velocity, int(rollout_cfg.get("steps", 2)),
          float(rollout_cfg.get("min_t", 0.2)), float(rollout_cfg.get("max_t", 0.9)))
      else:
        state, t, target = sample_rf_batch(real)
      window_start = (index // accumulation) * accumulation
      window_size = min(accumulation, batches - window_start)
      update_now = (index + 1) % accumulation == 0 or index + 1 == batches
      context = model.no_sync() if distributed and not update_now else nullcontext()
      with context:
        with autocast(enabled=amp_enabled):
          prediction = model(state, t, xrays, spacing, ori_size)
          loss = F.mse_loss(prediction.float(), target.float())
        if not torch.isfinite(loss):
          raise FloatingPointError("Nonfinite training loss at epoch {}, batch {}".format(epoch, index))
        scaler.scale(loss / window_size).backward()
      epoch_loss += float(loss.detach()) * real.shape[0]
      sample_count += real.shape[0]
      if update_now:
        scaler.unscale_(optimizer)
        grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
        if grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
          global_step += 1
          if ema is not None:
            ema.update(raw_model)
        optimizer.zero_grad(set_to_none=True)
      if rank == 0 and (index + 1) % int(train_cfg.get("log_freq", 20)) == 0:
        print("epoch={} batch={}/{} loss={:.6f} step={}".format(
          epoch, index + 1, batches, float(loss.detach()), global_step), flush=True)
    totals = torch.tensor([epoch_loss, sample_count, rollout_batches], dtype=torch.float64, device=device)
    if distributed:
      dist.all_reduce(totals)
    if rank == 0:
      row = {"epoch": epoch, "global_step": global_step,
             "loss": float(totals[0] / totals[1]), "samples": int(totals[1]),
             "rollout_batches": int(totals[2]),
             "training_seconds": time.perf_counter() - started}
      _append_csv(run_root / "logs" / "train.csv", row)
      print(row, flush=True)
      frequency = int(cfg.get("validation", {}).get("sample_every_epochs", 20))
      if frequency > 0 and epoch % frequency == 0:
        export_validation_samples(raw_model, ema, cfg, run_root, epoch, device)
    if distributed:
      dist.barrier()
    save_frequency = int(train_cfg.get("save_epoch_freq", 10))
    if epoch == epochs or (save_frequency > 0 and epoch % save_frequency == 0):
      save_checkpoint(run_root / "checkpoints" / "epoch_{:04d}.pt".format(epoch),
                      model, optimizer, scaler, epoch, global_step, cfg, ema)
      if distributed:
        dist.barrier()
  return {"epoch": epochs, "global_step": global_step, "run_root": str(run_root)}


def main(argv=None) -> None:
  args = parse_args(argv)
  train(apply_overrides(load_config(args.config), args), args.resume, args.device)


if __name__ == "__main__":
  main()

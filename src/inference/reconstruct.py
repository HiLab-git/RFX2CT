"""Reconstruct CT volumes from X-rays with RFX2CT."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import time
from pathlib import Path
from typing import Dict

import torch

from src.training.common import (
  ARCHITECTURE, TIME_CONVENTION, load_checkpoint, save_mha,
)
from src.training.train import (
  make_dataset, make_model, sample_ct, validate_config, validate_splits,
)
from src.utils.config import load_config, save_config, write_json


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--dataroot", default="")
  parser.add_argument("--test-split", default="")
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--device", default="")
  parser.add_argument("--sample-steps", type=int, default=0)
  parser.add_argument("--mha-depth-order", choices=["x2ct", "array"], default="")
  parser.add_argument("--seed", type=int, default=20260629)
  parser.add_argument("--no-ema", action="store_true")
  return parser.parse_args(argv)


def load_runtime_config(args: argparse.Namespace, checkpoint: Dict) -> Dict:
  cfg = load_config(args.config) if args.config else copy.deepcopy(checkpoint.get("config", {}))
  if not cfg:
    raise RuntimeError("Checkpoint has no embedded config; pass --config")
  embedded = checkpoint.get("config", {})
  for section in ("model", "projectnet"):
    if embedded and cfg.get(section) != embedded.get(section):
      raise ValueError("Export {} differs from checkpoint architecture".format(section))
  semantic_defaults = {"xray_keys": None, "volume_size": 128, "ct_min": 0.0,
                       "ct_max": 2500.0, "ct_key": "ct",
                       "value_space": "normalized_0_1", "tensor_axes": "array_DHW"}
  for key, default in semantic_defaults.items():
    if embedded and cfg.get("data", {}).get(key, default) != embedded.get("data", {}).get(key, default):
      raise ValueError("Export data.{} differs from checkpoint preprocessing".format(key))
  if args.dataroot:
    cfg["data"]["dataroot"] = args.dataroot
  if args.test_split:
    cfg["data"]["test_split"] = args.test_split
  if args.sample_steps > 0:
    cfg.setdefault("rectified_flow", {})["sample_steps"] = args.sample_steps
  if args.sample_steps < 0:
    raise ValueError("sample_steps must be positive")
  if args.mha_depth_order:
    cfg.setdefault("validation", {})["mha_depth_order"] = args.mha_depth_order
  validate_config(cfg)
  if not cfg["data"].get("test_split"):
    raise ValueError("Missing data.test_split or --test-split; validation is never used as test")
  cfg["data"]["split_sha256"] = validate_splits(cfg)
  return cfg


def case_noise_seed(base_seed: int, case_id: str) -> int:
  """Stable across split reordering; independent of a case's dataset index."""
  digest = hashlib.sha256("{}:{}".format(base_seed, case_id).encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def sync_device(device: torch.device) -> None:
  if device.type == "cuda":
    torch.cuda.synchronize(device)


def write_timing_outputs(out_root: Path, rows: list) -> None:
  if not rows:
    return
  with (out_root / "inference_times.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  times = torch.tensor([row["inference_seconds"] for row in rows], dtype=torch.float64)
  write_json({
    "num_cases": len(rows), "mean_inference_seconds_per_case": float(times.mean()),
    "median_inference_seconds_per_case": float(times.quantile(0.5)),
    "min_inference_seconds_per_case": float(times.min()),
    "max_inference_seconds_per_case": float(times.max()),
    "std_inference_seconds_per_case": float(times.std(unbiased=False)),
    "total_inference_seconds": float(times.sum()),
  }, out_root / "inference_time_summary.json")


@torch.no_grad()
def main(argv=None) -> None:
  args = parse_args(argv)
  checkpoint = load_checkpoint(args.checkpoint)
  cfg = load_runtime_config(args, checkpoint)
  device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
  model = make_model(cfg).to(device)
  use_ema = not args.no_ema
  if use_ema and "ema_model" not in checkpoint:
    raise ValueError("Checkpoint has no EMA weights; explicitly pass --no-ema to use raw weights")
  model.load_state_dict(checkpoint["ema_model"] if use_ema else checkpoint["model"], strict=True)
  model.eval()
  dataset = make_dataset(cfg, "test_split")
  out_root = Path(args.out_dir)
  if out_root.exists() and any(out_root.iterdir()):
    raise FileExistsError("Refusing to mix or overwrite an existing export: " + str(out_root))
  out_root.mkdir(parents=True, exist_ok=True)
  steps = int(cfg.get("rectified_flow", {}).get("sample_steps", 25))
  if steps <= 0:
    raise ValueError("rectified_flow.sample_steps must be positive")
  save_config(cfg, out_root / "config_resolved.yml")
  write_json({
    "architecture": ARCHITECTURE, "time_convention": TIME_CONVENTION,
    "checkpoint": str(Path(args.checkpoint).resolve()), "epoch": checkpoint.get("epoch"),
    "weight_source": "ema" if use_ema else "raw", "sample_steps": steps,
    "test_split": cfg["data"]["test_split"], "num_samples": len(dataset),
    "seed": args.seed, "seed_mapping": "sha256(base_seed:case_id), first8bytes mod (2^63-1)",
    "xray_keys": cfg["data"]["xray_keys"], "device": str(device),
  }, out_root / "export_info.json")
  count = len(dataset)
  rows = []
  for index in range(count):
    item = dataset[index]
    case_id = str(item["sample_id"])
    sample_id = case_id.replace("/", "_").replace("\\", "_")
    if sample_id in (".", ".."):
      raise ValueError("Invalid case ID: " + case_id)
    seed = case_noise_seed(args.seed, case_id)
    generator = torch.Generator(device=device).manual_seed(seed)
    xrays = item["xrays"].unsqueeze(0).to(device)
    spacing = item["spacing"].unsqueeze(0).to(device)
    ori_size = item["ori_size"].unsqueeze(0).to(device)
    sync_device(device)
    started = time.perf_counter()
    fake = sample_ct(
      model, xrays, spacing, ori_size, int(cfg["data"].get("volume_size", 128)),
      steps, bool(cfg.get("rectified_flow", {}).get("clamp_sample", True)),
      device.type == "cuda" and bool(cfg.get("train", {}).get("amp", True)),
      generator=generator)
    sync_device(device)
    elapsed = time.perf_counter() - started
    folder = out_root / sample_id
    if folder.exists():
      raise FileExistsError("Duplicate or colliding case ID: " + case_id)
    save_mha(fake[0, 0].cpu().numpy(), folder / "fake_ct.mha", cfg)
    save_mha(item["real_ct"][0].numpy(), folder / "real_ct.mha", cfg)
    write_json({"sample_id": case_id, "h5_path": item["h5_path"],
                "noise_seed": seed, "sample_steps": steps,
                "weight_source": "ema" if use_ema else "raw"}, folder / "metadata.json")
    rows.append({"index": index, "sample_id": case_id,
                 "inference_seconds": elapsed, "sample_steps": steps,
                 "method": "rfx2ct", "device": str(device)})
    print("exported {}/{} {} ({:.3f}s)".format(index + 1, count, case_id, elapsed), flush=True)
  write_timing_outputs(out_root, rows)


if __name__ == "__main__":
  main()

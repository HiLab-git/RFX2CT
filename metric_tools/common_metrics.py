#!/usr/bin/env python3
"""PSNR, SSIM, LPIPS-VGG and FID for saved RFX2CT volumes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.linalg
import scipy.ndimage
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim
from torchvision import models


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT_DIR = REPO_ROOT / "metric_weights"
DEFAULT_LPIPS_HEAD = DEFAULT_WEIGHT_DIR / "lpips_vgg_v0.1.pth"
DEFAULT_VGG = DEFAULT_WEIGHT_DIR / "vgg16-397923af.pth"
DEFAULT_INCEPTION = DEFAULT_WEIGHT_DIR / "inception_v3_google-0cc3c7bd.pth"
KNOWN_VOLUME_SUFFIXES = (".nii.gz", ".hdf5", ".h5", ".mha", ".mhd", ".nii", ".npy", ".npz")


@dataclass
class VolumePair:
    sample: str
    fake_path: Path
    real_path: Optional[Path] = None
    fake_key: Optional[str] = None
    real_key: Optional[str] = None
    real_flip_axis: Optional[int] = None


def build_arg_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Compute offline metrics for {method} reconstructions."
    )
    parser.add_argument("--pred-root", required=True, type=Path,
                        help="Directory containing already-saved reconstructions.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory. Default: <pred-root>/offline_metrics")
    parser.add_argument("--gt-root", type=Path, default=None,
                        help="Optional LIDC-HDF5 root for predictions that do not save real_ct.")
    parser.add_argument("--gt-split", type=Path, default=None,
                        help="Optional split file restricting the ground-truth case IDs.")
    parser.add_argument("--eval-size", type=int, default=128,
                        help="Fallback isotropic D/H/W size used when --eval-shape is omitted.")
    parser.add_argument("--eval-shape", default=None,
                        help="Metric D,H,W target shape, e.g. 128,128,128 or 128,192,192. "
                             "Use 'native' to keep saved prediction/GT shape.")
    parser.add_argument("--value-space", default="auto",
                        choices=["auto", "normalized", "ct", "hu"],
                        help="Input volume value space before metric normalization.")
    parser.add_argument("--fake-value-space", default=None,
                        choices=["auto", "normalized", "ct", "hu"],
                        help="Override value space for prediction volumes.")
    parser.add_argument("--real-value-space", default=None,
                        choices=["auto", "normalized", "ct", "hu"],
                        help="Override value space for ground-truth volumes.")
    parser.add_argument("--ct-min", type=float, default=0.0)
    parser.add_argument("--ct-max", type=float, default=2500.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size-slices", type=int, default=32)
    parser.add_argument("--slice-stride", type=int, default=1,
                        help="Stride for LPIPS/FID slice extraction. Use 1 for full evaluation.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lpips-head", type=Path, default=DEFAULT_LPIPS_HEAD)
    parser.add_argument("--vgg-weights", type=Path, default=DEFAULT_VGG)
    parser.add_argument("--inception-weights", type=Path, default=DEFAULT_INCEPTION)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--real-flip-axis", type=int, default=None,
                        help="Flip loaded real volumes along this axis before evaluation.")
    return parser


def run_cli(method: str, args: argparse.Namespace, prefer_h5_container: bool = False) -> None:
    out_dir = args.out_dir or (args.pred_root / "offline_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(
        args.pred_root,
        gt_root=args.gt_root,
        gt_split=args.gt_split,
        prefer_h5_container=prefer_h5_container,
        real_flip_axis=args.real_flip_axis,
    )
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(f"No evaluable prediction/ground-truth pairs found in {args.pred_root}")

    evaluator = OfflineEvaluator(args)
    rows, fid_metrics = evaluator.evaluate_pairs(pairs)

    metric_keys = [
        "MAE0", "MSE0", "CosineSimilarity",
        "MAE", "MSE",
        "psnr-3d", "PSNR-1", "PSNR-2", "PSNR-3", "PSNR-avg",
        "SSIM-3d", "SSIM-1", "SSIM-2", "SSIM-3", "SSIM-avg",
        "LPIPS-1", "LPIPS-2", "LPIPS-3", "LPIPS-avg",
    ]
    summary = {
        "method": method,
        "pred_root": str(args.pred_root),
        "gt_root": str(args.gt_root) if args.gt_root else None,
        "eval_size": args.eval_size,
        "eval_shape": format_eval_shape(parse_eval_shape(args.eval_shape, args.eval_size)),
        "value_space": args.value_space,
        "fake_value_space": args.fake_value_space or args.value_space,
        "real_value_space": args.real_value_space or args.value_space,
        "ct_min": args.ct_min,
        "ct_max": args.ct_max,
        "metric_dtype": "float64",
        "psnr_zero_mode": "x2ct",
        "slice_stride": args.slice_stride,
        "lpips_backbone": None if args.no_lpips else "vgg",
        "lpips_head": str(args.lpips_head) if not args.no_lpips else None,
        "vgg_weights": str(args.vgg_weights) if not args.no_lpips else None,
        "alex_weights": None,
        "squeeze_weights": None,
        "inception_weights": str(args.inception_weights) if not args.no_fid else None,
        "num_samples": len(rows),
        "source_shape_counts": count_row_values(rows, "_source_shape"),
        "eval_shape_counts": count_row_values(rows, "_eval_shape"),
        "metrics_mean": {},
        "metrics_std": {},
        "fid": fid_metrics,
    }
    for key in metric_keys:
        vals = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if vals:
            summary["metrics_mean"][key] = float(np.mean(vals))
            summary["metrics_std"][key] = float(np.std(vals))
    if "FID-avg" in fid_metrics:
        summary["metrics_mean"]["FID"] = fid_metrics["FID-avg"]
        summary["metrics_mean"]["FID-avg"] = fid_metrics["FID-avg"]
        for axis_key in ("FID-1", "FID-2", "FID-3"):
            if axis_key in fid_metrics:
                summary["metrics_mean"][axis_key] = fid_metrics[axis_key]

    csv_path = out_dir / "metrics_per_sample.csv"
    fieldnames = ["sample", "source_shape", "eval_shape"] + metric_keys
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = {
                **row,
                "source_shape": row.get("_source_shape", ""),
                "eval_shape": row.get("_eval_shape", ""),
            }
            writer.writerow(row)

    json_path = out_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(rows)} samples for {method}")
    print(f"Per-sample metrics: {csv_path}")
    print(f"Summary metrics: {json_path}")


def count_row_values(rows: Sequence[Dict[str, float]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def format_array_shape(shape: Sequence[int]) -> str:
    return ",".join(str(int(dim)) for dim in shape)


class OfflineEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.eval_shape = parse_eval_shape(args.eval_shape, args.eval_size)
        self.ct_range = float(args.ct_max - args.ct_min)
        if self.ct_range <= 0:
            raise ValueError("--ct-max must be greater than --ct-min")
        self.device = torch.device(args.device)
        self.lpips_model = None
        self.inception_model = None
        if not args.no_lpips:
            self.lpips_model = LPIPSNet(args.vgg_weights, args.lpips_head).to(self.device).eval()
        if not args.no_fid:
            self.inception_model = InceptionFeatures(args.inception_weights).to(self.device).eval()

    def evaluate_pairs(self, pairs: Sequence[VolumePair]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        fid_fake: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        fid_real: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

        for idx, pair in enumerate(pairs, 1):
            fake_raw = load_volume(pair.fake_path, pair.fake_key)
            real_raw = load_real_volume(pair)
            if pair.real_flip_axis is not None:
                real_raw = np.flip(real_raw, axis=pair.real_flip_axis)

            fake_space = self.args.fake_value_space or self.args.value_space
            real_space = self.args.real_value_space or self.args.value_space
            fake_norm = prepare_volume(fake_raw, self.eval_shape, fake_space,
                                       self.args.ct_min, self.args.ct_max)
            real_norm = prepare_volume(real_raw, self.eval_shape, real_space,
                                       self.args.ct_min, self.args.ct_max)
            if fake_norm.shape != real_norm.shape:
                raise ValueError(
                    f"Prediction and ground truth shapes differ for {pair.sample}: "
                    f"{fake_norm.shape} vs {real_norm.shape}. Pass an explicit "
                    "--eval-shape D,H,W for fair resampling."
                )
            row: Dict[str, float] = {
                "sample": pair.sample,
                "_source_shape": f"{format_array_shape(fake_raw.shape)}|{format_array_shape(real_raw.shape)}",
                "_eval_shape": format_array_shape(fake_norm.shape),
            }
            row.update(compute_numpy_metrics(real_norm, fake_norm, self.ct_range))

            if not self.args.no_lpips:
                row.update(self.compute_lpips(real_norm, fake_norm))
            rows.append(row)

            if not self.args.no_fid:
                for axis in (0, 1, 2):
                    fake_features = self.compute_inception_features(fake_norm, axis)
                    real_features = self.compute_inception_features(real_norm, axis)
                    fid_fake[axis].append(fake_features)
                    fid_real[axis].append(real_features)

            print(f"[{idx}/{len(pairs)}] {pair.sample} done")

        fid_metrics: Dict[str, float] = {}
        if not self.args.no_fid:
            values = []
            for axis in (0, 1, 2):
                fake_axis = np.concatenate(fid_fake[axis], axis=0)
                real_axis = np.concatenate(fid_real[axis], axis=0)
                value = calculate_fid(fake_axis, real_axis)
                fid_metrics[f"FID-{axis + 1}"] = float(value)
                values.append(value)
            fid_metrics["FID-avg"] = float(np.mean(values))
            fid_metrics["FID"] = fid_metrics["FID-avg"]
        return rows, fid_metrics

    def compute_lpips(self, real_norm: np.ndarray, fake_norm: np.ndarray) -> Dict[str, float]:
        assert self.lpips_model is not None
        values = []
        out: Dict[str, float] = {}
        for axis in (0, 1, 2):
            real_slices = volume_slices(real_norm, axis, self.args.slice_stride)
            fake_slices = volume_slices(fake_norm, axis, self.args.slice_stride)
            distances = []
            for start in range(0, len(real_slices), self.args.batch_size_slices):
                r = slices_to_tensor(real_slices[start:start + self.args.batch_size_slices], self.device)
                f = slices_to_tensor(fake_slices[start:start + self.args.batch_size_slices], self.device)
                with torch.no_grad():
                    distances.append(self.lpips_model(r, f).detach().cpu().numpy())
            value = float(np.concatenate(distances).mean())
            out[f"LPIPS-{axis + 1}"] = value
            values.append(value)
        out["LPIPS-avg"] = float(np.mean(values))
        return out

    def compute_inception_features(self, volume: np.ndarray, axis: int) -> np.ndarray:
        assert self.inception_model is not None
        slices = volume_slices(volume, axis, self.args.slice_stride)
        features = []
        for start in range(0, len(slices), self.args.batch_size_slices):
            batch = self.prepare_inception_batch(slices[start:start + self.args.batch_size_slices])
            with torch.no_grad():
                features.append(self.inception_model(batch).detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    def prepare_inception_batch(self, slices: Sequence[np.ndarray]) -> torch.Tensor:
        batch = slices_to_tensor(slices, self.device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        return (batch - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)


def discover_pairs(
    pred_root: Path,
    gt_root: Optional[Path] = None,
    gt_split: Optional[Path] = None,
    prefer_h5_container: bool = False,
    real_flip_axis: Optional[int] = None,
    prefer_gt_real: bool = False,
) -> List[VolumePair]:
    pred_root = pred_root.resolve()
    gt_index = build_gt_index(
        gt_root,
        gt_split=gt_split,
    ) if gt_root else {}
    pairs: List[VolumePair] = []

    fake_files = sorted(
        (p for p in pred_root.rglob("*")
         if p.is_file() and is_volume_file(p) and "fake" in p.name.lower()),
        key=lambda p: (str(p.parent), extension_priority(p), p.name),
    )
    seen_fake_dirs = set()
    for fake_path in fake_files:
        if fake_path.stem.lower() == "fake_ct":
            if fake_path.parent in seen_fake_dirs:
                continue
            seen_fake_dirs.add(fake_path.parent)
        real_path = None if prefer_gt_real else find_real_sibling(fake_path)
        sample = fake_path.parent.name if fake_path.stem.lower().startswith("fake") else fake_path.stem
        if gt_index and (real_path is None or prefer_gt_real):
            matched_gt = match_gt_file(sample, gt_index)
            if matched_gt is not None:
                real_path = matched_gt
        if real_path is not None:
            pairs.append(VolumePair(sample=sample, fake_path=fake_path, real_path=real_path,
                                    real_flip_axis=real_flip_axis))

    if prefer_h5_container or not pairs:
        containers = sorted(
            p for p in pred_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".h5", ".hdf5", ".npz"}
        )
        for path in containers:
            if any(path == pair.fake_path for pair in pairs):
                continue
            sample = path.stem
            fake_key, real_key = infer_container_keys(path)
            real_path = None
            if prefer_gt_real:
                real_key = None
            if gt_index and (real_key is None or prefer_gt_real):
                real_path = match_gt_file(sample, gt_index)
            if real_key is not None or real_path is not None:
                pairs.append(VolumePair(sample=sample, fake_path=path, fake_key=fake_key,
                                        real_path=real_path, real_key=real_key,
                                        real_flip_axis=real_flip_axis))
    return pairs


def build_gt_index(
    gt_root: Optional[Path],
    gt_split: Optional[Path] = None,
) -> Dict[str, Path]:
    if gt_root is None:
        return {}
    gt_root = gt_root.resolve()
    files = list(gt_root.rglob("ct_xray_data.h5")) + list(gt_root.rglob("ct_xray_data.hdf5"))
    index = {normalize_sample_id(path.parent.name): path for path in files}
    if gt_split is not None:
        samples = {normalize_sample_id(line) for line in gt_split.read_text().splitlines() if line.strip()}
        index = {sample: path for sample, path in index.items() if sample in samples}
    return index


def match_gt_file(sample: str, gt_index: Dict[str, Path]) -> Optional[Path]:
    sample_lower = normalize_sample_id(sample)
    exact = gt_index.get(sample_lower)
    if exact is not None:
        return exact
    candidates = []
    for name_lower, path in gt_index.items():
        if name_lower.startswith(sample_lower) or sample_lower.startswith(name_lower):
            candidates.append(path)
    unique = sorted(set(candidates), key=lambda p: str(p))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        matches = ", ".join(path.parent.name for path in unique[:8])
        raise ValueError(f"Ambiguous GT match for sample '{sample}': {matches}")
    return None


def normalize_sample_id(sample: str) -> str:
    path = Path(str(sample).strip())
    name = path.name
    lower_name = name.lower()
    if lower_name in {"ct_xray_data", "ct_xray_data.h5", "ct_xray_data.hdf5"} and path.parent.name:
        name = path.parent.name
    else:
        # DICOM UIDs contain dots, so strip only known file suffixes.
        for suffix in KNOWN_VOLUME_SUFFIXES:
            if lower_name.endswith(suffix):
                name = name[:-len(suffix)]
                break
    sample = name.lower()
    sample = re.sub(r"^sample[_-]?", "", sample)
    sample = re.sub(r"^\d+[_-]", "", sample)
    return sample


def find_real_sibling(fake_path: Path) -> Optional[Path]:
    candidates = []
    name = fake_path.name
    for fake_token, real_token in (("fake", "real"), ("pred", "real"), ("output", "real")):
        if fake_token in name.lower():
            idx = name.lower().find(fake_token)
            candidates.append(fake_path.with_name(name[:idx] + real_token + name[idx + len(fake_token):]))
    candidates.extend(sorted(fake_path.parent.glob("real_ct.*")))
    candidates.extend(sorted(fake_path.parent.glob("*real*")))
    for path in candidates:
        if path.exists() and path.is_file() and is_volume_file(path):
            return path
    return None


def is_volume_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".mha", ".mhd", ".nii", ".nii.gz", ".npy", ".npz", ".h5", ".hdf5"))


def extension_priority(path: Path) -> int:
    name = path.name.lower()
    if name.endswith((".mha", ".mhd")):
        return 0
    if name.endswith((".nii", ".nii.gz")):
        return 1
    if name.endswith(".npy"):
        return 2
    if name.endswith(".npz"):
        return 3
    if name.endswith((".h5", ".hdf5")):
        return 4
    return 9


def infer_container_keys(path: Path) -> Tuple[Optional[str], Optional[str]]:
    fake_candidates = ("fake_ct", "ct_pred", "pred_ct", "prediction", "pred", "output", "ct")
    real_candidates = ("real_ct", "ct_true", "target_ct", "target", "gt", "real")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
        fake_key = next((key for key in fake_candidates if key in keys), None)
        real_key = next((key for key in real_candidates if key in keys), None)
        return fake_key, real_key
    with h5py.File(path, "r") as f:
        keys = set(flat_h5_dataset_keys(f))
    fake_key = next((key for key in fake_candidates if key in keys), None)
    real_key = next((key for key in real_candidates if key in keys), None)
    if fake_key is None and len(keys) == 1:
        fake_key = next(iter(keys))
    return fake_key, real_key


def flat_h5_dataset_keys(group: h5py.Group) -> List[str]:
    keys: List[str] = []
    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            keys.append(name)
    group.visititems(visitor)
    return keys


def load_real_volume(pair: VolumePair) -> np.ndarray:
    if pair.real_path is None:
        return load_volume(pair.fake_path, pair.real_key)
    real_key = pair.real_key
    if real_key is None and pair.real_path.name.startswith("ct_xray_data"):
        real_key = "ct"
    return load_volume(pair.real_path, real_key)


def load_volume(path: Path, key: Optional[str] = None) -> np.ndarray:
    name = path.name.lower()
    if name.endswith((".mha", ".mhd")):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    elif name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    elif name.endswith(".npy"):
        arr = np.load(path)
    elif name.endswith(".npz"):
        arr = read_npz_float32(path, key)
    elif name.endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as f:
            actual_key = key or choose_h5_key(f)
            arr = read_h5_float32(f, actual_key)
    else:
        raise ValueError(f"Unsupported volume file: {path}")
    return ensure_3d(arr)


def read_npz_float32(path: Path, key: Optional[str] = None) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as data:
            actual_key = key or choose_np_key(data.files)
            return np.asarray(data[actual_key], dtype=np.float32)
    except ValueError as e:
        if "Object arrays cannot be loaded" not in str(e):
            raise

    # Compatibility for older locally generated metric volumes accidentally
    # saved as object arrays. This path is intentionally limited to .npz
    # reconstruction files and immediately coerces the payload to float32.
    with np.load(path, allow_pickle=True) as data:
        actual_key = key or choose_np_key(data.files)
        arr = data[actual_key]
    if getattr(arr, "dtype", None) == object:
        if arr.shape == ():
            arr = arr.item()
        elif arr.size == 1:
            arr = arr.reshape(-1)[0]
        else:
            arr = np.stack(arr.tolist(), axis=0)
    return np.asarray(arr, dtype=np.float32)


def choose_np_key(keys: Sequence[str]) -> str:
    for key in ("fake_ct", "ct_pred", "pred_ct", "prediction", "pred", "output",
                "ct", "real_ct", "ct_true", "target", "gt"):
        if key in keys:
            return key
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"Cannot infer npz volume key from {list(keys)}")


def choose_h5_key(f: h5py.File) -> str:
    keys = flat_h5_dataset_keys(f)
    for key in ("fake_ct", "ct_pred", "pred_ct", "prediction", "pred", "output",
                "ct", "real_ct", "ct_true", "target", "gt"):
        if key in keys:
            return key
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"Cannot infer h5 volume key from {keys}")


def read_h5_float32(h5_file: h5py.File, key: str) -> np.ndarray:
    dataset = h5_file[key]
    out = np.empty(dataset.shape, dtype=np.float32)
    try:
        dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, out, mtype=h5py.h5t.IEEE_F32LE)
        return out
    except Exception:
        return np.asarray(dataset.astype("float32")[()], dtype=np.float32)


def ensure_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume after squeezing, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def parse_eval_shape(eval_shape: Optional[str], eval_size: int) -> Optional[Tuple[int, int, int]]:
    if eval_shape is None:
        return (int(eval_size), int(eval_size), int(eval_size))
    value = eval_shape.strip().lower()
    if value == "native":
        return None
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size, size)
    if len(parts) == 3:
        shape = tuple(int(part) for part in parts)
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"--eval-shape dimensions must be positive: {eval_shape}")
        return (shape[0], shape[1], shape[2])
    raise ValueError(f"--eval-shape must be 'native', one int, or D,H,W; got {eval_shape!r}")


def format_eval_shape(shape: Optional[Tuple[int, int, int]]) -> str:
    if shape is None:
        return "native"
    return ",".join(str(dim) for dim in shape)


def prepare_volume(
    volume: np.ndarray,
    shape: Optional[Tuple[int, int, int]],
    value_space: str,
    ct_min: float,
    ct_max: float,
) -> np.ndarray:
    norm = to_normalized(volume, value_space, ct_min, ct_max)
    if shape is not None and tuple(norm.shape) != tuple(shape):
        factors = [target / current for target, current in zip(shape, norm.shape)]
        norm = scipy.ndimage.zoom(norm, factors, order=1)
    return np.clip(norm.astype(np.float32), 0.0, 1.0)


def to_normalized(volume: np.ndarray, value_space: str, ct_min: float, ct_max: float) -> np.ndarray:
    arr = volume.astype(np.float32, copy=False)
    ct_range = float(ct_max - ct_min)
    if value_space == "normalized":
        norm = arr
    elif value_space == "ct":
        norm = (arr - ct_min) / ct_range
    elif value_space == "hu":
        norm = (arr + 1024.0 - ct_min) / ct_range
    elif value_space == "auto":
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("Volume contains no finite values")
        min_v = float(finite.min())
        max_v = float(finite.max())
        if min_v <= -900.0 and max_v <= (ct_max - ct_min):
            norm = (arr + 1024.0 - ct_min) / ct_range
        elif max_v <= 1.5 and min_v >= -0.5:
            norm = arr
        else:
            norm = (arr - ct_min) / ct_range
    else:
        raise ValueError(f"Unknown volume value space: {value_space}")
    return np.clip(norm, 0.0, 1.0)


def compute_numpy_metrics(
    real_norm: np.ndarray,
    fake_norm: np.ndarray,
    ct_range: float,
) -> Dict[str, float]:
    dtype = np.float64
    real0 = real_norm.astype(dtype, copy=False)
    fake0 = fake_norm.astype(dtype, copy=False)
    scale = dtype(ct_range)
    real_ct = real0 * scale
    fake_ct = fake0 * scale
    diff = real_ct - fake_ct
    mse = float(np.mean(np.square(diff)))
    flat_real = real0.reshape(1, -1)
    flat_fake = fake0.reshape(1, -1)
    cosine = np.sum(flat_real * flat_fake, axis=1) / (
        np.linalg.norm(flat_real, axis=1) * np.linalg.norm(flat_fake, axis=1) + dtype(1.0e-12)
    )
    out: Dict[str, float] = {
        "MAE0": float(np.mean(np.abs(real0 - fake0))),
        "MSE0": float(np.mean(np.square(real0 - fake0))),
        "CosineSimilarity": float(cosine[0]),
        "MAE": float(np.mean(np.abs(diff))),
        "MSE": mse,
        "psnr-3d": float(psnr_from_mse_array(
            np.asarray([mse], dtype=dtype),
            scale,
        )[0]),
        "SSIM-3d": float(skimage_ssim(real_norm, fake_norm, data_range=1.0)),
    }
    psnr_values = []
    ssim_values = []
    for axis in (0, 1, 2):
        reduce_axes = tuple(dim for dim in range(3) if dim != axis)
        mse_axis = np.mean(np.square(diff), axis=reduce_axes)
        psnrs = psnr_from_mse_array(mse_axis, scale)
        out[f"PSNR-{axis + 1}"] = float(np.mean(psnrs))
        out[f"SSIM-{axis + 1}"] = x2ct_axis_ssim(real_norm, fake_norm, axis)
        psnr_values.append(out[f"PSNR-{axis + 1}"])
        ssim_values.append(out[f"SSIM-{axis + 1}"])
    out["PSNR-avg"] = float(np.mean(psnr_values))
    out["SSIM-avg"] = float(np.mean(ssim_values))
    return out


def x2ct_axis_ssim(real_norm: np.ndarray, fake_norm: np.ndarray, axis: int) -> float:
    real_norm = real_norm.astype(np.float64)
    fake_norm = fake_norm.astype(np.float64)
    if axis == 0:
        real_view = np.transpose(real_norm, (1, 2, 0))
        fake_view = np.transpose(fake_norm, (1, 2, 0))
    elif axis == 1:
        real_view = np.transpose(real_norm, (0, 2, 1))
        fake_view = np.transpose(fake_norm, (0, 2, 1))
    elif axis == 2:
        real_view = real_norm
        fake_view = fake_norm
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return float(skimage_ssim(real_view, fake_view, data_range=1.0, channel_axis=-1))


def psnr_from_mse_array(
    mse_values: np.ndarray,
    pixel_max: float,
) -> np.ndarray:
    mse = np.asarray(mse_values)
    positive_mse = np.where(mse <= 0, np.ones((), dtype=mse.dtype), mse)
    psnr = 20.0 * np.log10(pixel_max / np.sqrt(positive_mse))
    return np.where(mse <= 0, 100.0, psnr)


def volume_slices(volume: np.ndarray, axis: int, stride: int) -> List[np.ndarray]:
    stride = max(1, int(stride))
    return [np.take(volume, i, axis=axis) for i in range(0, volume.shape[axis], stride)]


def slices_to_tensor(slices: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.ascontiguousarray(np.stack(slices, axis=0), dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.ascontiguousarray(arr[:, None, :, :])
    tensor = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32).reshape(arr.shape)
    tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.to(device)


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class LPIPSNet(nn.Module):
    def __init__(self, backbone_weights: Path, lpips_head: Path) -> None:
        super().__init__()
        if not backbone_weights.exists():
            raise FileNotFoundError(f"Missing VGG weights: {backbone_weights}")
        if not lpips_head.exists():
            raise FileNotFoundError(f"Missing LPIPS head weights: {lpips_head}")
        self.scaling = ScalingLayer()
        self.slices, channels = self.build_slices(backbone_weights)
        for param in self.slices.parameters():
            param.requires_grad = False
        self.lins = nn.ModuleList([NetLinLayer(ch) for ch in channels])
        raw_head = torch.load(lpips_head, map_location="cpu")
        mapped_head = {
            key.replace(f"lin{i}.", f"{i}."): value
            for i in range(len(channels))
            for key, value in raw_head.items()
            if key.startswith(f"lin{i}.")
        }
        self.lins.load_state_dict(mapped_head, strict=True)

    @staticmethod
    def build_slices(weights_path: Path) -> tuple[nn.ModuleList, List[int]]:
        model = models.vgg16(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        features = model.features
        return nn.ModuleList([
            features[:4],
            features[4:9],
            features[9:16],
            features[16:23],
            features[23:30],
        ]), [64, 128, 256, 512, 512]

    def forward(self, in0: torch.Tensor, in1: torch.Tensor) -> torch.Tensor:
        in0 = self.scaling(in0 * 2.0 - 1.0)
        in1 = self.scaling(in1 * 2.0 - 1.0)
        feats0 = self.forward_backbone(in0)
        feats1 = self.forward_backbone(in1)
        distances = []
        for feat0, feat1, lin in zip(feats0, feats1, self.lins):
            diff = (normalize_tensor(feat0) - normalize_tensor(feat1)) ** 2
            distances.append(spatial_average(lin(diff)))
        return torch.stack(distances, dim=0).sum(dim=0).view(-1)

    def forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for module in self.slices:
            x = module(x)
            outs.append(x)
        return outs


class NetLinLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Dropout(), nn.Conv2d(channels, 1, 1, 1, 0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(2, 3), keepdim=False).squeeze(1)


IMAGENET_MEAN = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]


class InceptionFeatures(nn.Module):
    def __init__(self, inception_weights: Path) -> None:
        super().__init__()
        if not inception_weights.exists():
            raise FileNotFoundError(f"Missing Inception weights: {inception_weights}")
        inception = models.inception_v3(
            weights=None,
            aux_logits=True,
            transform_input=False,
            init_weights=False,
        )
        inception.load_state_dict(torch.load(inception_weights, map_location="cpu"), strict=True)
        self.inception = inception
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.inception
        x = model.Conv2d_1a_3x3(x)
        x = model.Conv2d_2a_3x3(x)
        x = model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = model.Conv2d_3b_1x1(x)
        x = model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = model.Mixed_5b(x)
        x = model.Mixed_5c(x)
        x = model.Mixed_5d(x)
        x = model.Mixed_6a(x)
        x = model.Mixed_6b(x)
        x = model.Mixed_6c(x)
        x = model.Mixed_6d(x)
        x = model.Mixed_6e(x)
        x = model.Mixed_7a(x)
        x = model.Mixed_7b(x)
        x = model.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        return torch.flatten(x, 1)


def calculate_fid(features1: np.ndarray, features2: np.ndarray) -> float:
    mu1 = np.mean(features1, axis=0)
    mu2 = np.mean(features2, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = scipy.linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(
            sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(fid, 0.0))

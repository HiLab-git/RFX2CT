#!/usr/bin/env python3
"""SegScore from TotalSegmentator labels of reconstructed and target CTs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

try:
    from common_metrics import (
        VolumePair,
        discover_pairs,
        format_array_shape,
        format_eval_shape,
        load_real_volume,
        load_volume,
        parse_eval_shape,
        prepare_volume,
    )
except ImportError:
    from .common_metrics import (
        VolumePair,
        discover_pairs,
        format_array_shape,
        format_eval_shape,
        load_real_volume,
        load_volume,
        parse_eval_shape,
        prepare_volume,
    )


def build_segscore_arg_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Compute TotalSegmentator foreground multiclass Dice for {method}."
    )
    parser.add_argument("--pred-root", required=True, type=Path,
                        help="Directory containing already-saved reconstructions.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory. Default: <pred-root>/segscore_metrics")
    parser.add_argument("--gt-root", type=Path, default=None,
                        help="Optional HDF5 CT/X-ray root for predictions that do not save real_ct.")
    parser.add_argument("--gt-split", type=Path, default=None,
                        help="Optional split file restricting the ground-truth case IDs.")
    parser.add_argument("--eval-size", type=int, default=128,
                        help="Fallback isotropic D/H/W size used when --eval-shape is omitted.")
    parser.add_argument("--eval-shape", default=None,
                        help="SegScore D,H,W target shape, e.g. 128,128,128 or 128,192,192. "
                             "Use 'native' to keep saved prediction/GT shape.")
    parser.add_argument("--value-space", default="auto",
                        choices=["auto", "normalized", "ct", "hu"],
                        help="Input volume value space before normalization.")
    parser.add_argument("--fake-value-space", default=None,
                        choices=["auto", "normalized", "ct", "hu"],
                        help="Override value space for prediction volumes.")
    parser.add_argument("--real-value-space", default=None,
                        choices=["auto", "normalized", "ct", "hu"],
                        help="Override value space for ground-truth volumes.")
    parser.add_argument("--ct-min", type=float, default=0.0)
    parser.add_argument("--ct-max", type=float, default=2500.0)
    parser.add_argument("--hu-shift", type=float, default=1024.0,
                        help="Subtract this from normalized CT-window values before TotalSegmentator. "
                             "Default maps this repo's [0,2500] CT window to HU-like [-1024,1476].")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Segmentation/temporary CT cache. Default: <out-dir>/segscore_cache")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--real-flip-axis", type=int, default=None,
                        help="Flip loaded real volumes along this axis before segmentation. "
                             "Use -1 to disable.")
    parser.add_argument("--fake-flip-axis", type=int, default=None,
                        help="Flip loaded prediction volumes along this axis before segmentation. "
                             "Use -1 to disable.")
    parser.add_argument("--prefer-gt-real", action="store_true",
                        help="When --gt-root is provided, use that HDF5 CT as real target even if "
                             "the prediction directory also contains real_ct files.")
    parser.add_argument("--nifti-spacing", default=None,
                        help="Spacing written to temporary HU NIfTI inputs, e.g. 2,2,2. "
                             "Default: 1,1,1.")
    parser.add_argument("--ts-device", default="gpu:0",
                        help="TotalSegmentator device, e.g. gpu:0, gpu, or cpu.")
    parser.add_argument("--ts-task", default="total",
                        help="TotalSegmentator task. CT SegScore should use 'total'.")
    parser.add_argument("--no-ml", action="store_true",
                        help="Disable TotalSegmentator multilabel output. Not recommended for SegScore.")
    parser.add_argument("--nr-thr-resamp", type=int, default=1)
    parser.add_argument("--nr-thr-saving", type=int, default=6)
    parser.add_argument("--model-size", default="big",
                        help="TotalSegmentator model_size argument.")
    parser.add_argument("--quiet", action="store_true",
                        help="Pass quiet=True to TotalSegmentator.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached segmentations and rerun TotalSegmentator.")
    parser.add_argument("--keep-ct-inputs", action="store_true",
                        help="Keep intermediate HU NIfTI CT inputs in the cache.")
    return parser


def run_segscore_cli(method: str, args: argparse.Namespace,
                     prefer_h5_container: bool = False) -> None:
    if args.real_flip_axis is not None and args.real_flip_axis < 0:
        args.real_flip_axis = None
    if args.fake_flip_axis is not None and args.fake_flip_axis < 0:
        args.fake_flip_axis = None
    out_dir = args.out_dir or (args.pred_root / "segscore_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (out_dir / "segscore_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(
        args.pred_root,
        gt_root=args.gt_root,
        gt_split=args.gt_split,
        prefer_h5_container=prefer_h5_container,
        real_flip_axis=args.real_flip_axis,
        prefer_gt_real=args.prefer_gt_real,
    )
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(f"No evaluable prediction/ground-truth pairs found in {args.pred_root}")

    evaluator = SegScoreEvaluator(method, args, out_dir, cache_dir)
    rows = evaluator.evaluate_pairs(pairs)
    evaluator.write_outputs(rows)


class SegScoreEvaluator:
    def __init__(self, method: str, args: argparse.Namespace,
                 out_dir: Path, cache_dir: Path) -> None:
        self.method = method
        self.method_slug = safe_name(method)
        self.args = args
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        self.eval_shape = parse_eval_shape(args.eval_shape, args.eval_size)
        self.nifti_spacing = parse_nifti_spacing(args.nifti_spacing)
        self.ct_range = float(args.ct_max - args.ct_min)
        if self.ct_range <= 0:
            raise ValueError("--ct-max must be greater than --ct-min")

    def evaluate_pairs(self, pairs: Sequence[VolumePair]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for idx, pair in enumerate(pairs, 1):
            fake_raw = load_volume(pair.fake_path, pair.fake_key)
            if self.args.fake_flip_axis is not None:
                fake_raw = np.flip(fake_raw, axis=self.args.fake_flip_axis)
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

            fake_hu = normalized_to_hu(fake_norm, self.args.ct_min,
                                       self.args.ct_max, self.args.hu_shift)
            real_hu = normalized_to_hu(real_norm, self.args.ct_min,
                                       self.args.ct_max, self.args.hu_shift)

            fake_seg_path = self.segment_volume(pair.sample, "fake", fake_hu)
            real_seg_path = self.segment_volume(pair.sample, "real", real_hu)
            fake_seg = read_segmentation(fake_seg_path)
            real_seg = read_segmentation(real_seg_path)
            score, classes, class_scores = multiclass_dice(fake_seg, real_seg)

            row: Dict[str, object] = {
                "sample": pair.sample,
                "segscore": score,
                "num_classes": int(len(classes)),
                "classes": ";".join(str(int(c)) for c in classes),
                "class_dice_json": json.dumps(
                    {str(int(c)): float(v) for c, v in zip(classes, class_scores)},
                    sort_keys=True,
                ),
                "pred_seg": str(fake_seg_path),
                "target_seg": str(real_seg_path),
                "source_shape": f"{format_array_shape(fake_raw.shape)}|{format_array_shape(real_raw.shape)}",
                "eval_shape": format_array_shape(fake_norm.shape),
                "pred_seg_shape": format_array_shape(fake_seg.shape),
                "target_seg_shape": format_array_shape(real_seg.shape),
            }
            rows.append(row)
            print(f"[{idx}/{len(pairs)}] {pair.sample} segscore={score}")
        return rows

    def segment_volume(self, sample: str, role: str, hu_volume: np.ndarray) -> Path:
        if role not in {"fake", "real"}:
            raise ValueError(f"Unknown role: {role}")
        sample_slug = safe_name(sample)
        role_root = "targets" if role == "real" else self.method_slug
        sample_dir = self.cache_dir / role_root / sample_slug
        sample_dir.mkdir(parents=True, exist_ok=True)
        seg_path = sample_dir / f"{role}_seg.nii.gz"
        meta_path = sample_dir / f"{role}_seg.meta.json"
        ct_path = sample_dir / f"{role}_ct_hu.nii.gz"

        metadata = {
            "sample": sample,
            "role": role,
            "method": self.method if role == "fake" else "target",
            "shape": list(hu_volume.shape),
            "ct_min": self.args.ct_min,
            "ct_max": self.args.ct_max,
            "hu_shift": self.args.hu_shift,
            "fake_flip_axis": self.args.fake_flip_axis,
            "real_flip_axis": self.args.real_flip_axis,
            "prefer_gt_real": self.args.prefer_gt_real,
            "nifti_spacing": list(self.nifti_spacing),
            "eval_shape": format_eval_shape(self.eval_shape),
            "task": self.args.ts_task,
            "ml": not self.args.no_ml,
            "model_size": self.args.model_size,
            "hash": hash_volume(hu_volume),
        }

        if (
            not self.args.force
            and seg_path.exists()
            and meta_path.exists()
            and json.loads(meta_path.read_text(encoding="utf-8")) == metadata
        ):
            return seg_path

        write_hu_nifti(hu_volume, ct_path, self.nifti_spacing)
        run_totalsegmentator(
            ct_path,
            seg_path,
            task=self.args.ts_task,
            ml=not self.args.no_ml,
            device=self.args.ts_device,
            nr_thr_resamp=self.args.nr_thr_resamp,
            nr_thr_saving=self.args.nr_thr_saving,
            quiet=self.args.quiet,
            model_size=self.args.model_size,
        )
        if not seg_path.exists():
            raise FileNotFoundError(f"TotalSegmentator did not create expected output: {seg_path}")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        if not self.args.keep_ct_inputs:
            ct_path.unlink(missing_ok=True)
        return seg_path

    def write_outputs(self, rows: Sequence[Dict[str, object]]) -> None:
        csv_path = self.out_dir / "segscore_per_sample.csv"
        fieldnames = [
            "sample", "segscore", "num_classes", "classes", "class_dice_json",
            "source_shape", "eval_shape", "pred_seg_shape", "target_seg_shape",
            "pred_seg", "target_seg",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        values = np.asarray([row["segscore"] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary = {
            "method": self.method,
            "pred_root": str(self.args.pred_root),
            "gt_root": str(self.args.gt_root) if self.args.gt_root else None,
            "eval_size": self.args.eval_size,
            "eval_shape": format_eval_shape(self.eval_shape),
            "value_space": self.args.value_space,
            "fake_value_space": self.args.fake_value_space or self.args.value_space,
            "real_value_space": self.args.real_value_space or self.args.value_space,
            "ct_min": self.args.ct_min,
            "ct_max": self.args.ct_max,
            "hu_shift": self.args.hu_shift,
            "fake_flip_axis": self.args.fake_flip_axis,
            "real_flip_axis": self.args.real_flip_axis,
            "prefer_gt_real": self.args.prefer_gt_real,
            "nifti_spacing": list(self.nifti_spacing),
            "totalsegmentator": {
                "task": self.args.ts_task,
                "ml": not self.args.no_ml,
                "device": self.args.ts_device,
                "model_size": self.args.model_size,
            },
            "num_samples": len(rows),
            "num_valid": int(finite.size),
            "num_nan": int(values.size - finite.size),
            "segscore_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "segscore_std": float(np.std(finite)) if finite.size else float("nan"),
            "source_shape_counts": count_row_values(rows, "source_shape"),
            "eval_shape_counts": count_row_values(rows, "eval_shape"),
            "pred_seg_shape_counts": count_row_values(rows, "pred_seg_shape"),
            "target_seg_shape_counts": count_row_values(rows, "target_seg_shape"),
        }
        json_path = self.out_dir / "segscore_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Evaluated {len(rows)} samples for {self.method}")
        print(f"Per-sample SegScore: {csv_path}")
        print(f"Summary SegScore: {json_path}")


def run_totalsegmentator(
    input_path: Path,
    output_path: Path,
    task: str,
    ml: bool,
    device: str,
    nr_thr_resamp: int,
    nr_thr_saving: int,
    quiet: bool,
    model_size: str,
) -> None:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is required for SegScore. Install it in zff, for example: "
            "conda run -n zff python -m pip install totalsegmentator --no-deps"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(totalsegmentator)
    call_device = device
    if "model_size" not in signature.parameters and device.startswith("gpu:"):
        call_device = "gpu"
    kwargs = {
        "input": str(input_path),
        "output": str(output_path),
        "task": task,
        "ml": ml,
        "device": call_device,
        "nr_thr_resamp": nr_thr_resamp,
        "nr_thr_saving": nr_thr_saving,
        "quiet": quiet,
    }
    if "model_size" in signature.parameters:
        kwargs["model_size"] = model_size
    totalsegmentator(**kwargs)


def normalized_to_hu(norm: np.ndarray, ct_min: float, ct_max: float, hu_shift: float) -> np.ndarray:
    hu = norm.astype(np.float32, copy=False) * float(ct_max - ct_min) + float(ct_min)
    hu = hu - float(hu_shift)
    return hu.astype(np.float32, copy=False)


def parse_nifti_spacing(value: Optional[str]) -> Tuple[float, float, float]:
    if value is None:
        return (1.0, 1.0, 1.0)
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        spacing = float(parts[0])
        parts_f = (spacing, spacing, spacing)
    elif len(parts) == 3:
        parts_f = tuple(float(part) for part in parts)
    else:
        raise ValueError(f"--nifti-spacing must be one value or Z,Y,X triplet; got {value!r}")
    if any(part <= 0 for part in parts_f):
        raise ValueError(f"--nifti-spacing values must be positive; got {value!r}")
    return parts_f


def write_hu_nifti(volume: np.ndarray, path: Path,
                   spacing_zyx: Tuple[float, float, float]) -> None:
    image = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32), isVector=False)
    image.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(path), True)


def read_segmentation(path: Path) -> np.ndarray:
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    seg = np.squeeze(seg)
    if seg.ndim != 3:
        raise ValueError(f"Expected 3D segmentation at {path}, got shape {seg.shape}")
    return seg.astype(np.int64, copy=False)


def dice_binary(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool, copy=False)
    target_mask = target_mask.astype(bool, copy=False)
    inter = np.logical_and(pred_mask, target_mask).sum(dtype=np.float64)
    denom = pred_mask.sum(dtype=np.float64) + target_mask.sum(dtype=np.float64)
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def multiclass_dice(pred_seg: np.ndarray, target_seg: np.ndarray) -> Tuple[float, np.ndarray, List[float]]:
    if pred_seg.shape != target_seg.shape:
        raise ValueError(
            f"pred_seg and target_seg shapes must match, got {pred_seg.shape} vs {target_seg.shape}"
        )
    classes = np.unique(target_seg)
    classes = classes[classes != 0]
    dice_scores = [
        dice_binary(pred_seg == class_id, target_seg == class_id)
        for class_id in classes
    ]
    if not dice_scores:
        return float("nan"), classes, dice_scores
    return float(np.nanmean(dice_scores)), classes, dice_scores


def hash_volume(volume: np.ndarray) -> str:
    arr = np.ascontiguousarray(volume.astype(np.float32, copy=False))
    digest = hashlib.sha1()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def count_row_values(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def safe_name(value: str) -> str:
    value = str(value).replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "sample"

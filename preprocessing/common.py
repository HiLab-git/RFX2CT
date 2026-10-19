# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# DRR convention adapted from Kai Ma's X2CT-GAN scripts; see NOTICE.md.
"""Shared HDF5 and DRR routines; see NOTICE.md for X2CT-GAN provenance."""

from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import h5py
import numpy as np
from PIL import Image
import SimpleITK as sitk


SPLIT_ROOT = Path(__file__).resolve().parents[1] / "splits"


def add_arguments(parser, default_output):
    parser.add_argument("--output-root", type=Path, default=Path(default_output))
    parser.add_argument("--plastimatch-bin", type=Path,
                        help="Directory containing plastimatch and optionally drr; otherwise use PATH.")
    parser.add_argument("--split", type=Path, action="append",
                        help="Case list; repeat for multiple lists. Default: supplied train/val/test lists.")
    parser.add_argument("--limit", type=int, default=0, help="Process the first N cases; 0 means all.")
    parser.add_argument("--run", action="store_true", help="Write data; otherwise only check input paths.")


def read_case_ids(dataset, split_files=None, limit=0):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    paths = split_files or [SPLIT_ROOT / dataset / (name + ".txt")
                            for name in ("train", "val", "test")]
    cases = []
    for path in paths:
        cases.extend(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Case lists must be nonempty and contain no duplicate IDs")
    if any(case in {".", ".."} or "/" in case or "\\" in case for case in cases):
        raise ValueError("Case IDs must be directory names, not paths")
    return cases[:limit] if limit else cases


def find_input(root, case_id):
    paths = [Path(root) / (case_id + suffix) for suffix in (".mha", ".mhd", ".nii.gz", ".nii")]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError("Expected one CT/mask volume for {} under {}; found {}".format(
            case_id, root, len(found)))
    return found[0]


def source_info(path):
    stat = path.stat()
    info = {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.suffix.lower() == ".mhd":
        # MHD stores voxels in a separate file; include it in the resume identity.
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "ElementDataFile":
                data = path.parent / value.strip()
                if not data.is_file():
                    raise FileNotFoundError("Expected a single MHD voxel file: {}".format(data))
                info["voxel_file"] = source_info(data)
    return info


def case_is_complete(args, case_id, metadata):
    path = args.output_root / case_id / "ct_xray_data.h5"
    if not path.exists():
        return False
    with h5py.File(path, "r") as h5:
        saved = json.loads(h5.attrs.get("preprocess", "{}"))
        settings = saved.get("settings", {})
        valid = (all(settings.get(key) == value for key, value in metadata.items())
                 and set(h5) == {"ct", "spacing", "ori_size", "xray1", "xray2"}
                 and h5["ct"].shape == (256, 256, 256)
                 and all(h5[key].shape == (256, 256) for key in ("xray1", "xray2")))
    if not valid:
        raise FileExistsError("{}: existing data differ or lack provenance; use a new output root".format(path))
    plastimatch, drr = find_renderer(args.plastimatch_bin)
    if (saved.get("renderer", {}).get("version") != renderer_version(plastimatch)
            or saved.get("renderer", {}).get("executable") != drr):
        raise FileExistsError("{}: renderer changed; use a new output root".format(path))
    return True


def find_renderer(directory):
    if directory is not None:
        plastimatch = Path(directory) / "plastimatch"
        if not plastimatch.is_file():
            raise FileNotFoundError(str(plastimatch))
        plastimatch = str(plastimatch.resolve())
        drr = Path(directory) / "drr"
        command = [str(drr.resolve())] if drr.is_file() else [plastimatch, "drr"]
    else:
        plastimatch = shutil.which("plastimatch")
        if not plastimatch:
            raise FileNotFoundError("Install Plastimatch or pass --plastimatch-bin")
        command = [plastimatch, "drr"]
    return plastimatch, command


@lru_cache(maxsize=8)
def renderer_version(plastimatch):
    return subprocess.check_output([plastimatch, "--version"], text=True, timeout=30).strip()


def read_pfm(path):
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"Pf":
            raise ValueError("Expected a grayscale PFM: {}".format(path))
        width, height = map(int, handle.readline().split())
        scale = float(handle.readline())
        if min(width, height) <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError("Invalid PFM dimensions/scale: {}".format(path))
        data = np.fromfile(handle, dtype="<f4" if scale < 0 else ">f4")
    if data.size != width * height or not np.isfinite(data).all():
        raise ValueError("Invalid PFM pixels: {}".format(path))
    return np.flipud(data.reshape(height, width))


def projection_to_xray(raw, frontal):
    maximum = float(raw.max())
    if maximum > 0:
        image = np.clip(raw / maximum * 255.0, 0, 255).astype(np.uint8)
    else:
        image = np.zeros(raw.shape, np.uint8)
    if frontal:
        image = np.fliplr(image)
    return np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)


def generate_xrays(projection, work, plastimatch, drr):
    shifted = work / "shifted.mha"
    adjusted = work / "projection.mha"
    sitk.WriteImage(projection, str(shifted), True)
    subprocess.run([plastimatch, "adjust", "--input", str(shifted), "--output", str(adjusted),
                    "--pw-linear", "0, -1000"], check=True)
    center = projection.TransformContinuousIndexToPhysicalPoint(
        tuple((size - 1.0) / 2.0 for size in projection.GetSize()))
    xrays, rasters, commands = {}, {}, []
    for key, normal in (("xray1", "0 1 0"), ("xray2", "1 0 0")):
        prefix = work / key
        command = drr + ["--threading", "cpu", "-t", "pfm", "--nrm", normal,
                         "--sad", "541", "--sid", "949", "-r", "320 320",
                         "-o", " ".join("{:.8g}".format(v) for v in center), "-z", "500 500",
                         "-I", str(adjusted), "-O", str(prefix)]
        subprocess.run(command, check=True)
        raw = read_pfm(Path(str(prefix) + "0000.pfm"))
        if raw.shape != (320, 320):
            warnings.warn("{}: requested 320x320 DRR, renderer returned {}; actual raster is recorded".format(key, raw.shape))
        xrays[key] = projection_to_xray(raw, frontal=(key == "xray1"))
        rasters[key] = list(raw.shape)
        commands.append(command)
    return xrays, {"version": renderer_version(plastimatch), "executable": drr, "drr_commands": commands,
                   "projection_size_xyz": list(projection.GetSize()),
                   "projection_spacing_xyz": list(projection.GetSpacing()),
                   "projection_origin_xyz": list(projection.GetOrigin()),
                   "projection_direction": list(projection.GetDirection()),
                   "requested_raster": [320, 320], "actual_raster": rasters,
                   "sad_mm": 541, "sdd_mm": 949, "detector_mm": [500, 500],
                   "projection_shift": -1000, "frontal_horizontal_flip": True}


def save_case(args, case_id, ct, projection, metadata):
    ct = np.asarray(ct, dtype=np.float32)
    if ct.shape != (256, 256, 256) or not np.isfinite(ct).all():
        raise ValueError("Expected a finite 256-cubed CT")
    if projection.GetSize() != (320, 320, 320):
        raise ValueError("Expected a 320-cubed projection-stage CT")
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / case_id / "ct_xray_data.h5"
    plastimatch, drr = find_renderer(args.plastimatch_bin)
    with tempfile.TemporaryDirectory(prefix="." + case_id + "-", dir=args.output_root) as temp:
        work = Path(temp)
        xrays, renderer = generate_xrays(projection, work, plastimatch, drr)
        packed = work / "ct_xray_data.h5"
        with h5py.File(packed, "x") as h5:
            h5.create_dataset("ct", data=ct, dtype="float32")
            h5.create_dataset("spacing", data=np.asarray(projection.GetSpacing(), dtype=np.float32))
            h5.create_dataset("ori_size", data=np.float32(320))
            for key, array in xrays.items():
                h5.create_dataset(key, data=array, dtype="float32")
            h5.attrs["source_dataset"] = metadata["dataset"]
            h5.attrs["preprocess"] = json.dumps({"settings": metadata, "renderer": renderer}, sort_keys=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete file atomically and fail if another writer created it.
        os.link(packed, target)
    return target

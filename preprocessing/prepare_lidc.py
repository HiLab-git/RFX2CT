# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Original CT resampling/cropping by Kai Ma (makai0324@gmail.com).
# Adapted for the RFX2CT CLI/HDF5 format; see NOTICE.md.
"""Prepare LIDC from converted CT volumes and aligned binary body masks."""

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from .common import (add_arguments, case_is_complete, find_input,
                     read_case_ids, save_case, source_info)


def resample(volume, spacing_xyz):
    spacing_zyx = np.asarray(spacing_xyz, dtype=np.float64)[::-1]
    shape = np.round(np.asarray(volume.shape) * spacing_zyx)
    factors = shape / np.asarray(volume.shape)
    result = ndimage.zoom(volume, factors, order=3, mode="nearest")
    return result, (spacing_zyx / factors)[::-1]


def crop_to_standard(volume, scale=320):
    for axis in range(3):
        size = volume.shape[axis]
        if size >= scale:
            start = size - scale if axis == 0 else (size - scale) // 2
            slices = [slice(None)] * 3
            slices[axis] = slice(start, start + scale)
            volume = volume[tuple(slices)]
        else:
            pads = [(0, 0)] * 3
            pads[axis] = ((scale - size) // 2, (scale - size + 1) // 2)
            volume = np.pad(volume.astype(np.float64), pads, mode="constant")
    return volume


def prepare_volume(image, mask, input_value_space):
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("CT and body mask must both be 3D")
    if (image.GetSize() != mask.GetSize()
            or not np.allclose(image.GetSpacing(), mask.GetSpacing())
            or not np.allclose(image.GetOrigin(), mask.GetOrigin())
            or not np.allclose(image.GetDirection(), mask.GetDirection())):
        raise ValueError("CT and body mask must use the same voxel grid")
    if not np.allclose(image.GetDirection(), np.eye(3).ravel()):
        raise ValueError("LIDC input must already have identity/LPS direction; reorient CT and mask together first")
    binary = sitk.GetArrayFromImage(mask)
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("Body mask must contain only 0 and 1")
    if not binary.any():
        raise ValueError("Body mask is empty")
    volume = sitk.GetArrayFromImage(image)
    if not np.isfinite(volume).all():
        raise ValueError("CT contains nonfinite values")
    if input_value_space == "hu":
        volume = volume.astype(np.float32) + 1024.0
    elif input_value_space != "shifted_1024":
        raise ValueError("Unknown CT input value space")
    volume, spacing = resample(volume * binary, image.GetSpacing())
    cube = crop_to_standard(volume)
    projection = sitk.GetImageFromArray(cube.astype(np.float32))
    projection.SetSpacing(tuple(float(value) for value in spacing))
    # Original scripts have no public CT320-to-HDF5 converter; this packs a
    # linear 256-cubed resize for the released loader, without claiming pixel equivalence.
    ct = ndimage.zoom(cube, 256.0 / 320.0, order=1)
    return np.clip(ct, 0, 2500).astype(np.float32), projection


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--input-value-space", choices=("hu", "shifted_1024"), required=True,
                        help="Use hu for physical HU; shifted_1024 only for CT already stored as HU+1024.")
    add_arguments(parser, "data/LIDC-HDF5-256")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases = read_case_ids("lidc", args.split, args.limit)
    sources = [(case, find_input(args.ct_root, case),
                find_input(args.mask_root, case + "_outMultiLabelMask")) for case in cases]
    print("LIDC-IDRI: {} cases -> {}{}".format(len(cases), args.output_root,
                                               "" if args.run else " (dry run; add --run to write)"))
    if not args.run:
        return
    for index, (case, path, mask_path) in enumerate(sources, 1):
        metadata = {"dataset": "LIDC-IDRI", "protocol": "lidc-x2ct-2view-v1", "case_id": case,
                    "source": source_info(path), "body_mask": source_info(mask_path),
                    "input_value_space": args.input_value_space, "stored_shift": 1024,
                    "ct_clip": [0, 2500], "ct_shape": [256, 256, 256],
                    "resample_order": 3, "packing_order": 1, "projection_size": 320}
        if case_is_complete(args, case, metadata):
            print("{}/{} {}: already prepared".format(index, len(cases), case))
            continue
        image = sitk.ReadImage(str(path))
        mask = sitk.ReadImage(str(mask_path))
        ct, projection = prepare_volume(image, mask, args.input_value_space)
        output = save_case(args, case, ct, projection, metadata)
        print("{}/{} {}: {}".format(index, len(cases), case, output), flush=True)


if __name__ == "__main__":
    main()

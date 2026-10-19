"""Prepare COLONOG CT volumes with the project's X2CT-GAN-style protocol.

The body cleaning and centered resampling are adapted from the original
``prepare_ctspine1k_colonog.py`` used for the COLONOG experiments. This is a
dataset-specific adaptation of the X2CT-GAN preparation, with a 448 mm field
of view and a [-1000, 1500] HU window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .common import (
    add_arguments,
    case_is_complete,
    find_input,
    read_case_ids,
    save_case,
    source_info,
)


FOV_MM = (448.0, 448.0, 448.0)
HU_WINDOW = (-1000.0, 1500.0)
CT_SHAPE = (256, 256, 256)
PROJECTION_SHAPE = (320, 320, 320)


def canonical_lps(image: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(image, "LPS")


def physical_from_continuous_index(image: sitk.Image, index_xyz: np.ndarray) -> np.ndarray:
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    return origin + direction @ (index_xyz.astype(np.float64) * spacing)


def image_center_physical(image: sitk.Image) -> np.ndarray:
    size_xyz = np.asarray(image.GetSize(), dtype=np.float64)
    return physical_from_continuous_index(image, (size_xyz - 1.0) / 2.0)


def component_areas(labels: np.ndarray, count: int) -> np.ndarray:
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    return areas


def body_mask_from_array(
    array_zyx: np.ndarray,
    threshold: float,
    open_iterations: int,
    dilate_voxels: int,
    min_mask_voxels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove small axial components and the table, keeping the main body."""
    initial_mask = np.asarray(array_zyx > threshold, dtype=bool)
    initial_voxels = int(initial_mask.sum())
    if initial_voxels < min_mask_voxels:
        return initial_mask, {"method": "threshold_fallback"}

    work_mask = initial_mask
    if open_iterations > 0:
        work_mask = ndi.binary_opening(
            work_mask,
            structure=np.ones((1, 3, 3), dtype=bool),
            iterations=int(open_iterations),
        )

    slice_mask = np.zeros_like(work_mask, dtype=bool)
    structure_2d = np.ones((3, 3), dtype=bool)
    for z_idx in range(work_mask.shape[0]):
        labels, count = ndi.label(work_mask[z_idx], structure=structure_2d)
        if count == 0:
            continue
        areas = component_areas(labels, count)
        max_area = int(areas.max())
        if max_area <= 0:
            continue
        # Keep large companion components to retain split pelvis/hip regions.
        min_area = max(64, int(round(max_area * 0.15)))
        keep_labels = np.flatnonzero(areas >= min_area)
        slice_mask[z_idx] = np.isin(labels, keep_labels)

    slice_voxels = int(slice_mask.sum())
    if slice_voxels < min_mask_voxels:
        return initial_mask, {"method": "slice_filter_fallback"}

    labels_3d, component_count = ndi.label(slice_mask, structure=np.ones((3, 3, 3), dtype=bool))
    if component_count == 0:
        body_mask = slice_mask
        body_voxels = slice_voxels
    else:
        areas_3d = component_areas(labels_3d, component_count)
        body_label = int(areas_3d.argmax())
        body_mask = labels_3d == body_label
        body_voxels = int(areas_3d[body_label])

    if body_voxels < min_mask_voxels:
        return initial_mask, {"method": "component_fallback"}

    body_mask = ndi.binary_fill_holes(body_mask)
    if dilate_voxels > 0:
        body_mask = ndi.binary_dilation(
            body_mask,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(dilate_voxels),
        )
    body_mask = np.asarray(body_mask, dtype=bool)
    return body_mask, {
        "method": "axial_lcc_3d_lcc",
        "removed_foreground_voxels": int(np.logical_and(initial_mask, ~body_mask).sum()),
    }


def estimate_body_center(image: sitk.Image, threshold: float, stride: int) -> tuple[np.ndarray, dict[str, object]]:
    stride = max(1, int(stride))
    array = sitk.GetArrayFromImage(image)
    sample = array[::stride, ::stride, ::stride]
    mask, _ = body_mask_from_array(
        sample, threshold=threshold, open_iterations=1, dilate_voxels=0, min_mask_voxels=128
    )
    if int(mask.sum()) < 128:
        return image_center_physical(image), {"method": "image_center"}

    z_idx, y_idx, x_idx = np.nonzero(mask)
    coords_zyx = np.stack([z_idx, y_idx, x_idx], axis=1).astype(np.float64) * stride
    mins = coords_zyx.min(axis=0)
    maxs = coords_zyx.max(axis=0)
    z_center = (mins[0] + maxs[0]) / 2.0
    y_center = float(np.median(coords_zyx[:, 1]))
    x_center = float(np.median(coords_zyx[:, 2]))
    center_xyz = physical_from_continuous_index(image, np.asarray([x_center, y_center, z_center]))
    return center_xyz, {
        "method": "body_mask_cleaned",
        "sample_stride": stride,
        "center_index_xyz": [x_center, y_center, z_center],
    }


def resample_centered(
    image: sitk.Image,
    center_xyz: np.ndarray,
    size_zyx: Sequence[int],
    fov_xyz: Sequence[float],
    default_hu: float,
) -> sitk.Image:
    size_xyz = np.asarray([size_zyx[2], size_zyx[1], size_zyx[0]], dtype=np.int64)
    fov_xyz = np.asarray(fov_xyz, dtype=np.float64)
    spacing_xyz = fov_xyz / size_xyz.astype(np.float64)
    origin_xyz = np.asarray(center_xyz, dtype=np.float64) - ((size_xyz - 1.0) * spacing_xyz / 2.0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_hu))
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetSize([int(v) for v in size_xyz.tolist()])
    resampler.SetOutputSpacing(tuple(float(v) for v in spacing_xyz.tolist()))
    resampler.SetOutputOrigin(tuple(float(v) for v in origin_xyz.tolist()))
    resampler.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return resampler.Execute(image)


def shifted_hu_array(image: sitk.Image, hu_window: tuple[float, float]) -> np.ndarray:
    hu = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    lo, hi = hu_window
    return (np.clip(hu, lo, hi) - lo).astype(np.float32, copy=False)


def image_from_array_like(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32, copy=False))
    image.CopyInformation(reference)
    return image


def clean_body_image(image: sitk.Image, args: argparse.Namespace, role: str) -> tuple[sitk.Image, dict[str, object]]:
    if not args.clean_body:
        return image, {"enabled": False, "role": role}
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    mask, info = body_mask_from_array(
        array,
        threshold=args.body_threshold,
        open_iterations=args.clean_open_iterations,
        dilate_voxels=args.clean_dilate_voxels,
        min_mask_voxels=args.clean_min_mask_voxels,
    )
    cleaned = np.array(array, dtype=np.float32, copy=True)
    cleaned[~mask] = float(args.hu_window[0])
    info.update({"enabled": True, "role": role, "masked_air_voxels": int((~mask).sum())})
    return image_from_array_like(cleaned, image), info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True, help="Directory containing the COLONOG CT volumes.")
    add_arguments(parser, default_output="data/CTSpine1K-COLONOG-HDF5-256-2views-clean")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    case_ids = read_case_ids("colonog", args.split, args.limit)
    cases = [(case_id, find_input(args.volume_root, case_id)) for case_id in case_ids]
    print(f"COLONOG: {len(cases)} cases; output: {args.output_root}")
    if not args.run:
        print("Dry run: input files found. Add --run to prepare the HDF5 dataset.")
        return

    cleaning = argparse.Namespace(
        clean_body=True,
        body_threshold=-500.0,
        clean_open_iterations=1,
        clean_dilate_voxels=3,
        clean_min_mask_voxels=4096,
        hu_window=HU_WINDOW,
    )
    for index, (case_id, source_path) in enumerate(cases, start=1):
        metadata = {
            "dataset": "CTSpine1K/COLONOG",
            "protocol": "colonog-clean-2view-v1",
            "case_id": case_id,
            "source": source_info(source_path),
            "ct_shape_zyx": list(CT_SHAPE),
            "projection_size": PROJECTION_SHAPE[0],
            "fov_mm_xyz": list(FOV_MM),
            "hu_window": list(HU_WINDOW),
            "clean_body": {
                "threshold_hu": -500.0,
                "opening_iterations": 1,
                "dilation_voxels": 3,
                "min_mask_voxels": 4096,
                "center_sample_stride": 4,
            },
        }
        if case_is_complete(args, case_id, metadata):
            print(f"[{index}/{len(cases)}] {case_id}: already complete")
            continue
        try:
            source = canonical_lps(sitk.ReadImage(str(source_path), sitk.sitkFloat32))
            if source.GetDimension() != 3:
                raise ValueError("expected a 3D CT volume")
            center, center_info = estimate_body_center(source, threshold=-500.0, stride=4)
            ct = resample_centered(source, center, CT_SHAPE, FOV_MM, HU_WINDOW[0])
            ct, ct_clean_info = clean_body_image(ct, cleaning, role="hdf5_ct")
            projection = resample_centered(source, center, PROJECTION_SHAPE, FOV_MM, HU_WINDOW[0])
            projection, projection_clean_info = clean_body_image(projection, cleaning, role="projection_drr")
            metadata.update(
                {
                    "source_size_xyz": list(source.GetSize()),
                    "source_spacing_xyz": list(source.GetSpacing()),
                    "center_xyz": center.tolist(),
                    "center_info": center_info,
                    "cleaning_results": {"ct": ct_clean_info, "projection": projection_clean_info},
                }
            )
            projection = image_from_array_like(shifted_hu_array(projection, HU_WINDOW), projection)
            output = save_case(args, case_id, shifted_hu_array(ct, HU_WINDOW), projection, metadata)
        except Exception as exc:
            raise RuntimeError(f"COLONOG case {case_id} failed: {exc}") from exc
        print(f"[{index}/{len(cases)}] {case_id}: {output}")


if __name__ == "__main__":
    main()

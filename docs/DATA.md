# Data

Each prepared case is stored as:

```text
<dataroot>/<case_id>/ct_xray_data.h5
```

The HDF5 file contains a 3D `ct` array in `[z,y,x]` order and square 2D
X-rays. Single-view reconstruction uses `xray1`; biplanar-view reconstruction
uses `xray1+xray2`, at 0° and 90°. These are the frontal and lateral images
from the original X2CT-GAN two-view format. Do not pass angle-indexed
multi-view files: their second image may represent a different angle.

CT values are stored as shifted intensities. They are clipped to `[0,2500]`,
normalized to `[0,1]` and resized to 128³ with PyTorch trilinear interpolation
(`align_corners=False`). X-ray images remain at their stored 256² resolution;
values are clipped to `[0,255]` and divided by 255. The detector orientation and lateral
flip are specified by each configuration's `projectnet` section. Spacing and
original-size metadata are read when present.

The configurations expect these directories; `--dataroot` can select another
location:

```text
data/LIDC-HDF5-256/
data/CTSpine1K-COLONOG-HDF5-256-2views-clean/
```

## Preprocessing

The entry points in `preprocessing/` generate paired HDF5 inputs using
[X2CT-GAN's CT preparation and DRR conventions](https://github.com/kylekma/X2CT/tree/a9f0473ee2b8acfc2abcc9acd7d7cf50435435e3/CT2XRAY).
Install the Python requirements from the repository root and install
[Plastimatch](https://plastimatch.org/) separately. On Ubuntu:

```bash
pip install -r requirements.txt
sudo apt-get install plastimatch
plastimatch --version
```

The Python package dependencies are already included in `requirements.txt`.
The commands below use the modern Plastimatch CLI, tested with version 1.8.0.
If it is not on `PATH`, append `--plastimatch-bin /path/to/bin`; this directory
must contain `plastimatch` and may also contain a standalone `drr` executable.
The renderer runs on CPU. Raw datasets and renderer binaries are not bundled.

### LIDC-IDRI

Start with converted CT volumes and aligned **binary body masks**:

```text
/path/to/LIDC-volumes/<case_id>.mhd
/path/to/LIDC-masks/<case_id>_outMultiLabelMask.mhd
```

`.mha`, `.nii` and `.nii.gz` are also accepted. MHD files must retain their
referenced voxel files. Case IDs must match the supplied split lists. CT and
mask must have identical size, spacing, origin and identity/LPS direction;
masks contain 0 outside the body and 1 inside. If reorientation is needed,
reorient CT and mask together before running this entry point.

For CT stored in physical HU, first check the input paths with a dry run:

```bash
python -m preprocessing.prepare_lidc \
  --ct-root /path/to/LIDC-volumes \
  --mask-root /path/to/LIDC-masks \
  --input-value-space hu \
  --output-root data/LIDC-HDF5-256
```

Append `--run` to generate the HDF5 files. If CT values are **already HU+1024**,
replace `--input-value-space hu` with `--input-value-space shifted_1024`;
otherwise the intensities would be shifted twice. The input encoding is
required explicitly and is not inferred from voxel values.

The LIDC pipeline applies the body mask to shifted intensities, resamples
with cubic SciPy interpolation to approximately 1 mm spacing, then crops or
zero-pads to 320³. It retains the last 320 slices in z and centers the crop
in y/x, following the original CT2XRAY code. The projection-stage image uses
origin zero and identity direction. CT is linearly resized to 256³ and
clipped to `[0,2500]` for HDF5 storage; its offset is HU+1024.

Original CT2XRAY starts with converted volumes and external masks. It does
not provide a complete DICOM-to-HU/body-mask pipeline or the original HDF5
packing program. Our entry point preserves integer interpolation for an
already-shifted input; the physical-HU input adapter uses float32. Resampled
spacing is written in x/y/z order for SimpleITK. The new packing step is an
explicit adaptation, not proof of pixel-identical reproduction of the
historically released LIDC data. Existing prepared two-view HDF5 inputs can
be used directly without regeneration.

### COLONOG

Use the CTSpine1K COLONOG volume directory:

```text
/path/to/CTSpine1K/raw_data/volumes/COLONOG/<case_id>.nii.gz
```

The volumes must contain physical HU. Segmentation labels are not used in
this preprocessing. Check paths, then generate the dataset:

```bash
python -m preprocessing.prepare_colonog \
  --volume-root /path/to/CTSpine1K/raw_data/volumes/COLONOG \
  --output-root data/CTSpine1K-COLONOG-HDF5-256-2views-clean

python -m preprocessing.prepare_colonog \
  --volume-root /path/to/CTSpine1K/raw_data/volumes/COLONOG \
  --output-root data/CTSpine1K-COLONOG-HDF5-256-2views-clean \
  --run
```

This entry retains the COLONOG preparation used in this project: reorient
to LPS, estimate the body center, resample a centered 448³ mm field of view,
and remove small components/table regions. Body cleaning uses a −500 HU
threshold, one axial opening iteration, connected components, hole filling
and three-voxel dilation. CT256³ and projection320³ are sampled independently
with linear interpolation. Values are clipped to `[-1000,1500]` and stored
after adding 1000. This is a dataset-specific adaptation of the LIDC protocol.

### DRR and output

Both entries use the orthogonal frontal/lateral normals `(0,1,0)` and
`(1,0,0)`, SAD 541 mm, source-to-detector distance 949 mm and a 500² mm
detector. They retain the original `plastimatch adjust --pw-linear "0, -1000"`
projection conversion. Each DRR is divided by its maximum, mapped to uint8,
and bilinearly resized to 256². Only the frontal DRR is horizontally flipped.
The X-rays are saved as `xray1` and `xray2` in the HDF5 file.

The requested detector raster is 320². Some Plastimatch 1.8.0 merged-command
builds return 128² despite that request. The entry points warn and record the
**actual raster**, executable, version and commands in the HDF5 `preprocess`
attribute; they do not silently force a different rendering protocol. Choose
and keep the same renderer build for all cases. The original X2CT-GAN README
names Plastimatch 1.7.3.12 on Windows, and its PNG path also used Matplotlib
and OpenCV; this implementation directly packs the resized grayscale pixels.

By default, both entries process the existing train, validation and test
lists without modifying them. Use `--split splits/lidc/train.txt` (or the
COLONOG list) to select one split; repeat `--split` for multiple lists.
Append `--limit 1 --run` to process one case first. Dry runs only check paths
and case IDs, not voxel content or renderer availability.

The only persistent output per case is `ct_xray_data.h5`, containing `ct`,
`xray1`, `xray2`, `spacing` and `ori_size`. `spacing` describes the 320³
projection-stage image; `ori_size` is 320. Temporary CT/PFM files are removed.
Matching completed outputs are skipped on reruns; different input metadata,
settings or renderer paths/versions cause an error. Existing HDF5 files are
never overwritten. Use a fresh output directory when changing the protocol.

COLONOG's storage offset is +1000; LIDC's is +1024. The current reconstruction
exporter still subtracts 1024 for both datasets, so COLONOG exported HU have a
−24 HU offset. This preprocessing addition does not alter that existing export
behavior, model training, input normalization or evaluation.

Source attribution and retained upstream licenses are in
[preprocessing/NOTICE.md](../preprocessing/NOTICE.md).

## Splits

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| LIDC-IDRI | 814 | 102 | 102 |
| COLONOG | 626 | 79 | 79 |

Each dataset directory under `splits/` contains `train.txt`, `val.txt` and
`test.txt`, with one case ID per line. Use these fixed lists directly; no split
generation step is needed. Validation was sampled from the original training
pools using seed `20261019`; the test lists are unchanged. The lists are
disjoint and cover all cases. Training previews use validation; final
reconstruction uses the test list.

Historical checkpoints trained on the full original pools have seen the new
validation cases. Those cases are not an independent validation cohort for
the historical models.

# Data Preparation

Preprocessing is adapted from [X2CT-GAN](https://github.com/kylekma/X2CT/tree/a9f0473ee2b8acfc2abcc9acd7d7cf50435435e3/CT2XRAY),
with a dataset-specific COLONOG pipeline. Install the Python requirements and
Plastimatch, then run the commands below from the repository root:

```bash
pip install -r requirements.txt
sudo apt-get install plastimatch
```

## LIDC-IDRI

Use converted CT volumes named `<case_id>.mhd` and binary body masks named
`<case_id>_outMultiLabelMask.mhd`. CT and mask must share the same voxel grid
with identity/LPS orientation. `.mha`, `.nii` and `.nii.gz` are also supported.

```bash
python -m preprocessing.prepare_lidc \
  --ct-root /path/to/LIDC-volumes \
  --mask-root /path/to/LIDC-masks \
  --input-value-space hu \
  --output-root data/LIDC-HDF5-256 \
  --run
```

Use `--input-value-space shifted_1024` for CT already stored as HU+1024.

CT intensities are shifted by +1024 and masked, resampled to approximately
1 mm spacing with cubic interpolation, then cropped or padded to 320³.
The crop keeps the last 320 slices in z and centers y/x. This volume is used
for projection and linearly resized to 256³ for CT storage, clipped to `[0,2500]`.

## COLONOG

Use CTSpine1K COLONOG volumes in physical HU, named `<case_id>.nii.gz`:

```bash
python -m preprocessing.prepare_colonog \
  --volume-root /path/to/CTSpine1K/raw_data/volumes/COLONOG \
  --output-root data/CTSpine1K-COLONOG-HDF5-256-2views-clean \
  --run
```

Volumes are reoriented to LPS and resampled around the body center within
a 448³ mm field of view. A −500 HU body threshold, component filtering,
hole filling and dilation remove background and table regions. CT and
projection volumes are linearly resampled to 256³ and 320³, respectively.
Intensities are clipped to `[-1000,1500]` HU and shifted by +1000.

## X-ray Generation

Plastimatch generates orthogonal frontal (0°) and lateral (90°) DRRs with
source-to-axis distance 541 mm, source-to-detector distance 949 mm and a
500 × 500 mm detector. Projection intensities use
`plastimatch adjust --pw-linear "0, -1000"`. Each DRR is normalized by its
maximum, mapped to `[0,255]` and bilinearly resized to 256². The frontal
image is horizontally flipped. Views are saved as `xray1` and `xray2`.

The commands process the supplied train, validation and test lists in `splits/`.
Use `--split splits/lidc/train.txt` to select a split, or `--limit 1` to
process one case. Omit `--run` to check input paths without generating data.

## Output Format

Each case is saved to `<dataroot>/<case_id>/ct_xray_data.h5`:

| Key | Content |
| --- | --- |
| `ct` | 256³ CT array in `[z,y,x]` order, stored as shifted intensities |
| `xray1` | 256² frontal X-ray |
| `xray2` | 256² lateral X-ray |
| `spacing`, `ori_size` | Projection volume spacing and size |

During training, CT values are clipped to `[0,2500]`, divided by 2500 and
resized to 128³ with trilinear interpolation. X-rays are divided by 255.

Source attribution and licenses: [preprocessing/NOTICE.md](../preprocessing/NOTICE.md).

# RFX2CT

PyTorch implementation of **RFX2CT: End-to-End Conditioned Rectified
Flow for CT Reconstruction from Few X-Rays**.

[![RFX2CT framework](pictures/pipeline.png)](pictures/pipeline.pdf)

## Dataset

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| LIDC-IDRI | 814 | 102 | 102 |
| COLONOG | 626 | 79 | 79 |

Split files are in `splits/`. See [Data Preparation](docs/DATA.md) for preprocessing.
The model reconstructs 128³ CT volumes from frontal (`xray1`) or biplanar
frontal/lateral (`xray1`, `xray2`) X-rays.

## Installation

Tested with Python 3.9 and PyTorch 1.13.1:

```bash
pip install -r requirements.txt
```

Run the following commands from the repository root.

## Training

| Dataset | Single-view | Biplanar-view |
| --- | --- | --- |
| LIDC-IDRI | `configs/lidc_single_view.yml` | `configs/lidc_biplanar.yml` |
| COLONOG | `configs/colonog_single_view.yml` | `configs/colonog_biplanar.yml` |

LIDC-IDRI biplanar training with two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 -m src.training.train \
  --config configs/lidc_biplanar.yml \
  --dataroot /path/to/LIDC-HDF5-256 \
  --out-dir checkpoints --tag rfx2ct_lidc_biplanar
```

For one GPU, replace `torchrun --standalone --nproc_per_node=2` with `python`
and add `--grad-accum-steps 2`. Use `--resume /path/to/checkpoint.pt` to resume training.

## Reconstruction

Reconstruct the test set using EMA weights and 25-step Euler sampling:

```bash
python -m src.inference.reconstruct \
  --checkpoint checkpoints/rfx2ct_lidc_biplanar/checkpoints/epoch_0200.pt \
  --dataroot /path/to/LIDC-HDF5-256 \
  --test-split splits/lidc/test.txt \
  --out-dir results/lidc_biplanar/CT \
  --sample-steps 25
```

Each case produces `fake_ct.mha` and `real_ct.mha`.

## Evaluation

Place `vgg16-397923af.pth`, `lpips_vgg_v0.1.pth` and
`inception_v3_google-0cc3c7bd.pth` in `metric_weights/`, then run:

```bash
python metric_tools/evaluate.py \
  --pred-root results/lidc_biplanar/CT \
  --value-space hu --eval-shape 128,128,128 --device cuda
```

Results are saved to `offline_metrics/metrics_per_sample.csv` and
`offline_metrics/metrics_summary.json` under the prediction directory.
Metrics include 3D PSNR/SSIM, LPIPS-VGG and FID, with LPIPS/FID averaged over
three slice directions. SegScore evaluation is available in
`metric_tools/evaluate_segscore.py` and requires TotalSegmentator 2.0.5.

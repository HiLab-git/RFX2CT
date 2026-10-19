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

This repository starts from prepared HDF5 inputs. It does not include the
external CT-to-DRR renderer or raw datasets.

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

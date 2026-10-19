# Preprocessing sources

LIDC resampling, crop/pad and DRR conventions are adapted from
[X2CT-GAN's CT2XRAY scripts](https://github.com/kylekma/X2CT/tree/a9f0473ee2b8acfc2abcc9acd7d7cf50435435e3/CT2XRAY)
by Kai Ma, Copyright (c) Tencent. Their source headers specify GPLv3;
that notice is retained in `prepare_lidc.py` and `common.py`, with the license in
[LICENSE-GPL-3.0.txt](LICENSE-GPL-3.0.txt). The upstream repository's separate
license file says LGPLv3 and is preserved verbatim in
[LICENSE-UPSTREAM.txt](LICENSE-UPSTREAM.txt); these are the upstream notices,
not a new license declaration for the RFX2CT model.

The COLONOG body cleaning and centered resampling are ported from this
project's dataset preparation. DRR conversion and HDF5 packing reuse the
project's existing preparation routines. These are adaptations for RFX2CT,
not the missing original X2CT-GAN HDF5 conversion program. Details and
commands are in [docs/DATA.md](../docs/DATA.md).

Please cite the preprocessing reference:

Xingde Ying, Heng Guo, Kai Ma, Jian Wu, Zhengxin Weng, and Yefeng Zheng.
*X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial
Networks*. CVPR, 2019.

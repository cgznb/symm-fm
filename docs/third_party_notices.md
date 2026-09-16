# Third-party notices

## MeWM MRI VQ-GAN

The compatibility implementation in
`src/ispy2_symmflow/models/mewm_vqgan.py` is adapted from
`mewm_ispy2/vqgan.py` in the MeWM source snapshot below.

| Field | Notice |
| --- | --- |
| Work | MeWM, "Medical World Model: Generative Simulation of Tumor Evolution for Treatment Planning" |
| Upstream repository | <https://github.com/scott-yjyang/MeWM> |
| Audited revision | `[revision retained in original local source]` |
| Upstream file | `mewm_ispy2/vqgan.py` |
| Creators and copyright | Yijun Yang, Zhao-Yang Wang, Qiuping Liu, Shuwen Sun, Kang Wang, Rama Chellappa, Zongwei Zhou, Alan Yuille, Lei Zhu, Yu-Dong Zhang, and Jieneng Chen; MeWM's README states copyright Yijun Yang; the audited I-SPY2 VQ-GAN revision history names contributor `xiangxucao` (`cgznb`) |
| License | Creative Commons Attribution-NonCommercial 4.0 International, <https://creativecommons.org/licenses/by-nc/4.0/> |
| Local adapted file | `src/ispy2_symmflow/models/mewm_vqgan.py` |
| Adaptation date | 2026-09-09 |

The local adaptation retains the compatible one-channel 3D encoder,
pre-quantization projection, EMA codebook quantizer, post-quantization
projection, and decoder needed to load the audited MRI checkpoint. It removes
the upstream training-only Lightning, discriminator, and perceptual-loss
surface and adds strict architecture, tensor-shape, numeric-contract,
checkpoint, codebook, and SHA-256 validation plus bounded-memory nearest-code
lookup. These changes are identified here; no endorsement by the upstream
authors is implied.

The upstream root `LICENSE.md` and its README license section both specify CC
BY-NC 4.0. One earlier sentence in that README says "CC BY-NC 2.0", which
conflicts with those two more specific sources. This repository follows the
formal root license and records the inconsistency rather than silently
discarding it.

CC BY-NC 4.0 permits sharing and adaptation only for noncommercial purposes
and requires attribution, a license reference, and an indication of changes.
The Apache-2.0 grant in the repository root does not override those terms for
the adapted file. Commercial use of that file requires separate permission
from the relevant rights holders.

The audited MeWM VQ-GAN configuration records initialization lineage from
`MrGiovanni/DiffTumor` revision
`[revision retained in original local source]`; MeWM's bundled codebook also
notes adaptation from TATS and carries a Meta copyright notice. Those
transitive rights were not independently resolved as one uniform grant during
this integration. They require review in addition to the MeWM notice before
redistribution or commercial use.

## External checkpoint and data

The MeWM all-pairs configuration explicitly loads a pretrained VQ-GAN
checkpoint supplied outside this repository. The checkpoint is not generated
or distributed by this project. Its audited SHA-256 is
`[revision retained in original local source]`.
The source repository's software license does not by itself prove permission
to redistribute that weight file, its training data, or the paired I-SPY2
data. Users must establish the applicable data-use and checkpoint rights
before copying, publishing, or using those artifacts, especially in a
commercial setting.

The local MeWM data paths documented in this repository are machine-local
audit records, not redistributed data and not a license grant. No SSH
credentials are recorded in the repository.

The MU-Glioma transfer configuration loads a second external VQ-GAN checkpoint
with SHA-256
`[revision retained in original local source]`.
The local four-modality wrapper is an additional adaptation of the same MeWM
VQ-GAN implementation: it preserves the frozen shared one-channel codec,
splits and concatenates the fixed `t1c,t1n,t2f,t2w` latent groups, and applies
the MU codebook-range normalization. The same CC BY-NC 4.0 and unresolved
checkpoint/data redistribution cautions apply. The copied MU-Glioma data and
weights remain machine-local and are excluded from version control.

# 🎭 PuppetMaster
### Disentangled Face Swap via Stable Diffusion and Controlled Denoising

> Identity-preserving face swap using Stable Diffusion conditioning with independent control over **identity**, **pose**, **expression**, and **shape**.

> **Status: research prototype — the core method does not work yet.** The conditioning
> adapters (`IdentityProjector`, `PoseExpressionAdapter`, `IlluminationAdaIN`) are
> **untrained**: they are randomly initialised and nothing in this repo optimises them or
> loads a checkpoint. The identity/pose tokens injected into the UNet are therefore
> effectively random, and the metrics below reflect that.

---

## Overview

PuppetMaster is a research pipeline aimed at identity-preserving face swapping. Unlike naive blending approaches, it disentangles facial attributes — encoding identity, pose, expression, and shape as separate conditioning signals — before feeding them into a Stable Diffusion UNet across DDIM inference steps.

The goal is a face swap that preserves the source identity while pose, expression and shape are controlled independently. That goal is not yet met: see the status note above.

---

## Architecture

```
Source Image ──► ArcFace (buffalo_sc) ──► Identity Embedding (512-d)
                                                    │
Target Image ──► BFM Extractor ──► Pose / Expression / Shape Coefficients
                                                    │
                    ┌───────────────────────────────┘
                    ▼
         Multi-Stage Conditioning Fusion
                    │
                    ▼
         Stable Diffusion UNet (SD v1-4)
         DDIM Scheduler (30–50 steps)
                    │
                    ▼
             Swapped Output Image
```

### Key Components

| Module | Purpose |
|--------|---------|
| `ArcFaceIdentityExtractor` | Extracts 512-d L2-normalised identity embedding via InsightFace `buffalo_sc` |
| `BFMExtractor` | Landmark-based Basel Face Model coefficient approximation for pose/expression/shape |
| `SD UNet Conditioning` | Injects disentangled attributes into cross-attention layers of the SD v1-4 UNet |
| `DDIMScheduler` | 30–50 step deterministic inference for stable, high-quality outputs |

---

## Results (untrained adapters)

| Metric | Score |
|--------|-------|
| **ID Retrieval** | 0.044 |
| **Pose Isolation Score** | 0.045 |
| **Expression Isolation Score** | 0.034 |

These numbers were measured with **untrained, randomly initialised adapters** and should be
read as a failure baseline, not as evidence that the method works: an ID retrieval of 0.044
means the swapped face is essentially not recognised as the source identity. The earlier
8-condition ablation was run under the same conditions and is not meaningful either. A fixed
evaluation protocol with baselines, and adapter training, are the next phases of work.

---

## Installation

Python 3.11 is the tested version.

```bash
git clone https://github.com/YOUR_USERNAME/PuppetMaster.git
cd PuppetMaster
pip install -e ".[full]"        # full inference stack (pinned versions)
pip install -e ".[torch,test]"  # enough for the CPU unit tests
python scripts/download_weights.py   # SD, CLIP, buffalo_sc, BiSeNet into ./weights
```

Some checksums in `scripts/download_weights.py` are still placeholders; the script prints the
hash of each unpinned file so it can be filled in.

## Usage

```python
from puppetmaster import load_config, seed_everything
from puppetmaster.pipeline.swap import FaceSwapper

cfg = load_config("configs/infer.yaml", img_size=192)  # 512 needs ~8 GB VRAM, 192 ~4 GB
seed_everything(cfg.seed)
out = FaceSwapper(cfg).swap(source_rgb, target_rgb)     # warns: adapters untrained
```

All inference settings (`img_size`, step counts, guidance, lambdas, model IDs) live in
`configs/infer.yaml`. `PuppetMaster.ipynb` is the original notebook the package was
extracted from and is kept for reference.

## Tests

```bash
pytest -q
```

CPU-only shape/dtype tests on tiny tensors; no weights are downloaded. CI runs them on
Python 3.11 (`.github/workflows/ci.yml`).

---

## Project Structure

```
PuppetMaster/
├── configs/infer.yaml           # single source of inference settings
├── scripts/download_weights.py
├── src/puppetmaster/
│   ├── config.py, seed.py
│   ├── extractors/              # identity (ArcFace), bfm, parsing, clip_disentangler
│   ├── adapters/                # identity_projector, pose_expr_adapter, illumination (untrained)
│   └── pipeline/                # inversion, denoise, blend, swap
├── tests/
├── PuppetMaster.ipynb           # original notebook
└── pyproject.toml
```

---

## Limitations & Future Work
- Adapters are untrained; no training code exists yet
- The CLIP disentangler uses fixed random orthogonal projections, not learned ones
- BFM coefficient approximation is landmark-based (no BFM.mat required); full 3DMM fitting would improve shape accuracy
- Future: fine-tune UNet cross-attention on paired face-swap datasets for stronger identity binding

---

## Author
**Pragna Das** · B.E. Civil Engineering, BITS Pilani (2027)  
`F20230252@pilani.bits-pilani.ac.in`

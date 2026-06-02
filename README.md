# 🎭 PuppetMaster
### Disentangled Face Swap via Stable Diffusion and Controlled Denoising

> Identity-preserving face swap using Stable Diffusion conditioning with independent control over **identity**, **pose**, **expression**, and **shape**.

---

## Overview

PuppetMaster is a deep learning pipeline for high-fidelity, identity-preserving face swapping. Unlike naive blending approaches, it disentangles facial attributes — encoding identity, pose, expression, and shape as separate conditioning signals — before feeding them into a Stable Diffusion UNet across 30–50 DDIM inference steps.

The result is a face swap where the target identity is preserved while the source pose, expression, and shape can be independently controlled or transferred.

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
         Stable Diffusion UNet (SD 1.5)
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
| `SD UNet Conditioning` | Injects disentangled attributes into cross-attention layers of SD 1.5 UNet |
| `DDIMScheduler` | 30–50 step deterministic inference for stable, high-quality outputs |

---

## Results

| Metric | Score |
|--------|-------|
| **ID Retrieval** | 0.044 |
| **Pose Isolation Score** | 0.045 |
| **Expression Isolation Score** | 0.034 |

**Ablation Study (8 conditions):** Identity guidance contributes the maximum lift in ID retrieval, followed by shape. Expression and pose contribute isolatable, measurable deltas.

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (≥ 8 GB VRAM recommended; 4 GB possible at `IMG_SIZE=192`)

```bash
git clone https://github.com/YOUR_USERNAME/PuppetMaster.git
cd PuppetMaster
pip install -r requirements.txt
```

### Requirements
```
diffusers==0.27.2
transformers==4.40.0
accelerate==0.29.3
insightface==0.7.3
onnxruntime-gpu==1.17.1
face-alignment==1.4.1
kornia==0.7.2
scipy==1.11.4
torch>=2.0.0
opencv-python
Pillow
matplotlib
```

---

## Usage

### Quick Start (Notebook)
Open `PuppetMaster.ipynb` in Jupyter or Google Colab (GPU runtime required).

### Pipeline Steps
1. **Stage 1-A** — ArcFace identity extraction from source face
2. **Stage 1-B** — BFM 3DMM attribute extraction (pose, expression, shape)
3. **Stage 2** — Conditioning fusion and SD UNet injection
4. **Stage 3** — DDIM denoising (30–50 steps) → output image

### Resolution
```python
IMG_SIZE = 512   # Default — requires ~8 GB VRAM
IMG_SIZE = 192   # Low VRAM fallback (~4 GB)
```

---

## Project Structure

```
PuppetMaster/
├── PuppetMaster.ipynb       # Main pipeline notebook
├── requirements.txt
├── README.md
└── assets/
    └── architecture.png     # Pipeline diagram
```

---

## Limitations & Future Work
- Currently trained/evaluated on controlled portrait datasets; real-world occlusions reduce ID retrieval
- BFM coefficient approximation is landmark-based (no BFM.mat required); full 3DMM fitting would improve shape accuracy
- Future: fine-tune UNet cross-attention on paired face-swap datasets for stronger identity binding

---

## Author
**Pragna Das** · B.E. Civil Engineering, BITS Pilani (2027)  
`F20230252@pilani.bits-pilani.ac.in`

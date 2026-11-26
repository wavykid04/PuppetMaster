"""CLIP embedding split into pose/expr/light/id sub-spaces (notebook cell 16).

The projections are fixed random orthogonal bases, not learned.
"""

from __future__ import annotations

import numpy as np


def build_orthogonal_projections(clip_dim=512, dims=(32, 32, 16, 128), seed=42):
    """Linear layers whose weights are disjoint rows of one random orthogonal basis."""
    import torch
    import torch.nn as nn

    assert sum(dims) <= clip_dim
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(clip_dim, clip_dim, generator=g)).Q
    projs, off = [], 0
    for d in dims:
        lin = nn.Linear(clip_dim, d, bias=False)
        lin.weight.data = basis[off : off + d].clone()
        lin.requires_grad_(False)
        projs.append(lin)
        off += d
    return projs


class CLIPDisentangler:
    CLIP_DIM, POSE_DIM, EXPR_DIM, LIGHT_DIM, ID_DIM = 512, 32, 32, 16, 128

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        import torch
        from transformers import CLIPProcessor, CLIPVisionModelWithProjection  # lazy

        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.clip = CLIPVisionModelWithProjection.from_pretrained(model_id, torch_dtype=dtype).eval()
        self.clip.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.proj_pose, self.proj_expr, self.proj_light, self.proj_id = build_orthogonal_projections(
            self.CLIP_DIM, (self.POSE_DIM, self.EXPR_DIM, self.LIGHT_DIM, self.ID_DIM)
        )

    def extract(self, image_rgb: np.ndarray) -> dict:
        import torch
        from PIL import Image

        clip = self.clip.to(self.device)
        inputs = self.processor(images=Image.fromarray(image_rgb), return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, dtype=clip.dtype)
        with torch.no_grad():
            emb = clip(pixel_values=pv).image_embeds.float().cpu()
            return {
                "clip_embedding": emb,
                "pose_feat": self.proj_pose(emb),
                "expr_feat": self.proj_expr(emb),
                "light_feat": self.proj_light(emb),
                "id_feat": self.proj_id(emb),
            }

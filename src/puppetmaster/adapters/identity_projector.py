"""ArcFace + CLIP-id embeddings -> cross-attention tokens (notebook cell 18). Untrained."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityProjector(nn.Module):
    def __init__(self, arcface_dim: int = 512, clip_id_dim: int = 128,
                 cross_attn_dim: int = 768, num_tokens: int = 2):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attn_dim = cross_attn_dim
        self.arc_proj = nn.Sequential(
            nn.Linear(arcface_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_tokens * cross_attn_dim),
        )
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_id_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_tokens * cross_attn_dim),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, arcface_emb: torch.Tensor, clip_id_feat: torch.Tensor,
                lambda_id: float = 1.0) -> torch.Tensor:
        """(B, arcface_dim), (B, clip_id_dim) -> (B, num_tokens, cross_attn_dim)."""
        B = arcface_emb.shape[0]
        clip_id_feat = clip_id_feat.to(arcface_emb.device, arcface_emb.dtype)
        arc_tok = self.arc_proj(arcface_emb).view(B, self.num_tokens, self.cross_attn_dim)
        clip_tok = self.clip_proj(clip_id_feat).view(B, self.num_tokens, self.cross_attn_dim)
        alpha = torch.sigmoid(self.fusion_alpha)
        tokens = F.normalize(alpha * arc_tok + (1 - alpha) * clip_tok, dim=-1)
        return tokens * lambda_id

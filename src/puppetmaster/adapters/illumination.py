"""Lighting-statistics-conditioned AdaIN (notebook cell 22). Untrained (identity-initialised)."""

from __future__ import annotations

import torch
import torch.nn as nn


class IlluminationAdaIN(nn.Module):
    CHANNELS = (1280, 640, 320, 320)

    def __init__(self, sh_dim: int = 27):
        super().__init__()
        self.sh_mlp = nn.Sequential(nn.Linear(sh_dim, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU())
        self.gamma_heads = nn.ModuleList([nn.Linear(128, c) for c in self.CHANNELS])
        self.beta_heads = nn.ModuleList([nn.Linear(128, c) for c in self.CHANNELS])
        for g, b in zip(self.gamma_heads, self.beta_heads):
            nn.init.zeros_(g.weight)
            nn.init.ones_(g.bias)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)

    def forward(self, sh_coeffs, feature_map, res_idx: int, lambda_light: float = 1.0):
        """sh_coeffs (B, sh_dim), feature_map (B, C[res_idx], H, W) -> same shape as feature_map."""
        shared = self.sh_mlp(sh_coeffs)
        gamma = (1.0 + self.gamma_heads[res_idx](shared)).view(*shared.shape[:-1], -1, 1, 1)
        beta = self.beta_heads[res_idx](shared).view(*shared.shape[:-1], -1, 1, 1)
        mean = feature_map.mean(dim=[2, 3], keepdim=True)
        std = feature_map.std(dim=[2, 3], keepdim=True).clamp(min=1e-5)
        f_out = gamma * (feature_map - mean) / std + beta
        return (1 - lambda_light) * feature_map + lambda_light * f_out

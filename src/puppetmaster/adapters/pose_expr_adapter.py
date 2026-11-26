"""pose(6) + shape(40) + expr(10) -> per-resolution residuals (notebook cell 20). Untrained."""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseExpressionAdapter(nn.Module):
    LATENT_CHANNELS = (1280, 640, 320, 320)
    POSE_DIM, SHAPE_DIM, EXPR_DIM = 6, 40, 10
    INPUT_DIM = POSE_DIM + SHAPE_DIM + EXPR_DIM
    INJECTION_SCALE = 0.02

    def __init__(self):
        super().__init__()
        self.coeff_mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 512),
        )
        self.res_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(512, ch), nn.Tanh()) for ch in self.LATENT_CHANNELS]
        )
        self.pose_gate = nn.Parameter(torch.ones(1))
        self.shape_gate = nn.Parameter(torch.ones(1))
        self.expr_gate = nn.Parameter(torch.ones(1))

    def forward(self, pose_6d, shape_40d, expr_10d,
                lambda_pose: float = 1.0, lambda_shape: float = 1.0, lambda_expr: float = 1.0):
        """pose/expr from TARGET, shape from SOURCE. Returns residuals list + 512-d feature."""
        pose_s = pose_6d * lambda_pose * torch.sigmoid(self.pose_gate)
        shape_s = shape_40d * lambda_shape * torch.sigmoid(self.shape_gate)
        expr_s = expr_10d * lambda_expr * torch.sigmoid(self.expr_gate)
        feat = self.coeff_mlp(torch.cat([pose_s, shape_s, expr_s], dim=-1))
        return {"residuals": [h(feat) for h in self.res_heads], "heatmap_feat": feat}

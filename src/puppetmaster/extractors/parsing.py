"""Face mask: BiSeNet if weights + code are available, else landmark ellipse (cell 14).

Unlike the notebook, this never downloads or git-clones at runtime; fetch the
weights with scripts/download_weights.py and pass `bisenet_repo` explicitly
(a checkout of zllrunning/face-parsing.PyTorch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FACE_LABELS = [1, 2, 3, 4, 5, 10, 11, 12, 13]
HAIR_LABEL = 17


def ellipse_mask(lmks_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    import cv2

    mask = np.zeros((H, W), dtype=np.uint8)
    cx, cy = int(lmks_2d[:, 0].mean()), int(lmks_2d[:, 1].mean())
    rx = int((lmks_2d[:, 0].max() - lmks_2d[:, 0].min()) * 0.6)
    ry = int((lmks_2d[:, 1].max() - lmks_2d[:, 1].min()) * 0.65)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return mask.astype(bool)


class FaceSegmentor:
    def __init__(self, weights_path: str | None = None, bisenet_repo: str | None = None,
                 device: str = "cpu"):
        self.device = device
        self.net = None
        if weights_path and Path(weights_path).exists() and bisenet_repo:
            import torch

            sys.path.insert(0, str(bisenet_repo))
            from model import BiSeNet  # from face-parsing.PyTorch

            net = BiSeNet(n_classes=19)
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.net = net.eval()

    @property
    def available(self) -> bool:
        return self.net is not None

    def parse(self, image_rgb: np.ndarray, lmks_2d=None):
        """Return (seg int32 HxW, face_mask bool HxW, hair_mask bool HxW)."""
        H, W = image_rgb.shape[:2]
        if not self.available:
            fm = ellipse_mask(lmks_2d, H, W) if lmks_2d is not None else np.zeros((H, W), bool)
            return np.zeros((H, W), np.int32), fm, np.zeros((H, W), bool)

        import cv2
        import torch
        import torchvision.transforms as T

        tf = T.Compose([T.ToPILImage(), T.Resize((512, 512)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        net = self.net.to(self.device)
        with torch.no_grad():
            out = net(tf(image_rgb).unsqueeze(0).to(self.device))[0]
        seg = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        self.net = net.to("cpu")
        return seg, np.isin(seg, FACE_LABELS), seg == HAIR_LABEL

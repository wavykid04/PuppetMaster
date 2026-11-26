"""ArcFace identity embedding via InsightFace (notebook cell 9)."""

from __future__ import annotations

import numpy as np


class ArcFaceIdentityExtractor:
    """Extracts a 512-d L2-normalised identity embedding (InsightFace buffalo_sc)."""

    def __init__(self, name: str = "buffalo_sc", device: str = "cpu", det_size=(320, 320)):
        from insightface.app import FaceAnalysis  # lazy

        self.app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=tuple(det_size))
        self.device = device

    def extract(self, image_rgb: np.ndarray):
        """Return a (1, 512) float32 tensor for the largest face in an RGB image."""
        import cv2
        import torch

        faces = self.app.get(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not faces:
            raise ValueError("No face detected")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = torch.tensor(face.normed_embedding, dtype=torch.float32)
        return emb.unsqueeze(0).to(self.device)

    @staticmethod
    def cosine_sim(a, b) -> float:
        import torch.nn.functional as F

        return F.cosine_similarity(a, b).item()

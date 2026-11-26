"""Landmark-based BFM-style coefficient approximation (notebook cells 11-12).

This is NOT a real 3DMM fit: shape/expression "coefficients" are raw normalised
landmark coordinates. The pose estimator is the "robust" version from cell 12,
which the notebook monkey-patched onto the extractor.
"""

from __future__ import annotations

import numpy as np

N_SHAPE = 40
N_EXPR = 10

FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip    lmk 30
        [0.0, -330.0, -65.0],  # chin   lmk 8
        [-225.0, 170.0, -135.0],  # left eye corner  lmk 36
        [225.0, 170.0, -135.0],  # right eye corner  lmk 45
        [-150.0, -150.0, -125.0],  # left mouth      lmk 48
        [150.0, -150.0, -125.0],  # right mouth      lmk 54
    ],
    dtype=np.float64,
)


def fit_bfm_approx(lmks_2d: np.ndarray, img_hw) -> tuple[np.ndarray, np.ndarray]:
    H, W = img_hw
    x = lmks_2d.copy().astype(np.float32)
    x[:, 0] = (x[:, 0] / W) * 2 - 1
    x[:, 1] = (x[:, 1] / H) * 2 - 1
    flat = x.flatten()
    sp = np.zeros(N_SHAPE, dtype=np.float32)
    n = min(N_SHAPE, len(flat))
    sp[:n] = flat[:n]
    diffs = (lmks_2d - lmks_2d.mean(axis=0)).flatten()
    ep = np.zeros(N_EXPR, dtype=np.float32)
    m = min(N_EXPR, len(diffs))
    ep[:m] = diffs[:m] / (max(H, W) + 1e-6)
    return sp, ep


def estimate_pose(lmks_2d: np.ndarray, img_hw) -> np.ndarray:
    """solvePnP with a 6-point model -> [yaw, pitch, roll, tx, ty, tz]."""
    import cv2
    from scipy.spatial.transform import Rotation

    H, W = img_hw
    image_pts = np.array(
        [lmks_2d[30], lmks_2d[8], lmks_2d[36], lmks_2d[45], lmks_2d[48], lmks_2d[54]],
        dtype=np.float64,
    )
    focal = W
    cam = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return np.zeros(6, dtype=np.float32)
    rmat, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = Rotation.from_matrix(rmat).as_euler("yxz", degrees=True)
    if abs(yaw) > 90:
        yaw = yaw - np.sign(yaw) * 180
    return np.array(
        [yaw, pitch, roll, float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])],
        dtype=np.float32,
    )


def estimate_illumination(image_rgb: np.ndarray, lmks_2d: np.ndarray) -> np.ndarray:
    """Crude 27-d lighting statistics (not true SH coefficients; only 9 slots used)."""
    H, W = image_rgb.shape[:2]
    cx, cy = int(lmks_2d[:, 0].mean()), int(lmks_2d[:, 1].mean())
    r = int(min(H, W) * 0.2)
    patch = image_rgb[max(0, cy - r) : min(H, cy + r), max(0, cx - r) : min(W, cx + r)]
    fp = patch.astype(np.float32).reshape(-1, 3) / 255.0
    sh = np.zeros(27, dtype=np.float32)
    if len(fp) == 0:
        return sh
    sh[0:3] = fp.mean(axis=0)
    sh[3:6] = fp.std(axis=0)
    gray = fp.mean(axis=1)
    sh[6] = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    sh[7] = float(np.percentile(gray, 75))
    sh[8] = float(np.percentile(gray, 25))
    return sh


def safe_pose(pose, max_yaw: float = 45, max_pitch: float = 30):
    """Clamp yaw/pitch of a (6,) pose tensor."""
    p = pose.clone()
    p[0] = p[0].clamp(-max_yaw, max_yaw)
    p[1] = p[1].clamp(-max_pitch, max_pitch)
    return p


class BFMExtractor:
    N_SHAPE = N_SHAPE
    N_EXPR = N_EXPR

    def __init__(self, device: str = "cpu"):
        import face_alignment  # lazy

        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu"
        )

    def extract(self, image_rgb: np.ndarray) -> dict:
        import torch

        lmks_list = self.fa.get_landmarks(image_rgb)
        if not lmks_list:
            raise ValueError("No face landmarks detected")
        lmks = lmks_list[0]
        hw = image_rgb.shape[:2]
        sp, ep = fit_bfm_approx(lmks, hw)

        def t(a):
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        return {
            "shape_coeff": t(sp),
            "expr_coeff": t(ep),
            "pose": t(estimate_pose(lmks, hw)),
            "illumination": t(estimate_illumination(image_rgb, lmks)),
            "landmarks": lmks,
        }

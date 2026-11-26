"""Latent-space mask blending and VAE encode/decode (cells 26, 32)."""

from __future__ import annotations

import numpy as np


def mask_to_latent(face_mask, latent_hw, blend_hardness: float = 0.5, device="cpu"):
    """Resize an HxW mask (bool/uint8/float/tensor) to (1,1,h,w) in [0,1] and sharpen it."""
    import torch
    import torch.nn.functional as F

    if isinstance(face_mask, torch.Tensor):
        fm = face_mask.detach().squeeze().cpu().float()
    else:
        arr = np.asarray(face_mask).squeeze().astype(np.float32)
        if arr.size and arr.max() > 1.0:
            arr = arr / 255.0
        fm = torch.from_numpy(arr)
    while fm.ndim < 4:
        fm = fm.unsqueeze(0)
    m = F.interpolate(fm, size=tuple(latent_hw), mode="bilinear", align_corners=False)
    m = torch.clamp((m - 0.5) * (1 + blend_hardness) + 0.5, 0, 1)
    return m.to(device)


def blend_latents(z_swap, z_target, face_mask, blend_hardness: float = 0.5):
    m = mask_to_latent(face_mask, z_swap.shape[-2:], blend_hardness, device=z_swap.device)
    return m * z_swap.float() + (1 - m) * z_target.float()


def encode_image(image_rgb, vae, img_size: int, device="cpu"):
    import torch
    import torchvision.transforms as T
    from PIL import Image

    pil = Image.fromarray(image_rgb) if isinstance(image_rgb, np.ndarray) else image_rgb
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    x = tf(pil.convert("RGB")).unsqueeze(0).to(device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    return z * vae.config.scaling_factor


def decode_latent(z, vae) -> np.ndarray:
    """Decode in float32 (fp16 decode produces NaNs); returns HxWx3 uint8."""
    import torch

    orig = next(vae.parameters()).dtype
    vae.to(torch.float32)
    with torch.no_grad():
        x = vae.decode(z.float() / vae.config.scaling_factor).sample
    vae.to(orig)
    x = (x.clamp(-1, 1) + 1) / 2
    return (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

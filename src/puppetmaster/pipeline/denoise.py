"""Guided img2img denoising with identity tokens + pose residual injection (cells 30, 40)."""

from __future__ import annotations


def build_cond_hidden_states(null_emb, id_tokens):
    """Concatenate the null text embedding (1|B,77,D) with identity tokens (B,N,D)."""
    import torch

    B = id_tokens.shape[0]
    return torch.cat([null_emb.expand(B, -1, -1).to(id_tokens.dtype), id_tokens], dim=1)


def pose_residual_term(residuals, like, scale: float, clamp: float = 0.5):
    """Collapse the coarsest adapter residual (B,C) to a per-sample constant shaped like `like`."""
    import torch

    res = residuals[0].float().mean(dim=1)  # (B,)
    res = res.view(-1, *([1] * (like.ndim - 1))).expand_as(like)
    return scale * torch.clamp(res, -clamp, clamp)


def guided_img2img(z_target, unet, scheduler, null_emb, id_tokens, residuals=None, *,
                   num_steps: int = 50, strength: float = 0.65, guidance_scale: float = 3.5,
                   injection_scale: float = 0.02, lambda_pose: float = 1.0,
                   inject_frac: float = 0.9, generator=None):
    """Noise z_target to `strength`, then denoise with CFG over [null | id_tokens] conditioning."""
    import torch
    from diffusers import DDIMScheduler

    sch = DDIMScheduler.from_config(scheduler.config)
    sch.set_timesteps(num_steps)
    timesteps = sch.timesteps[int(num_steps * (1 - strength)):]
    unet_dtype = next(unet.parameters()).dtype
    device = z_target.device

    noise = torch.randn(z_target.shape, generator=generator, device=device, dtype=torch.float32)
    a_t = sch.alphas_cumprod[timesteps[0]].to(device)
    z = a_t.sqrt() * z_target.float() + (1 - a_t).sqrt() * noise

    cond_hs = build_cond_hidden_states(null_emb.to(device), id_tokens.to(device, unet_dtype))
    cfg_hs = torch.cat([torch.zeros_like(cond_hs), cond_hs])
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            out = unet(torch.cat([z.to(unet_dtype)] * 2), t, encoder_hidden_states=cfg_hs).sample
            eps_u, eps_c = out.float().chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            if residuals is not None and i < len(timesteps) * inject_frac:
                eps = eps + pose_residual_term(residuals, eps, injection_scale * lambda_pose)
            z = sch.step(eps, t, z).prev_sample.float().clamp(-8, 8)
    return z

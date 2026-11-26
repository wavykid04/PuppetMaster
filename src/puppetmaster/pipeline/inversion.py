"""DDIM inversion (notebook cell 28)."""

from __future__ import annotations


def ddim_inversion(z_start, unet, scheduler, null_emb, num_steps: int = 30):
    """Invert a clean latent to noise with the unconditional embedding.

    Latent math runs in float32 (fp16 drifts); the UNet is called in its own dtype.
    """
    import torch
    from diffusers import DDIMScheduler

    inv_sch = DDIMScheduler.from_config(scheduler.config)
    inv_sch.set_timesteps(num_steps)
    unet_dtype = next(unet.parameters()).dtype
    z = z_start.clone().float()
    ts = list(reversed(inv_sch.timesteps))
    with torch.no_grad():
        for i, t in enumerate(ts):
            hs = null_emb.expand(z.shape[0], -1, -1).to(unet_dtype)
            eps = unet(z.to(unet_dtype), t, encoder_hidden_states=hs).sample.float()
            a_t = inv_sch.alphas_cumprod[t].to(z.device)
            if i + 1 < len(ts):
                a_n = inv_sch.alphas_cumprod[ts[i + 1]].to(z.device)
            else:
                a_n = torch.tensor(1.0, device=z.device)
            pred_x0 = ((z - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-4, 4)
            z = (a_n.sqrt() * pred_x0 + (1 - a_n).sqrt() * eps).clamp(-6, 6)
    return z

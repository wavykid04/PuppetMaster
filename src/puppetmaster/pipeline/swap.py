"""End-to-end face swap (notebook cell 40), driven by InferConfig.

The adapters are untrained unless `models.adapter_checkpoint` points at a
trained checkpoint (none exists yet), so outputs are not meaningful swaps.
"""

from __future__ import annotations

import warnings
from typing import Any

from puppetmaster.config import InferConfig


class FaceSwapper:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self._m: dict[str, Any] = {}

    def load(self) -> FaceSwapper:
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        from puppetmaster.adapters import IdentityProjector, PoseExpressionAdapter
        from puppetmaster.extractors.bfm import BFMExtractor
        from puppetmaster.extractors.clip_disentangler import CLIPDisentangler
        from puppetmaster.extractors.identity import ArcFaceIdentityExtractor
        from puppetmaster.extractors.parsing import FaceSegmentor

        cfg, m = self.cfg, self._m
        device, dtype = cfg.resolve_device(), cfg.resolve_dtype()
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.models.sd_model_id, torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        m.update(vae=pipe.vae.eval(), unet=pipe.unet.eval(),
                 scheduler=DDIMScheduler.from_config(pipe.scheduler.config))
        with torch.no_grad():
            ids = pipe.tokenizer([""], return_tensors="pt", padding="max_length",
                                 max_length=77, truncation=True).input_ids.to(device)
            m["null_emb"] = pipe.text_encoder(ids)[0]
        m["arcface"] = ArcFaceIdentityExtractor(cfg.models.insightface_name, device)
        m["bfm"] = BFMExtractor(device)
        m["clip"] = CLIPDisentangler(cfg.models.clip_model_id, device)
        m["parser"] = FaceSegmentor(cfg.models.bisenet_path, device=device)
        m["id_proj"] = IdentityProjector().to(device).eval()
        m["pose_adapter"] = PoseExpressionAdapter().to(device).eval()
        if cfg.models.adapter_checkpoint:
            state = torch.load(cfg.models.adapter_checkpoint, map_location="cpu")
            m["id_proj"].load_state_dict(state["identity_projector"])
            m["pose_adapter"].load_state_dict(state["pose_expr_adapter"])
        else:
            warnings.warn("No adapter checkpoint configured: adapters are UNTRAINED (random init).",
                          stacklevel=2)
        m["device"] = device
        return self

    def swap(self, source_rgb, target_rgb) -> dict:
        import torch

        from puppetmaster.extractors.bfm import safe_pose
        from puppetmaster.pipeline.blend import blend_latents, decode_latent, encode_image
        from puppetmaster.pipeline.denoise import guided_img2img

        if not self._m:
            self.load()
        cfg, m, lam = self.cfg, self._m, self.cfg.lambdas
        dev = m["device"]
        with torch.no_grad():
            src_id = m["arcface"].extract(source_rgb)
            src_bfm, tgt_bfm = m["bfm"].extract(source_rgb), m["bfm"].extract(target_rgb)
            _, tgt_mask, _ = m["parser"].parse(target_rgb, tgt_bfm["landmarks"])
            clip_id = m["clip"].extract(source_rgb)["id_feat"].to(dev)
            id_tokens = m["id_proj"](src_id, clip_id, lambda_id=lam.id)
            pe = m["pose_adapter"](
                safe_pose(tgt_bfm["pose"]).unsqueeze(0).to(dev),
                src_bfm["shape_coeff"].unsqueeze(0).to(dev),
                tgt_bfm["expr_coeff"].unsqueeze(0).to(dev),
                lambda_pose=lam.pose, lambda_shape=lam.shape, lambda_expr=lam.expr,
            )
            z_tgt = encode_image(target_rgb, m["vae"], cfg.img_size, dev)
            gen = torch.Generator(device=dev).manual_seed(cfg.seed)
            z = guided_img2img(
                z_tgt, m["unet"], m["scheduler"], m["null_emb"], id_tokens, pe["residuals"],
                num_steps=cfg.num_steps, strength=cfg.strength,
                guidance_scale=cfg.guidance_scale, lambda_pose=lam.pose,
                injection_scale=m["pose_adapter"].INJECTION_SCALE, generator=gen,
            )
            result = decode_latent(blend_latents(z, z_tgt, tgt_mask, cfg.blend_hardness), m["vae"])
        try:
            id_sim = m["arcface"].cosine_sim(src_id, m["arcface"].extract(result))
        except ValueError:
            id_sim = None
        return {"result": result, "identity_sim": id_sim}

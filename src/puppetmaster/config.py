"""Config loader for configs/infer.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "infer.yaml"


@dataclass
class ModelsConfig:
    sd_model_id: str = "CompVis/stable-diffusion-v1-4"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    insightface_name: str = "buffalo_sc"
    bisenet_path: str = "weights/bisenet_face_parsing.pth"
    adapter_checkpoint: str | None = None


@dataclass
class LambdasConfig:
    id: float = 2.5
    pose: float = 0.3
    shape: float = 1.2
    expr: float = 0.3
    light: float = 0.5


@dataclass
class InferConfig:
    seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    img_size: int = 512
    num_steps: int = 50
    inversion_steps: int = 30
    guidance_scale: float = 3.5
    strength: float = 0.65
    blend_hardness: float = 0.5
    models: ModelsConfig = field(default_factory=ModelsConfig)
    lambdas: LambdasConfig = field(default_factory=LambdasConfig)

    def __post_init__(self) -> None:
        if self.img_size <= 0 or self.img_size % 8 != 0:
            raise ValueError(f"img_size must be a positive multiple of 8, got {self.img_size}")
        if self.num_steps <= 0 or self.inversion_steps <= 0:
            raise ValueError("num_steps and inversion_steps must be positive")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError(f"strength must be in (0, 1], got {self.strength}")

    @property
    def latent_size(self) -> int:
        return self.img_size // 8

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolve_dtype(self):
        import torch

        if self.dtype == "auto":
            return torch.float16 if self.resolve_device() == "cuda" else torch.float32
        return getattr(torch, self.dtype)


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: dict[str, Any] | None) -> InferConfig:
    data = dict(data or {})
    models = _build(ModelsConfig, data.pop("models", None) or {})
    lambdas = _build(LambdasConfig, data.pop("lambdas", None) or {})
    return _build(InferConfig, {**data, "models": models, "lambdas": lambdas})


def load_config(path: str | Path | None = None, **overrides: Any) -> InferConfig:
    """Load an InferConfig from YAML. Top-level keys can be overridden via kwargs."""
    import yaml

    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.update(overrides)
    return config_from_dict(data)

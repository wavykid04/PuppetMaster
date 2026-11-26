"""Conditioning adapters (notebook cells 18, 20, 22).

WARNING: these modules are UNTRAINED. The notebook never optimises them and no
checkpoint exists, so their outputs are effectively random projections.

Submodules import torch at module level; this package resolves them lazily so
``import puppetmaster.adapters`` stays cheap.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "IdentityProjector": "identity_projector",
    "PoseExpressionAdapter": "pose_expr_adapter",
    "IlluminationAdaIN": "illumination",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        mod = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(name)

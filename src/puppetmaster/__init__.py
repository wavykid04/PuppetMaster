"""PuppetMaster: disentangled face-swap research package.

Importing this package is cheap: torch, diffusers, transformers and insightface
are only imported inside the functions/classes that need them.

NOTE: the conditioning adapters are untrained (random init). See README.
"""

from puppetmaster.config import InferConfig, load_config
from puppetmaster.seed import seed_everything

__all__ = ["InferConfig", "load_config", "seed_everything"]
__version__ = "0.1.0"

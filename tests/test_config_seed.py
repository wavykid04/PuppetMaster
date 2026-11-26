import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from puppetmaster import load_config, seed_everything
from puppetmaster.config import config_from_dict


def test_default_config_loads():
    cfg = load_config()
    assert cfg.img_size % 8 == 0
    assert cfg.latent_size == cfg.img_size // 8
    assert cfg.num_steps > 0
    assert cfg.models.adapter_checkpoint is None


def test_overrides_and_validation():
    assert load_config(img_size=192).latent_size == 24
    with pytest.raises(ValueError):
        load_config(img_size=100)
    with pytest.raises(ValueError):
        config_from_dict({"bogus": 1})


def test_seed_everything_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand())
    seed_everything(123)
    assert a == (random.random(), np.random.rand())


def test_package_import_is_light():
    code = ("import sys, puppetmaster, puppetmaster.adapters, puppetmaster.pipeline.swap, "
            "puppetmaster.extractors.bfm; "
            "heavy = {'torch', 'diffusers', 'transformers', 'insightface', 'cv2'} & set(sys.modules); "
            "assert not heavy, heavy")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

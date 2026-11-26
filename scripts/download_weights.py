"""Download model weights used by PuppetMaster into ./weights (never run in CI).

Usage:
    python scripts/download_weights.py [--only sd clip buffalo_sc bisenet] [--dest weights]

SHA256 values marked None are unknown placeholders: the file is downloaded and its
hash printed so it can be pinned here. Known hashes are verified and mismatches abort.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

HF_REPOS = {
    # The notebook loads SD v1-4; switch both here and in configs/infer.yaml together.
    "sd": {"repo_id": "CompVis/stable-diffusion-v1-4", "revision": "main"},
    "clip": {"repo_id": "openai/clip-vit-base-patch32", "revision": "main"},
}

URL_FILES = {
    "buffalo_sc": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        "filename": "buffalo_sc.zip",
        "sha256": None,  # TODO: pin
        "extract_to": "insightface/models/buffalo_sc",
    },
    "bisenet": {
        # 79999_iter.pth from zllrunning/face-parsing.PyTorch is distributed via Google Drive;
        # set a stable mirror URL here before use.
        "url": None,
        "filename": "bisenet_face_parsing.pth",
        "sha256": None,  # TODO: pin
        "extract_to": None,
    },
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(name: str, spec: dict, dest: Path) -> None:
    out = dest / spec["filename"]
    if spec["url"] is None:
        print(f"[{name}] no download URL configured; place the file at {out} manually.")
    elif not out.exists():
        print(f"[{name}] downloading {spec['url']}")
        urllib.request.urlretrieve(spec["url"], out)
    if not out.exists():
        return
    digest = sha256sum(out)
    if spec["sha256"] is None:
        print(f"[{name}] sha256={digest} (unpinned; add it to this script)")
    elif digest != spec["sha256"]:
        sys.exit(f"[{name}] checksum mismatch: expected {spec['sha256']}, got {digest}")
    else:
        print(f"[{name}] checksum OK")
    if spec["extract_to"] and out.suffix == ".zip":
        target = dest / spec["extract_to"]
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out) as zf:
            zf.extractall(target)


def fetch_hf(name: str, spec: dict, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(spec["repo_id"], revision=spec["revision"],
                             cache_dir=str(dest / "hf"))
    print(f"[{name}] {spec['repo_id']}@{spec['revision']} -> {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=[*HF_REPOS, *URL_FILES])
    ap.add_argument("--dest", default="weights")
    args = ap.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in args.only or [*HF_REPOS, *URL_FILES]:
        if name in HF_REPOS:
            fetch_hf(name, HF_REPOS[name], dest)
        else:
            fetch_url(name, URL_FILES[name], dest)


if __name__ == "__main__":
    main()

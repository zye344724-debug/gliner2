#!/usr/bin/env python3
"""Download GLiNER2 base weights to a local dir (Mac-friendly, resumable)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "encoder_config/config.json",
    "tokenizer.json",
    "spm.model",
)


def cleanup_hf_locks() -> None:
    lock_root = Path.home() / ".cache/huggingface/hub/.locks"
    if not lock_root.exists():
        return
    for lock in lock_root.rglob("*.lock"):
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass
    blobs_root = Path.home() / ".cache/huggingface/hub"
    for incomplete in blobs_root.rglob("*.incomplete"):
        try:
            incomplete.unlink(missing_ok=True)
        except OSError:
            pass


def is_complete(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return all((model_dir / rel).exists() for rel in REQUIRED_FILES)


def download(repo_id: str, local_dir: Path, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    if is_complete(local_dir) and not force:
        print(f"Model already present: {local_dir}")
        return local_dir

    if force and local_dir.exists():
        shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    cleanup_hf_locks()

    # Mac / proxy: disable xet transfer (often hangs behind local proxy)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    if not is_complete(local_dir):
        missing = [rel for rel in REQUIRED_FILES if not (local_dir / rel).exists()]
        raise RuntimeError(f"Incomplete download, missing: {missing}")

    print(f"Download complete: {local_dir}")
    return local_dir


def verify_load(model_dir: Path) -> None:
    from gliner2 import GLiNER2

    print(f"Verifying load from {model_dir} ...")
    model = GLiNER2.from_pretrained(str(model_dir), map_location="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK: loaded model with {n_params:,} parameters on CPU")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure local GLiNER2 weights")
    parser.add_argument("--repo", default=cfg.BASE_MODEL)
    parser.add_argument("--local-dir", type=Path, default=cfg.LOCAL_MODEL_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    path = download(args.repo, args.local_dir, force=args.force)
    if not args.skip_verify:
        verify_load(path)


if __name__ == "__main__":
    main()

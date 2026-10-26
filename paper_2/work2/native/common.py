"""Small, resumable experiment records; no content digests or lease gates."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, default=_json_default, allow_nan=False) + "\n")


def seed_all(seed: int = 13, threads: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def require_cuda() -> dict:
    """A real computation is required; a cached availability flag is insufficient."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_OFFLINE: CUDA unavailable; experiment must stop")
    probe = torch.ones((32, 32), device="cuda")
    if not bool(torch.isfinite(probe @ probe).all()):
        raise RuntimeError("GPU_FATAL: CUDA computation produced nonfinite output")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "checked_at": time.time(),
    }


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)

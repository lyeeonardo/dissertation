"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import torch

def _finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    # Existing controller moments are FP32. Retain FP64 when supplied for audits.
    return value if value.dtype == torch.float64 else value.float()

def _moments(mean, base_variance):
    mean = _finite_tensor(mean, "mean")
    variance = _finite_tensor(base_variance, "base_variance")
    if mean.ndim < 1 or mean.shape != variance.shape:
        raise ValueError("mean and base_variance must have equal shapes ending in features")
    if mean.device != variance.device:
        raise ValueError("mean and base_variance must share a device")
    if (variance <= 0).any():
        raise ValueError("base_variance must be strictly positive")
    dtype = torch.promote_types(mean.dtype, variance.dtype)
    return mean.to(dtype), variance.to(dtype)

def _draw_count(draws, minimum=1):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < minimum:
        raise ValueError(f"draws must be an integer >= {minimum}")

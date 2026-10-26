"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import math

from dataclasses import dataclass

from numbers import Real

import torch

from .normalized_observation_law import _draw_count, _finite_tensor, _moments

SCHEMA_VERSION = 1

SCALE_SQUARED_BOUNDS = (0.001, 1000.0)

BIAS_BOUNDS = (-2.0, 2.0)

def _vector(value, name, *, positive=False):
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty numeric feature vector") from error
    if not result or any(
        isinstance(x, bool)
        or not isinstance(x, Real)
        or not math.isfinite(x)
        or (positive and x <= 0)
        for x in result
    ):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}feature vector")
    return tuple(float(x) for x in result)

class _ResidualMethods:
    def _validate(self):
        scale = _vector(self.scale_squared, "scale_squared", positive=True)
        bias = (0.0,) * len(scale) if self.bias is None else _vector(self.bias, "bias")
        if len(bias) != len(scale):
            raise ValueError("bias and scale_squared must have the same feature dimension")
        object.__setattr__(self, "scale_squared", scale)
        object.__setattr__(self, "bias", bias)

    def parameters(self, mean, base_variance):
        """Return corrected location and scale squared, not Student-t variance."""
        mean, base = _moments(mean, base_variance)
        if mean.shape[-1] != len(self.scale_squared):
            raise ValueError("Law and moments have different feature dimensions")
        scale = mean.new_tensor(self.scale_squared)
        bias = mean.new_tensor(self.bias)
        location, scale_squared = mean + bias * base.sqrt(), scale * base
        if not torch.isfinite(location).all() or not torch.isfinite(scale_squared).all():
            raise FloatingPointError("Unrepresentable law parameters")
        if (scale_squared <= 0).any():
            raise FloatingPointError("Scale squared underflows in the moments' dtype")
        return location, scale_squared

    def log_prob(self, observed, mean, base_variance):
        location, scale_squared = self.parameters(mean, base_variance)
        value = _finite_tensor(observed, "observed")
        if value.shape[-1:] != location.shape[-1:] or value.device != location.device:
            raise ValueError("Observed feature dimension/device must match the moments")
        residual = value - location
        if self.df is None:
            result = -0.5 * (
                residual.square() / scale_squared + scale_squared.log() + math.log(2 * math.pi)
            ).sum(-1)
        else:
            # logaddexp avoids squaring large but representable tail residuals.
            log_ratio = 2 * residual.abs().log() - scale_squared.log() - math.log(self.df)
            tail = torch.logaddexp(torch.zeros_like(log_ratio), log_ratio)
            constant = (
                math.lgamma((self.df + 1) / 2)
                - math.lgamma(self.df / 2)
                - 0.5 * math.log(self.df * math.pi)
            )
            result = (constant - 0.5 * scale_squared.log() - (self.df + 1) / 2 * tail).sum(-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite normalized residual log density")
        return result

    def _noise(self, shape, reference, generator):
        normal = torch.randn(
            shape, dtype=reference.dtype, device=reference.device, generator=generator
        )
        if self.df is None:
            return normal
        # Integer df=4/8 permits exact chi-square draws using only public torch RNG APIs.
        squares = (
            torch.randn(
                (*shape, self.df),
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            )
            .square()
            .sum(-1)
        )
        if (squares <= 0).any():
            raise FloatingPointError("Degenerate Student-t denominator draw")
        return normal / (squares / self.df).sqrt()

    def sample(self, mean, base_variance, draws=64, *, generator=None, noise=None):
        """Draw [..., draws, features]; explicit noise is STANDARD NORMAL or T(df).

        Each coordinate has an independent Student-t denominator (product law).
        Explicit noise can broadcast over leading axes for common candidate draws.
        """
        _draw_count(draws)
        location, scale_squared = self.parameters(mean, base_variance)
        shape = (*location.shape[:-1], draws, location.shape[-1])
        if noise is None:
            noise = self._noise(shape, location, generator)
        else:
            noise = _finite_tensor(noise, "noise")
            if noise.device != location.device:
                raise ValueError("noise must share the moments' device")
            try:
                matches = torch.broadcast_shapes(noise.shape, shape) == shape
            except RuntimeError as error:
                raise ValueError("noise must broadcast to the sample shape") from error
            if not matches:
                raise ValueError("noise must broadcast to the sample shape")
        result = location[..., None, :] + scale_squared.sqrt()[..., None, :] * noise
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite hypothetical observation")
        return result

    def mutual_information(
        self, mean, base_variance, prior, draws=64, *, generator=None, noise=None
    ):
        """Raw same-law stratified MI and MC standard error; negatives are retained."""
        _draw_count(draws, minimum=2)
        mean, base = _moments(mean, base_variance)
        if mean.ndim != 3:
            raise ValueError("MI moments must be [candidates, hypotheses, features]")
        _, hypotheses, dimensions = mean.shape
        prior = _finite_tensor(prior, "prior")
        if prior.shape != (hypotheses,) or prior.device != mean.device or (prior < 0).any():
            raise ValueError("MI requires one nonnegative common prior on the moments' device")
        if not torch.isclose(prior.sum(), prior.new_tensor(1.0), atol=1e-7, rtol=1e-6):
            raise ValueError("MI prior must sum to one")
        if noise is None:
            noise = self._noise((hypotheses, draws, dimensions), mean, generator)
        elif not isinstance(noise, torch.Tensor) or noise.shape != (hypotheses, draws, dimensions):
            raise ValueError("MI noise must be [hypotheses, draws, features]")
        observations = self.sample(mean, base, draws, noise=noise)
        all_log_prob = self.log_prob(
            observations[:, :, :, None], mean[:, None, None], base[:, None, None]
        )
        log_prior = torch.where(prior > 0, prior.log(), -torch.inf)
        mixture = torch.logsumexp(all_log_prob + log_prior, dim=-1)
        own = self.log_prob(observations, mean[:, :, None], base[:, :, None])
        ratios = own - mixture
        estimate = (ratios.mean(-1) * prior).sum(-1)
        error = ((ratios.var(-1, unbiased=True) * prior.square()).sum(-1) / draws).sqrt()
        if not torch.isfinite(estimate).all() or not torch.isfinite(error).all():
            raise FloatingPointError("Nonfinite information estimate")
        return estimate, error

    def to_spec(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "family": self.family,
            "df": self.df,
            "scale_squared": list(self.scale_squared),
            "bias": list(self.bias),
            "parameterization": (
                "location=mean+bias*sqrt(base_variance); scale2=scale_squared*base_variance"
            ),
        }

    def to_dict(self):
        return self.to_spec()

    @classmethod
    def from_spec(cls, spec):
        result = observation_law_from_spec(spec)
        if type(result) is not cls:
            raise ValueError(f"Serialized law must be a {cls.__name__}")
        return result

@dataclass(frozen=True)
class ProductStudentT(_ResidualMethods):
    df: int
    scale_squared: tuple[float, ...]
    bias: tuple[float, ...] | None = None
    family = "product_student_t"

    def __post_init__(self):
        if isinstance(self.df, bool) or not isinstance(self.df, int) or self.df not in (4, 8):
            raise ValueError("The declared Student-t family permits only integer df=4 or df=8")
        self._validate()

@dataclass(frozen=True)
class GaussianResidual(_ResidualMethods):
    scale_squared: tuple[float, ...]
    bias: tuple[float, ...] | None = None
    family = "gaussian_residual"
    df = None

    def __post_init__(self):
        self._validate()

def observation_law_from_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {
        "schema_version",
        "family",
        "df",
        "scale_squared",
        "bias",
        "parameterization",
    }:
        raise ValueError("Serialized law has missing or unexpected fields")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported observation-law schema")
    if spec["family"] == "product_student_t":
        result = ProductStudentT(spec["df"], spec["scale_squared"], spec["bias"])
    elif spec["family"] == "gaussian_residual" and spec["df"] is None:
        result = GaussianResidual(spec["scale_squared"], spec["bias"])
    else:
        raise ValueError("Unknown or inconsistent observation-law family")
    if result.to_spec() != spec:
        raise ValueError("Serialized parameterization is not the declared residual law")
    return result

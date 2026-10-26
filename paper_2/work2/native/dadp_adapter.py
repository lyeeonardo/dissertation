"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import importlib

import importlib.util

import sys

from contextlib import contextmanager

from functools import lru_cache

from pathlib import Path

from types import ModuleType, SimpleNamespace

import torch

from torch import nn

from torch.nn.attention import SDPBackend, sdpa_kernel

from .common import DEFAULT_OUTPUT

REPRESENTATION_NUMERICS = {
    "precision": "float32",
    "attention_backend": "math",
    "tf32": False,
}

@contextmanager
def representation_precision(device_type="cuda"):
    """Stable upstream representation execution; restore the caller's settings."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn

@lru_cache(maxsize=1)
def official_modules(source=None):
    root = Path(source or Path(__file__).resolve().parents[2] / "work2/vendor/DADP_official").resolve()
    if not (root / "dadp/dadp.py").exists():
        raise FileNotFoundError("The inspected official DADP source repository is required")

    # Upstream package __init__ files eagerly import unused image encoders,
    # torchvision and other optional packages. Load the actual selected modules
    # unchanged, with narrowly populated package exports instead.
    def package(name, path):
        if name in sys.modules:
            module = sys.modules[name]
            if str(path) not in getattr(module, "__path__", []):
                raise RuntimeError(f"Another {name} installation is already loaded")
            return module
        module = ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        module.__spec__ = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
        return module

    package("dadp", root / "dadp")
    representation = importlib.import_module("dadp.dadp")
    package("cleandiffuser", root / "cleandiffuser")
    utility_package = package("cleandiffuser.utils", root / "cleandiffuser/utils")
    utilities = importlib.import_module("cleandiffuser.utils.utils")
    for name in (
        "at_least_ndim",
        "to_tensor",
        "SUPPORTED_NOISE_SCHEDULES",
        "SUPPORTED_DISCRETIZATIONS",
        "SUPPORTED_SAMPLING_STEP_SCHEDULE",
        "SUPPORTED_TIMESTEP_EMBEDDING",
        "SinusoidalEmbedding",
    ):
        setattr(utility_package, name, getattr(utilities, name))
    for subpackage, filename, names in (
        ("nn_diffusion", "base_nn_diffusion", ("BaseNNDiffusion",)),
        ("nn_condition", "base_nn_condition", ("BaseNNCondition", "IdentityCondition")),
        ("nn_classifier", "base_nn_classifier", ("BaseNNClassifier",)),
        ("classifier", "base", ("BaseClassifier",)),
    ):
        target = package(f"cleandiffuser.{subpackage}", root / "cleandiffuser" / subpackage)
        module = importlib.import_module(f"cleandiffuser.{subpackage}.{filename}")
        for name in names:
            setattr(target, name, getattr(module, name))
    package("cleandiffuser.diffusion", root / "cleandiffuser/diffusion")
    diffusion = importlib.import_module("cleandiffuser.diffusion.diffusionsde")
    dit = importlib.import_module("cleandiffuser.nn_diffusion.dit")
    return representation, dit.DiT1d, diffusion.MixedDDIM

def make_representation(device="cuda", source=None, width=256, layers=4):
    upstream, _, _ = official_modules(source)
    embedding = upstream.EmbeddingConfig(
        embedding_size=23,
        d_model=width,
        n_layer=layers,
        n_head=8,
        d_ff=width * 4,
        dropout=0.1,
        adaptive_pooling_heads=8,
        adaptive_pooling_dropout=0.1,
        norm_z=True,
    )
    dynamics = upstream.DynamicsConfig(
        state_dim=17, action_dim=6, factor_dim=3, head_hidden=256, embedding_config=embedding
    )
    training = upstream.TrainingConfig(
        history=16,
        window_size=2,
        delta_t=1,
        inverse_loss_weight=1,
        forward_loss_weight=1,
        state_loss_weight=1,
        factor_loss_weight=1,
        policy_loss_weight=1,
        intra_traj_consistency_loss_weight=0,
        inter_traj_consistency_loss_weight=0,
        cross_prediction=True,
        detach_embedding_for_state=True,
        detach_embedding_for_factor=True,
        detach_embedding_for_policy=True,
        min_visible_length=16,
        learning_rate=3e-4,
        batch_size=128,
        device=str(device),
    )
    model = upstream.DADP(dynamics, training).to(device)
    # Recreate the optimizer after device transfer; preserve the upstream recipe.
    model.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    return model

def make_planner(device="cuda", source=None, width=256, depth=6):
    _, dit, mixed = official_modules(source)
    network = dit(
        23,
        emb_dim=width,
        d_model=width,
        n_heads=width // 32,
        depth=depth,
        timestep_emb_type="fourier",
        dropout=0.1,
        attn_mask=None,
    )
    fixed = torch.zeros(20, 23)
    fixed[:16] = 1
    fixed[16, :17] = 1
    return mixed(
        network,
        fix_mask=fixed,
        loss_weight=torch.ones(20, 23),
        ema_rate=0.9999,
        device=str(device),
        predict_noise=True,
        noise_schedule="linear",
        guide_noise_scale=0.1,
        predict_embedding=True,
    )

class DADPPolicy(nn.Module):
    """One joint sample, as in upstream evaluation; no true-condition input or critic."""

    def __init__(self, representation, planner, normalization):
        super().__init__()
        self.representation = representation
        self.planner = planner
        self.denoiser = planner.model
        self.denoiser_ema = planner.model_ema
        self.config = SimpleNamespace(history=17)
        device = next(representation.parameters()).device
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(normalization["observation_mean"], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "observation_std",
            torch.as_tensor(normalization["observation_std"], dtype=torch.float32, device=device),
        )

    @torch.no_grad()
    def act(self, history, seed=13):
        if history["observations"].shape[1] != 17 or history["actions"].shape[1] != 16:
            raise ValueError(
                "DADP requires 16 lagged state/action pairs plus the current observation"
            )
        self.representation.eval()
        self.planner.eval()
        device = self.observation_mean.device
        n = len(history["observations"])
        observations = (history["observations"] - self.observation_mean) / self.observation_std
        # Missing reset history is padded with the first observed state and zero
        # actions. No future state, label, switch flag or online fitting is used.
        mask = history["mask"].bool()
        first = mask.long().argmax(1)
        initial = observations[torch.arange(n, device=device), first]
        observations = torch.where(mask[..., None], observations, initial[:, None])
        with representation_precision(device.type):
            embedding = self.representation.dynamics.encode_history(
                observations[:, :-1], history["actions"]
            )
        prior = torch.zeros(n, 20, 23, device=device)
        prior[:, :16] = torch.cat((observations[:, :-1], history["actions"]), -1)
        prior[:, 16, :17] = observations[:, -1]
        # Upstream samplers use the global generator. Scope/restore it so extra
        # model queries cannot change other cases' random streams.
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            sampled, _ = self.planner.sample(
                prior,
                solver="ddim",
                n_samples=n,
                sample_steps=5,
                use_ema=True,
                temperature=1.0,
                w_cfg=1.0,
                embedding=embedding[:, None].expand(n, 20, 23),
            )
        actions = sampled[:, 16, 17:].clamp(-1, 1)
        if not torch.isfinite(actions).all():
            raise FloatingPointError("FATAL_DADP: nonfinite action from official sampler")
        return actions

"""Han-style DIOKO economic MPC on the common public-information benchmark.

The learned lifted dynamics and positive quadratic cost retain the architecture
of Han et al. (2024), arXiv:2405.12478.  The common benchmark supplies 57 public
measurements and normalized actions.  No simulator or privileged state enters
this module.  Training uses complete epochs; the default batch of 512 is an
explicit throughput adaptation of the paper's batch of 128.
"""

from __future__ import annotations

from dataclasses import dataclass

import json

import os

from pathlib import Path

import numpy as np

from scipy import optimize, sparse

import torch

from torch import nn

from torch.nn import functional as F

@dataclass(frozen=True)
class KoopmanConfig:
    observation_dim: int = 57
    action_dim: int = 2
    width: int = 128
    latent_dim: int = 60
    horizon: int = 30
    epochs: int = 400
    batch_size: int = 512
    learning_rate: float = 1e-3
    l2_coefficient: float = 0.1
    validation_windows: int = 500
    seed: int = 17

    def __post_init__(self):
        for key in ("observation_dim", "action_dim", "width", "latent_dim", "horizon",
                    "epochs", "batch_size", "validation_windows"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.action_dim != 2 or self.learning_rate <= 0 or self.l2_coefficient < 0:
            raise ValueError("Two actions, positive learning rate and nonnegative L2 required")

def valid_window_starts(episode_id, step, terminated, horizon):
    """Start indices of complete contiguous windows without episode crossings."""
    episode_id, step, terminated = map(np.asarray, (episode_id, step, terminated))
    n = len(episode_id)
    if horizon < 1 or step.shape != (n,) or terminated.shape != (n,):
        raise ValueError("Expected aligned episode, step and termination vectors")
    if n < horizon:
        return np.empty(0, dtype=np.int64)
    bad_edge = ((episode_id[1:] != episode_id[:-1]) |
                (step[1:] != step[:-1] + 1) | terminated[:-1].astype(bool))
    cumulative = np.r_[0, np.cumsum(bad_edge, dtype=np.int64)]
    starts = np.arange(n - horizon + 1, dtype=np.int64)
    return starts[cumulative[starts + horizon - 1] == cumulative[starts]]

def load_data(path, config):
    required = ("observations", "next_observations", "actions", "costs", "episode_id", "step", "terminated")
    with np.load(path, allow_pickle=False) as source:
        result = {key: np.array(source[key], copy=True) for key in required}
    n = len(result["costs"])
    for key, shape in (("observations", (n, config.observation_dim)),
                       ("next_observations", (n, config.observation_dim)),
                       ("actions", (n, config.action_dim)), ("costs", (n,))):
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key} shape or nonfinite value in {path}")
        result[key] = result[key].astype(np.float32, copy=False)
    if np.any(result["actions"] < -1e-6) or np.any(result["actions"] > 1 + 1e-6):
        raise ValueError("Common actions must be normalized into [0, 1]")
    result["starts"] = valid_window_starts(result["episode_id"], result["step"],
                                          result["terminated"], config.horizon)
    if not len(result["starts"]):
        raise ValueError(f"No complete {config.horizon}-step windows in {path}")
    return result

class DIOKO(nn.Module):
    def __init__(self, config=KoopmanConfig(), *, observation_mean=None,
                 observation_scale=None, cost_scale=1.):
        super().__init__()
        self.config = config
        c = config
        self.encoder = nn.Sequential(nn.Linear(c.observation_dim, c.width), nn.ELU(),
                                     nn.Linear(c.width, c.width), nn.ELU(),
                                     nn.Linear(c.width, c.latent_dim))
        self.A = nn.Parameter(.98 * torch.eye(c.latent_dim))
        self.B = nn.Parameter(.01 * torch.randn(c.latent_dim, c.action_dim))
        self.log_q = nn.Parameter(torch.full((c.latent_dim,), -3.))
        self.linear_cost = nn.Parameter(torch.zeros(c.latent_dim))
        self.cost_bias = nn.Parameter(torch.ones(()))
        self.register_buffer("observation_mean", torch.as_tensor(
            np.zeros(c.observation_dim) if observation_mean is None else observation_mean,
            dtype=torch.float32))
        self.register_buffer("observation_scale", torch.as_tensor(
            np.ones(c.observation_dim) if observation_scale is None else observation_scale,
            dtype=torch.float32))
        self.register_buffer("cost_scale", torch.as_tensor(float(cost_scale), dtype=torch.float32))
        if self.observation_mean.shape != (c.observation_dim,) or self.observation_scale.shape != (c.observation_dim,):
            raise ValueError("Observation normalization dimension mismatch")
        if not torch.all(self.observation_scale > 0) or not self.cost_scale > 0:
            raise ValueError("Normalization scales must be positive")

    def encode(self, observation):
        return self.encoder((observation - self.observation_mean) / self.observation_scale)

    def normalized_cost(self, latent):
        return (latent.square() * self.log_q.exp()).sum(-1) + latent @ self.linear_cost + self.cost_bias

    def rollout(self, initial, actions):
        """All future lifted states, vectorized across data and action time.

        The short matrix-power loop is independent of batch size. A temporal
        convolution evaluates the complete causal input-response sum on GPU.
        """
        if actions.ndim != 3 or initial.shape != (len(actions), self.config.latent_dim):
            raise ValueError("Expected initial [batch,latent] and actions [batch,time,2]")
        horizon = actions.shape[1]
        if horizon < 1 or actions.shape[-1] != self.config.action_dim:
            raise ValueError("Positive horizon and matching actions required")
        powers = [torch.eye(self.config.latent_dim, device=self.A.device, dtype=self.A.dtype)]
        for _ in range(horizon):
            powers.append(powers[-1] @ self.A)
        powers = torch.stack(powers)
        autonomous = torch.einsum("bq,tpq->btp", initial, powers[1:])
        response = powers[:-1] @ self.B
        kernel = response.flip(0).permute(1, 2, 0).contiguous()
        driven = F.conv1d(F.pad(actions.transpose(1, 2), (horizon - 1, 0)), kernel).transpose(1, 2)
        return autonomous + driven

    def loss(self, observations, actions, costs):
        """Observations contain root and H next packets; costs label H transitions."""
        if observations.shape[1] != actions.shape[1] + 1 or costs.shape != actions.shape[:2]:
            raise ValueError("Expected H+1 observations, H actions and H costs")
        encoded = self.encode(observations)
        predicted = self.rollout(encoded[:, 0], actions)
        # Sum lifted coordinates as in Eq.21, then average samples and time.
        transition = (predicted - encoded[:, 1:]).square().sum(-1).mean()
        cost = (self.normalized_cost(predicted) - costs / self.cost_scale).square().mean()
        # The paper does not identify regularized tensors. Use mean parameter
        # L2 over encoder/dynamics weights, excluding biases and economic head.
        parameters = [self.A, self.B] + [layer.weight for layer in self.encoder if isinstance(layer, nn.Linear)]
        regularization = sum(p.square().sum() for p in parameters) / sum(p.numel() for p in parameters)
        total = transition + cost + self.config.l2_coefficient * regularization
        return total, {"transition": transition, "cost": cost, "regularization": regularization,
                       "selection_score": transition + cost}

def condensed_qp(A, B, q, linear, horizon, move_weights):
    """Fixed matrices for a convex quadratic MPC with normalized box inputs."""
    A, B, q, linear, move_weights = [np.asarray(x, dtype=np.float64)
                                     for x in (A, B, q, linear, move_weights)]
    latent, actions = B.shape
    if horizon < 1 or A.shape != (latent, latent) or q.shape != (latent,) or np.any(q <= 0):
        raise ValueError("Invalid lifted system or positive quadratic diagonal")
    if linear.shape != (latent,) or move_weights.shape != (actions,) or np.any(move_weights < 0):
        raise ValueError("Invalid linear cost or move weights")
    powers = [np.eye(latent)]
    for _ in range(horizon):
        powers.append(powers[-1] @ A)
    initial_map = np.concatenate(powers[1:], axis=0)
    input_map = np.zeros((horizon * latent, horizon * actions))
    for t in range(horizon):
        for j in range(t + 1):
            input_map[t * latent:(t + 1) * latent, j * actions:(j + 1) * actions] = powers[t - j] @ B
    difference = np.eye(horizon * actions)
    difference[actions:, :-actions] -= np.eye((horizon - 1) * actions)
    q_all, r_all = np.tile(q, horizon), np.tile(move_weights, horizon)
    hessian = 2 * (input_map.T @ (q_all[:, None] * input_map) +
                   difference.T @ (r_all[:, None] * difference))
    hessian = (hessian + hessian.T) / 2
    linear_state = 2 * input_map.T @ (q_all[:, None] * initial_map)
    linear_fixed = input_map.T @ np.tile(linear, horizon)
    previous_map = np.zeros((horizon * actions, actions))
    previous_map[:actions] = -np.eye(actions)
    linear_previous = 2 * difference.T @ (r_all[:, None] * previous_map)
    if not all(np.isfinite(v).all() for v in (hessian, linear_state, linear_fixed, linear_previous)):
        raise ValueError("Nonfinite condensed QP")
    return hessian, linear_state, linear_fixed, linear_previous

def solve_box_qp(hessian, linear, initial=None):
    """Solve the actual convex box QP with analytic gradients (SciPy fallback)."""
    hessian, linear = np.asarray(hessian, dtype=float), np.asarray(linear, dtype=float)
    if hessian.shape != (len(linear), len(linear)) or not np.isfinite(hessian).all() or not np.isfinite(linear).all():
        raise ValueError("Finite aligned QP coefficients required")
    start = np.full(len(linear), .5) if initial is None else np.clip(initial, 0., 1.)
    def objective(x):
        product = hessian @ x
        return .5 * x @ product + linear @ x, product + linear
    result = optimize.minimize(objective, start, jac=True, method="L-BFGS-B",
        bounds=[(0., 1.)] * len(linear), options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8, maxls=40))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Convex box QP failed: {result.message}")
    return np.clip(result.x, 0., 1.), dict(backend="scipy_lbfgsb_convex_box_qp", iterations=int(result.nit))

class KoopmanController:
    """Public observation/action interface; no environment or truth references."""
    def __init__(self, model, *, solver="auto"):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.horizon = model.config.horizon
        limits = np.array([92230., 240.])
        move = np.array([1.2e-8, 1.77733e-5]) * limits**2
        move /= float(model.cost_scale)
        def array(tensor):
            return tensor.detach().cpu().numpy().astype(np.float64)
        self.hessian, self.linear_state, self.linear_fixed, self.linear_previous = condensed_qp(
            array(model.A), array(model.B), np.exp(array(model.log_q)), array(model.linear_cost), self.horizon, move)
        self.previous_plan = None
        self.last_diagnostics = {}
        self.qp = None
        if solver not in ("auto", "scipy", "osqp"):
            raise ValueError("Unknown QP solver")
        if solver != "scipy":
            try:
                import osqp
            except ImportError:
                if solver == "osqp":
                    raise
            else:
                self.qp = osqp.OSQP()
                dimension = len(self.hessian)
                self.qp.setup(P=sparse.csc_matrix(np.triu(self.hessian)), q=np.zeros(dimension),
                    A=sparse.eye(dimension, format="csc"), l=np.zeros(dimension), u=np.ones(dimension),
                    verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, polishing=True)

    @torch.no_grad()
    def act(self, observation, previous_action):
        observation, previous_action = np.asarray(observation), np.asarray(previous_action)
        if observation.shape != (self.model.config.observation_dim,) or previous_action.shape != (2,):
            raise ValueError("Expected one public observation and previous normalized action")
        if not np.isfinite(observation).all() or not np.isfinite(previous_action).all():
            raise ValueError("Public inputs must be finite")
        if np.any(previous_action < -1e-6) or np.any(previous_action > 1 + 1e-6):
            raise ValueError("Previous action must be normalized into [0,1]")
        latent = self.model.encode(torch.as_tensor(observation, device=self.device, dtype=torch.float32)).cpu().numpy()
        linear = self.linear_state @ latent + self.linear_fixed + self.linear_previous @ previous_action
        initial = None if self.previous_plan is None else np.r_[self.previous_plan[2:], self.previous_plan[-2:]]
        if self.qp is None:
            plan, diagnostic = solve_box_qp(self.hessian, linear, initial)
        else:
            self.qp.update(q=linear)
            if initial is not None:
                self.qp.warm_start(x=initial)
            solution = self.qp.solve(raise_error=False)
            if solution.info.status_val not in (1, 2) or not np.isfinite(solution.x).all():
                raise RuntimeError(f"OSQP failed: {solution.info.status}")
            plan = np.clip(solution.x, 0., 1.)
            diagnostic = dict(backend="osqp", iterations=int(solution.info.iter), status=solution.info.status,
                              primal_residual=float(solution.info.prim_res), dual_residual=float(solution.info.dual_res))
        self.previous_plan = plan.copy()
        self.last_diagnostics = diagnostic
        return plan[:2].astype(np.float64, copy=True)

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def device_data(data, device):
    return {key: torch.as_tensor(data[key], device=device)
            for key in ("observations", "next_observations", "actions", "costs")}

def gather_windows(data, starts, horizon):
    indices = starts[:, None] + torch.arange(horizon, device=starts.device)[None]
    observations = torch.cat([data["observations"][starts, None], data["next_observations"][indices]], dim=1)
    return observations, data["actions"][indices], data["costs"][indices]

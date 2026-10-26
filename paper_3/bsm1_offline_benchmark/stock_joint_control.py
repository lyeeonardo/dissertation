"""Frozen joint diffusion control with oracle truth or continuous public beliefs.

Eight particles approximate continuous missing state coordinates, not plant modes.
Denoising energies define a fixed-support generalized Bayes update and surrogate
VFE; the one-step entropy contrast is not asserted to be exact mutual information.
No simulator, future forcing, external critic or online model optimizer is used.
"""

from __future__ import annotations

import copy

from pathlib import Path

import time

import numpy as np

import torch

from .joint_diffusion import load_model, STATE_DIM, PUBLIC_DIM, JOINT_DIM

from .stock_shared_data import PUBLIC_INDICES

EVIDENCE_TIMES = (.2, .4, .6, .8)

INITIAL_PREVIOUS_ACTION = np.array([.6, .35])

def entropy(weights):
    weights = torch.as_tensor(weights, dtype=torch.float64)
    return -torch.where(weights > 0, weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log(), 0.).sum(-1)

def posterior_from_energy(prior, energy, tau):
    """Exact minimizer of KL(q||prior)+E_q[energy]/tau on this fixed support."""
    if tau is None or not np.isfinite(tau) or tau <= 0:
        raise ValueError("A positive CAL-frozen evidence temperature is required")
    energy = torch.as_tensor(energy, dtype=torch.float64)
    prior = torch.as_tensor(prior, device=energy.device, dtype=torch.float64)
    if prior.shape[-1] != energy.shape[-1] or not torch.isfinite(energy).all():
        raise ValueError("Finite energies must align with the fixed prior support")
    if not torch.isfinite(prior).all() or (prior < 0).any() or not torch.allclose(prior.sum(-1), torch.ones_like(prior.sum(-1)), atol=1e-10, rtol=1e-10):
        raise ValueError("Prior weights must be nonnegative and sum to one")
    log_prior = torch.where(prior > 0, prior.log(), -torch.inf)
    logits = log_prior - energy / float(tau)
    log_z = torch.logsumexp(logits, -1)
    log_q = logits - log_z[..., None]
    q = log_q.exp()
    kl = torch.where(q > 0, q * (log_q-log_prior), 0.).sum(-1)
    expected_energy = (q * energy).sum(-1)
    value = kl + expected_energy / float(tau)
    return q, dict(vfe=value, prior_vfe=(prior * energy).sum(-1) / float(tau),
                   minimum_vfe=-log_z, kl=kl, expected_energy=expected_energy,
                   entropy=entropy(q), ess=1 / q.square().sum(-1))

def _packet(value, dimensions, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (dimensions,) or not np.isfinite(value).all():
        raise ValueError(f"{name} requires {dimensions} finite current values")
    return value.copy()

def _action(value):
    value = _packet(value, 2, "Executed action")
    if np.any(value < 0) or np.any(value > 1):
        raise ValueError("Executed action must be in the normalized physical box")
    return value

def _repeat_history(history, count):
    return {key: value.expand(count, *value.shape[1:]) for key, value in history.items()}

def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return copy.deepcopy(value)

class JointController:
    def __init__(self, checkpoint, device="cuda", *, mode="genaif", history_length=8,
                 passes=20, candidates=3, belief_update=True, refine=False,
                 information_weight=0., tau=None, seed=17, particles=8,
                 horizon=30, discount=.99, remaining=1344):
        if mode not in ("genaif", "oracle", "history") or history_length not in (1, 8, 24):
            raise ValueError("Choose oracle/genaif/history and public history1/8/24")
        if mode == "history" and (belief_update or information_weight != 0):
            raise ValueError("History-only initialization requires no VFE update or information term")
        if passes not in (10, 20) or candidates < 1 or not 1 <= horizon <= 30:
            raise ValueError("Use10/20 denoising passes and a valid candidate count/H30 prefix")
        if mode != "oracle" and (particles != 8 or candidates != 3):
            raise ValueError("This bounded GenAIF recipe fixes8 continuous hypotheses and3 proposals")
        if mode == "oracle" and candidates not in (3, 64):
            raise ValueError("Oracle uses the matched3 or capability64 candidate recipe")
        if refine and candidates != 3:
            raise ValueError("Refinement retains one original plus two partial-denoising candidates")
        if not np.isfinite(information_weight) or information_weight < 0 or not 0 < discount <= 1:
            raise ValueError("Finite nonnegative information weight and valid preference discount required")
        if mode == "genaif" and (belief_update or information_weight > 0) and (tau is None or not np.isfinite(tau) or tau <= 0):
            raise ValueError("GenAIF evidence requires a positive CAL-frozen tau; no default temperature")
        if tau is not None and (not np.isfinite(tau) or tau <= 0):
            raise ValueError("tau must be positive")
        if int(remaining) != remaining or remaining < 1:
            raise ValueError("A positive number of remaining public evaluation decisions is required")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = torch.device(device)
        self.model = load_model(checkpoint, device).requires_grad_(False).eval()
        self.mode, self.history_length, self.passes = mode, history_length, passes
        self.candidates, self.belief_update, self.refinement = candidates, bool(belief_update), bool(refine)
        self.information_weight, self.tau = float(information_weight), None if tau is None else float(tau)
        self.seed, self.particle_count, self.horizon = int(seed), 1 if mode == "oracle" else particles, int(horizon)
        self.discount, self.remaining = float(discount), int(remaining)
        self.cost_scale = float(self.model.cost_scale)
        if not np.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("The frozen model must supply a positive TRAIN cost mean")
        self.public_indices = torch.as_tensor(PUBLIC_INDICES, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.history_records = []
        self.particles = None
        self.weights = None
        self.pending = None
        self.act_count, self.transition_count = 0, 0
        self.last_diagnostics = {}

    def _generator(self, stream):
        # Counter-based streams keep the next step's original sample matched
        # when another arm enables information scoring or partial refinements.
        offsets = dict(initial=101, hypothesis=1009, proposal=2003, forecast=3001,
                       real_evidence=4001, imagined_evidence=5003)
        value = (self.seed + offsets[stream] + 104729*self.transition_count) % (2**63-1)
        if stream == "proposal":
            self.generator.manual_seed(value)
            return self.generator
        return torch.Generator(device=self.device).manual_seed(value)

    def _history(self):
        if not self.history_records:
            raise RuntimeError("Start with a real public root observation")
        records = self.history_records[-self.history_length:]
        length = self.model.config.history
        obs = torch.zeros(1, length, PUBLIC_DIM, device=self.device)
        actions = torch.zeros(1, length, 2, device=self.device)
        rewards = torch.zeros(1, length, 1, device=self.device)
        valid = torch.zeros(1, length, dtype=torch.bool, device=self.device)
        reward_valid = torch.zeros_like(valid)
        for position, row in enumerate(records, length-len(records)):
            obs[0, position] = torch.as_tensor(row["public"], device=self.device)
            actions[0, position] = torch.as_tensor(row["previous_action"], device=self.device)
            rewards[0, position, 0] = row["previous_reward"]
            valid[0, position], reward_valid[0, position] = True, row["reward_valid"]
        return dict(observations=obs, previous_actions=actions, previous_rewards=rewards,
                    valid=valid, reward_valid=reward_valid)

    @torch.inference_mode()
    def start(self, public55, previous_action=None):
        """Initialize a saved-public-trace replay without choosing any action."""
        if self.history_records:
            raise RuntimeError("Controller already started; do not reset or re-assimilate a transition")
        observation = _packet(public55, PUBLIC_DIM, "Public observation")
        previous = _action(INITIAL_PREVIOUS_ACTION if previous_action is None else previous_action)
        self.history_records.append(dict(public=observation, previous_action=previous,
                                         previous_reward=0., reward_valid=False))
        if self.mode != "oracle":
            initialized = self.model.initialize(self._history(), samples=self.particle_count,
                                                generator=self._generator("initial"))
            self.particles = initialized["samples"][0].clone()
            self.particles[:, self.public_indices] = torch.as_tensor(observation, device=self.device)
            self.weights = torch.full((self.particle_count,), 1/self.particle_count,
                                      device=self.device, dtype=torch.float64)
        self.last_diagnostics = dict(mode=self.mode, transition_count=0, initialized=True,
                                     public_history_available=1, current_truth_access=False)

    def _refresh_history_initializer(self, history):
        if self.mode == "history" and self.transition_count > 0:
            initialized = self.model.initialize(history, samples=self.particle_count,
                                                generator=self._generator("initial"))
            self.particles = initialized["samples"][0].clone()
            self.particles[:, self.public_indices] = history["observations"][0, -1]
            self.weights = torch.full((self.particle_count,), 1/self.particle_count,
                                      device=self.device, dtype=torch.float64)

    def _root(self, observation, previous_action, current_truth):
        if self.mode != "oracle" and current_truth is not None:
            raise ValueError("GenAIF cannot receive evaluator current truth")
        observation = _packet(observation, PUBLIC_DIM, "Public observation")
        if not self.history_records:
            self.start(observation, previous_action)
        row = self.history_records[-1]
        if not np.array_equal(observation, row["public"]):
            raise ValueError("New observations must enter through observe, exactly once")
        if previous_action is not None and not np.allclose(_action(previous_action), row["previous_action"], atol=1e-7, rtol=0):
            raise ValueError("Previous command disagrees with the recorded executed action")
        if self.mode == "oracle":
            if current_truth is None:
                raise ValueError("Oracle requires its current159 packet at every act call")
            truth = _packet(current_truth, STATE_DIM, "Oracle current truth")
            if not np.allclose(truth[PUBLIC_INDICES], observation, atol=1e-5, rtol=1e-6):
                raise ValueError("Oracle current truth/public observation do not describe the same root")
            self.particles = torch.as_tensor(truth, device=self.device)[None]
            self.weights = torch.ones(1, dtype=torch.float64, device=self.device)
        history = self._history()
        self._refresh_history_initializer(history)
        return history

    def _proposal_bank(self, history, horizon):
        proposal_rng = self._generator("proposal")
        hypothesis_rng = self._generator("hypothesis")
        def generating_index():
            draw = torch.rand((), device=self.device, generator=hypothesis_rng, dtype=torch.float64)
            return min(int(torch.searchsorted(self.weights.cumsum(0), draw)), len(self.weights)-1)
        first_index = generating_index()
        first = self.model.sample_proposals(self.particles[first_index:first_index+1], history,
            quality=1., passes=self.passes, horizon=horizon, generator=proposal_rng)
        action_chunks, indices = [first["actions"]], [first_index]
        if self.refinement:
            # Both partial denoisings start from the retained original sample,
            # and run together on GPU; neither starts from the other refinement.
            refined = self.model.refine(self.particles[first_index:first_index+1].expand(2, -1),
                _repeat_history(history, 2), first["clean"].expand(2, -1, -1),
                quality=1., passes=self.passes, midpoint=self.passes//2, generator=proposal_rng)
            action_chunks.append(refined["actions"])
            indices.extend([first_index, first_index])
        else:
            # Independent complete proposals; the original draw above is shared
            # with the refined arm. All remaining candidates are one GPU batch.
            count = self.candidates-1
            draws = torch.rand(count, device=self.device, generator=hypothesis_rng, dtype=torch.float64)
            generated = torch.searchsorted(self.weights.cumsum(0), draws).clamp_max(len(self.weights)-1)
            rest = self.model.sample_proposals(self.particles[generated], _repeat_history(history, count),
                quality=1., passes=self.passes, horizon=horizon, generator=proposal_rng)
            action_chunks.append(rest["actions"])
            indices.extend(generated.cpu().tolist())
        actions = torch.cat(action_chunks)
        if actions.shape != (self.candidates, horizon, 2) or not torch.isfinite(actions).all():
            raise FloatingPointError("Invalid joint proposal action bank")
        if bool(((actions < 0) | (actions > 1)).any()):
            raise ValueError("Proposal left the original physical actuator box")
        return actions, indices

    def _forecast(self, history, actions):
        candidates, horizon = actions.shape[:2]
        count = len(self.weights)
        roots = self.particles[None].expand(candidates, -1, -1).reshape(candidates*count, STATE_DIM)
        imposed = actions[:, None].expand(-1, count, -1, -1).reshape(candidates*count, horizon, 2)
        generator = self._generator("forecast")
        shape = (1, self.model.config.horizon, JOINT_DIM)
        noise = torch.randn(shape, device=self.device, generator=generator)
        known_noise = torch.randn(shape, device=self.device, generator=generator)
        prediction = self.model.predict_actions(roots, _repeat_history(history, len(roots)), imposed,
            passes=self.passes, generator=generator, initial_noise=noise.expand(len(roots), -1, -1),
            known_noise=known_noise.expand(len(roots), -1, -1))
        result = {key: prediction[key].reshape(candidates, count, horizon, -1)
                  for key in ("next_private", "public")}
        result["rewards"] = prediction["rewards"].reshape(candidates, count, horizon)
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite imposed-action physical forecast")
        return result

    def _information(self, history, actions, prediction):
        c, k = actions.shape[0], len(self.weights)
        if self.information_weight == 0 or k == 1:
            return torch.zeros(c, dtype=torch.float64, device=self.device), None
        # Dimensions: candidate, generating/root-ancestry hypothesis, scored
        # hypothesis. Every imagined event starts with the same real prior.
        roots = self.particles[None, None].expand(c, k, k, STATE_DIM).reshape(c*k*k, STATE_DIM)
        public = prediction["public"][:, :, 0, None].expand(c, k, k, PUBLIC_DIM).reshape(c*k*k, PUBLIC_DIM)
        rewards = prediction["rewards"][:, :, 0, None].expand(c, k, k).reshape(c*k*k)
        first_actions = actions[:, 0, None, None].expand(c, k, k, 2).reshape(c*k*k, 2)
        noise = torch.randn((len(EVIDENCE_TIMES), PUBLIC_DIM+3), device=self.device,
                            generator=self._generator("imagined_evidence"))
        evidence = self.model.evidence(roots, _repeat_history(history, len(roots)), public,
            rewards, first_actions, times=EVIDENCE_TIMES, noise=noise)
        energy = evidence["energy"].reshape(c, k, k).to(torch.float64)
        posterior, values = posterior_from_energy(self.weights, energy, self.tau)
        expected_entropy = (values["entropy"] * self.weights[None]).sum(-1)
        information = entropy(self.weights) - expected_entropy  # negative estimates retained
        return information, dict(expected_posterior_entropy=expected_entropy,
            minimum=energy.amin((1, 2)), maximum=energy.amax((1, 2)),
            mean_posterior_ess=(values["ess"]*self.weights[None]).sum(-1))

    @torch.inference_mode()
    def act(self, public55, previous_action=None, current_truth=None):
        if self.pending is not None:
            raise RuntimeError("Observe the previous executed transition before selecting again")
        if self.remaining <= 0:
            raise RuntimeError("Known evaluation episode has ended")
        started = time.perf_counter()
        history = self._root(public55, previous_action, current_truth)
        # The plant is continuing; the reporting record's end is not a physical
        # terminal. Stock Han and this controller retain H30 at every decision.
        horizon = self.horizon
        actions, generating_indices = self._proposal_bank(history, horizon)
        prediction = self._forecast(history, actions)
        discounts = self.discount ** torch.arange(horizon, device=self.device, dtype=torch.float64)
        rewards = (prediction["rewards"].to(torch.float64) * discounts).sum(-1)
        expected_return = (rewards * self.weights[None]).sum(-1)
        information, info_details = self._information(history, actions, prediction)
        efe = -expected_return - self.information_weight*information
        selected = int(efe.argmin())
        chosen = actions[selected, 0].cpu().numpy().astype(np.float64)
        self.pending = dict(particles=self.particles.clone(), weights=self.weights.clone(),
            history={key: value.clone() for key, value in history.items()}, action=chosen.copy(),
            next_particles=prediction["next_private"][selected, :, 0].clone())
        self.act_count += 1
        self.last_diagnostics = dict(mode=self.mode, act_count=self.act_count,
            transition_count=self.transition_count, current_truth_access=self.mode == "oracle",
            history_requested=self.history_length, history_available=int(history["valid"].sum()),
            horizon=horizon, remaining=self.remaining, passes=self.passes, candidates=self.candidates,
            continuous_hypotheses=len(self.weights), refinement=self.refinement,
            belief_source=("current_oracle_truth" if self.mode == "oracle" else
                "history_initializer_each_decision" if self.mode == "history" else
                "propagated_VFE" if self.belief_update else "propagated_prior"),
            refinement_midpoint=self.passes//2 if self.refinement else None,
            refinement_time=float(self.model.config.maximum_time/2) if self.refinement else None,
            partial_updates=self.passes-self.passes//2 if self.refinement else None,
            proposal_denoiser_evaluations=self.passes+2*(self.passes-self.passes//2) if self.refinement else self.candidates*self.passes,
            forecast_denoiser_evaluations=self.candidates*len(self.weights)*self.passes,
            proposal_generating_hypotheses=generating_indices, original_candidate=0,
            proposal_batch_sizes=[1, 2 if self.refinement else self.candidates-1],
            candidate_first_actions=actions[:, 0].cpu().tolist(),
            predicted_returns=expected_return.cpu().tolist(), information_l1=information.cpu().tolist(),
            efe=efe.cpu().tolist(), selected_candidate=selected,
            reward_action_contrast=float(expected_return.max()-expected_return.min()),
            information_action_contrast=float(information.max()-information.min()),
            prior_weights=self.weights.cpu().tolist(), belief_entropy=float(entropy(self.weights)),
            belief_ess=float(1/self.weights.square().sum()), tau=self.tau,
            information_weight=self.information_weight, finite_preference_discount=self.discount,
            decision_seconds=time.perf_counter()-started,
            uncertainty_definition="finite continuous-state hypotheses; no categorical hazard",
            information_definition="one-step weighted entropy reduction from common prior; negative values retained")
        if info_details is not None:
            self.last_diagnostics["information_details"] = {key: value.cpu().tolist() for key, value in info_details.items()}
        return chosen

    @torch.inference_mode()
    def prepare_transition(self, executed_action):
        """Predict one prescribed action for belief-only replay, without selection."""
        if not self.history_records or self.particles is None or self.remaining <= 0:
            raise RuntimeError("A live initialized root is required")
        if self.pending is not None:
            raise RuntimeError("The pending real transition must be observed first")
        action = _action(executed_action)
        history = self._history()
        self._refresh_history_initializer(history)
        prediction = self._forecast(history, torch.as_tensor(action, device=self.device)[None, None])
        self.pending = dict(particles=self.particles.clone(), weights=self.weights.clone(),
            history={key: value.clone() for key, value in history.items()}, action=action.copy(),
            next_particles=prediction["next_private"][0, :, 0].clone())
        return prediction["public"][0, :, 0].cpu().numpy().copy()

    @torch.inference_mode()
    def observe(self, next_public55, executed_action, raw_cost):
        """Assimilate one real transition using pre-transition history exactly once."""
        if self.pending is None:
            raise RuntimeError("One act or prepare_transition is required for each new observe; no duplicate update")
        if not self.history_records or self.particles is None or self.remaining <= 0:
            raise RuntimeError("A live initialized root is required before observe")
        observation, action = _packet(next_public55, PUBLIC_DIM, "Next public observation"), _action(executed_action)
        if not np.isfinite(raw_cost):
            raise ValueError("Observed interval cost must be finite")
        reward = -float(raw_cost) / self.cost_scale
        history, roots, prior = (self.pending[key] for key in ("history", "particles", "weights"))
        if not np.allclose(action, self.pending["action"], atol=1e-7, rtol=0):
            imposed = torch.as_tensor(action, device=self.device)[None, None]
            next_particles = self._forecast(history, imposed)["next_private"][0, :, 0]
        else:
            next_particles = self.pending["next_particles"]
        diagnostic = dict(observed_transition=self.transition_count, raw_cost=float(raw_cost),
            normalized_reward=reward, applied_update=False, prior_entropy=float(entropy(prior)),
            evidence_uses_pre_transition_history=True)
        if self.mode == "genaif" and self.belief_update:
            noise = torch.randn((len(EVIDENCE_TIMES), PUBLIC_DIM+3), device=self.device,
                                generator=self._generator("real_evidence"))
            result = self.model.evidence(roots, _repeat_history(history, len(roots)), observation,
                reward, action, times=EVIDENCE_TIMES, noise=noise)
            energies = result["energy"].to(torch.float64)
            updated, values = posterior_from_energy(prior, energies, self.tau)
            self.weights = updated if self.belief_update else prior.clone()
            diagnostic.update(applied_update=self.belief_update, energies=energies.cpu().tolist(),
                prior_vfe=float(values["prior_vfe"]), optimal_surrogate_vfe=float(values["vfe"]),
                applied_vfe=float(values["vfe"] if self.belief_update else values["prior_vfe"]),
                optimum_identity_residual=float(values["vfe"]-values["minimum_vfe"]),
                posterior_kl=float(values["kl"]) if self.belief_update else 0.,
                evidence_energy_mean=float(energies.mean()), evidence_energy_span=float(energies.max()-energies.min()))
        else:
            self.weights = prior.clone()
        self.particles = next_particles.clone()
        self.particles[:, self.public_indices] = torch.as_tensor(observation, device=self.device)
        self.history_records.append(dict(public=observation, previous_action=action,
                                         previous_reward=reward, reward_valid=True))
        self.history_records = self.history_records[-self.model.config.history:]
        self.pending = None
        self.transition_count += 1
        self.remaining -= 1
        diagnostic.update(posterior_weights=self.weights.cpu().tolist(),
            posterior_entropy=float(entropy(self.weights)), posterior_ess=float(1/self.weights.square().sum()),
            public_coordinates_clamped=True, remaining=self.remaining)
        self.last_diagnostics = dict(self.last_diagnostics, observed_transition=diagnostic,
                                      transition_count=self.transition_count)
        return copy.deepcopy(diagnostic)

    def runtime_state(self):
        return _cpu_tree(dict(format="stock_joint_control_v1", checkpoint=self.checkpoint,
            config=dict(mode=self.mode, history_length=self.history_length, passes=self.passes,
                candidates=self.candidates, belief_update=self.belief_update, refine=self.refinement,
                information_weight=self.information_weight, tau=self.tau, seed=self.seed,
                particles=self.particle_count, horizon=self.horizon, discount=self.discount),
            history_records=self.history_records, particles=self.particles, weights=self.weights,
            pending=self.pending, act_count=self.act_count, transition_count=self.transition_count,
            remaining=self.remaining, generator=self.generator.get_state(), last_diagnostics=self.last_diagnostics))

    def restore_runtime_state(self, state):
        if "joint_state" in state:
            state = state["joint_state"]
        current = self.runtime_state()
        if state.get("format") != current["format"] or state.get("checkpoint") != self.checkpoint or state.get("config") != current["config"]:
            raise ValueError("Runtime state must match the same frozen joint controller recipe")
        self.history_records = copy.deepcopy(state["history_records"])
        self.particles = None if state["particles"] is None else state["particles"].to(self.device)
        self.weights = None if state["weights"] is None else state["weights"].to(self.device)
        self.pending = copy.deepcopy(state["pending"])
        if self.pending is not None:
            for key in ("particles", "weights", "next_particles"):
                self.pending[key] = self.pending[key].to(self.device)
            self.pending["history"] = {key: value.to(self.device) for key, value in self.pending["history"].items()}
        self.act_count, self.transition_count = state["act_count"], state["transition_count"]
        self.remaining = state["remaining"]
        self.generator.set_state(state["generator"].cpu())
        self.last_diagnostics = copy.deepcopy(state["last_diagnostics"])

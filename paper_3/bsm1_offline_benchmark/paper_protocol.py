"""A declared reconstruction of Han et al. (2024), not recovered author code.

The constants below are transcribed from Tables 3--8 of arXiv:2405.12478v2.
The 41 sensor indices, noise covariance convention, endpoint accounting and
normalization are not specified by the manuscript; those choices stay explicit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import json

import os

from pathlib import Path

import time

import numpy as np

from bsm1_repro.plant import StreamingPlant

from bsm1_repro.rewards import inventory, rates

from bsm2_python.bsm2.settler1d_bsm2 import get_output

from simulator.canonical import plant_state

from simulator.inputs import constant_input

from .environment import ACTION_SCALE, CanonicalBSM1Env, SENSOR_NAMES, DISTURBANCE_NAMES, sensor41

PAPER_RETURN_FLOW = 18846.0

PAPER_WASTE_FLOW = 385.0

PAPER_OBSERVATION_NAMES = SENSOR_NAMES + DISTURBANCE_NAMES

PAPER_DECISIONS = 14 * 24 * 4

PAPER_RATE_NAMES = ("AE_kWh_per_day", "PE_kWh_per_day", "ME_kWh_per_day", "EQI_kgPU_per_day", "waste_kgSS_per_day")

PAPER_REACTOR_STATES = np.array([
    [30, 30, 30, 30, 30],
    [3.24, 1.67, 1.22, .97, .81],
    [1149.21, 1149.21, 1149.21, 1149.21, 1149.21],
    [98.60, 91.70, 69.69, 54.45, 44.48],
    [2552.12, 2552.39, 2560.22, 2563.33, 2562.87],
    [151.67, 151.53, 152.69, 153.71, 154.17],
    [446.96, 448.12, 449.67, 451.22, 452.77],
    [7.696e-3, 6.027e-5, 1.63, 2.47, 2.00],
    [3.51, 1.00, 6.23, 11.07, 13.52],
    [11.83, 12.55, 7.32, 2.78, .67],
    [1.36, .79, .83, .75, .66],
    [6.18, 5.95, 4.71, 3.84, 3.26],
    [5.34, 5.57, 4.82, 4.15, 3.83],
], dtype=np.float64)

PAPER_SETTLER_BOTTOM_TO_TOP = np.array([
    [30.] * 10, [.808] * 10, [2.] * 10, [13.52] * 10,
    [.67] * 10, [.66] * 10, [3.83] * 10,
    [6399.44, 356.29, 356.29, 356.29, 356.29, 356.29, 69., 29.55, 18.12, 12.50],
], dtype=np.float64)

def published_initial_state():
    """Construct consistent state and algebraic streams without advancing time.

    Rounded paper states are used literally. Initial recycle and aeration
    commands are absent from the table, so the standard 55338/84 commands are
    declared bookkeeping values until the first controller decision.
    """
    plant = StreamingPlant()
    inlet = constant_input(.01)[0, 1:].copy()
    inlet[14] = 18446.0
    plant.qintr = 55338.0
    plant.klas = np.array([0., 0., 240., 240., 84.])
    plant.settler.q_r = PAPER_RETURN_FLOW
    plant.settler.q_w = PAPER_WASTE_FLOW
    for index in range(1, 6):
        reactor = getattr(plant, f"reactor{index}")
        state = reactor.y0.copy()
        state[:13] = PAPER_REACTOR_STATES[:, index - 1]
        state[13] = .75 * state[2:7].sum()
        state[14] = inlet[14] + PAPER_RETURN_FLOW + plant.qintr
        reactor.y0 = state.copy()
        setattr(plant, f"y_out{index}", state.copy())
    plant.settler.ys0[:80] = PAPER_SETTLER_BOTTOM_TO_TOP[:, ::-1].reshape(-1)
    plant.ys_in = plant.y_out5.copy()
    plant.ys_in[14] = inlet[14] + PAPER_RETURN_FLOW
    plant.y_out5_r = plant.y_out5.copy()
    plant.y_out5_r[14] = plant.qintr
    (plant.ys_out, plant.ys_was, plant.ys_eff,
     plant.sludge_height, plant.ys_tss_internal) = get_output(
        plant.settler.ys0, plant.ys_in, int(plant.settler.layer[1]),
        plant.settler.tempmodel, PAPER_RETURN_FLOW, PAPER_WASTE_FLOW,
        plant.settler.dim, plant.settler.asm1par, plant.settler.sedpar)
    plant.y_in1 = plant.combiner.output(inlet, plant.ys_out, plant.y_out5_r)
    return plant_state(plant)

class PaperBSM1Env(CanonicalBSM1Env):
    """Clean 41-channel declared sensor reconstruction plus 14 current inlet.

    The physical task retains the full box of the paper. No support restriction
    or artificial aeration constraint is added by this protocol.
    """
    observation_dim = 55

    def __init__(self, weather="dry", *, seed=17, episode_days=14., **kwargs):
        if {"initial_state", "sensor_noise", "return_sludge", "waste_sludge"} & set(kwargs):
            raise ValueError("The paper protocol fixes initial state, clean sensors and return flow")
        super().__init__(weather=weather, initial_state=published_initial_state(),
                         sensor_noise=False, return_sludge=PAPER_RETURN_FLOW,
                         waste_sludge=PAPER_WASTE_FLOW,
                         seed=seed, episode_days=episode_days, **kwargs)

    def clean_observation(self):
        inlet = self._inlet(self.minute / 1440)
        return np.concatenate((sensor41(self.plant), inlet[:13], [inlet[14]]))

class PaperExcitation:
    """Paper Section 5.1 excitation, with explicitly chosen SD and clipping.

    N(0, 0.05*u_bar) is interpreted as per-coordinate standard deviation equal
    to five percent of the current block mean, in normalized action units.
    Executed commands are clipped to the published action box.
    """
    def __init__(self, seed=17):
        self.rng = np.random.default_rng(seed)
        self.mean = None
        self.step = 0

    def action(self):
        if self.step % 20 == 0:
            self.mean = self.rng.uniform(0., 1., 2)
        result = np.clip(self.mean + self.rng.normal(size=2) * .05 * self.mean, 0., 1.)
        self.step += 1
        return result

def paper_collection_manifest(seed=17, samples=100000):
    """Transition interpretation of the paper's 100k samples, preserving resets.

    A random official weather is chosen for each 14-day episode. The last
    episode is truncated to reach exactly 100k. The split indices are the paper's
    sequential 80/10/10 rule; temporal windows must stay inside an episode and
    split. There are no new parameter seeds or test-selected choices.
    """
    if samples != 100000:
        raise ValueError("Published protocol requires exactly 100000 transition samples")
    rng = np.random.default_rng(seed)
    episodes, offset = [], 0
    while offset < samples:
        count = min(PAPER_DECISIONS, samples - offset)
        episodes.append(dict(episode_id=len(episodes), weather=str(rng.choice(["dry", "rain", "storm"])),
                             seed=int(rng.integers(0, 2**31 - 1)), steps=count,
                             start=offset, stop=offset + count))
        offset += count
    return dict(schema="han2024_paper_style_transition_protocol_v1", seed=seed,
                samples=samples, split_ranges={"train": [0, 80000], "validation": [80000, 90000],
                                              "test": [90000, 100000]},
                episode_days=14., decision_minutes=15, episodes=episodes,
                observation_dim=55, noise=False, return_flow=PAPER_RETURN_FLOW,
                waste_flow=PAPER_WASTE_FLOW, action_scale=ACTION_SCALE.tolist(),
                sample_interpretation="individual transitions; paper does not disambiguate sequence samples",
                noise_interpretation="SD = 5 percent of block mean, then clip to action box")

def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)

def collect_paper_episode(output, episode):
    """One CPU physical episode. Existing results are preserved, never retried."""
    path = Path(output) / "episodes" / f"episode_{episode['episode_id']:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".json")
    if receipt.exists() or path.exists():
        raise FileExistsError(f"Preserve existing episode; automatic resume is disabled: {path}")
    started = time.time()
    _atomic_json(receipt, dict(status="running", episode=episode, started_unix=started))
    env = PaperBSM1Env(episode["weather"], seed=episode["seed"])
    controller = PaperExcitation(episode["seed"])
    rows = {key: [] for key in ("observations", "next_observations", "actions", "costs", "components",
                               "privileged", "next_privileged", "previous_actions", "root_rates",
                               "next_rates", "root_inventory", "next_inventory")}
    try:
        for index in range(episode["steps"]):
            root_rates, root_inventory = rates(env.plant), inventory(env.plant)
            env.step(controller.action())
            t = env.last_transition
            for target, source in (("observations", "observation"), ("next_observations", "next_observation"),
                                   ("actions", "action"), ("costs", "cost"), ("components", "components"),
                                   ("privileged", "privileged"), ("next_privileged", "next_privileged"),
                                   ("previous_actions", "previous_action")):
                rows[target].append(t[source])
            endpoint_rates = rates(env.plant)
            rows["root_rates"].append([root_rates[key] for key in PAPER_RATE_NAMES])
            rows["next_rates"].append([endpoint_rates[key] for key in PAPER_RATE_NAMES])
            rows["root_inventory"].append(root_inventory)
            rows["next_inventory"].append(inventory(env.plant))
            if (index + 1) % 32 == 0:
                _atomic_json(receipt, dict(status="running", episode=episode, step=index + 1,
                                          updated_unix=time.time()))
        rows = {key: np.asarray(value) for key, value in rows.items()}
        count = episode["steps"]
        rows.update(episode_id=np.full(count, episode["episode_id"], dtype=np.int32),
                    step=np.arange(count, dtype=np.int32),
                    terminated=np.arange(count) == count - 1,
                    quality=np.ones(count, dtype=np.float32))
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **rows)
        os.replace(temporary, path)
        result = dict(status="complete", episode=episode, steps=count, summary=env.summary(),
                      elapsed_seconds=time.time() - started, file=str(path))
        _atomic_json(receipt, result)
        return result
    except Exception as exc:
        _atomic_json(receipt, dict(status="failed", episode=episode,
                                  error=f"{type(exc).__name__}: {exc}", updated_unix=time.time()))
        raise

def collect_paper_dataset(output, workers=12, seed=17):
    """Collect once, then assemble exact 80k/10k/10k transition files."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Paper data output must be new; no automatic restart")
    output.mkdir(parents=True, exist_ok=True)
    manifest = paper_collection_manifest(seed)
    manifest["accounting"] = "Minute-integrated EQ, AE, PE, ME and waste; telescoping sludge inventory; interval-average cost"
    manifest["observation_names"] = list(PAPER_OBSERVATION_NAMES)
    manifest["raw_rate_names"] = list(PAPER_RATE_NAMES)
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "progress.json", dict(status="running", completed=0, total=len(manifest["episodes"])))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(collect_paper_episode, str(output), episode) for episode in manifest["episodes"]]
            for future in as_completed(futures):
                results.append(future.result())
                _atomic_json(output / "progress.json", dict(status="running", completed=len(results),
                                                           total=len(futures), updated_unix=time.time()))
        arrays = {}
        for episode in manifest["episodes"]:
            path = output / "episodes" / f"episode_{episode['episode_id']:03d}.npz"
            with np.load(path, allow_pickle=False) as source:
                for key in source.files:
                    arrays.setdefault(key, []).append(source[key])
        arrays = {key: np.concatenate(value) for key, value in arrays.items()}
        for split, (start, stop) in manifest["split_ranges"].items():
            data = {key: value[start:stop].copy() for key, value in arrays.items()}
            data["terminated"][-1] = True
            np.savez_compressed(output / f"{split}.npz", **data)
        _atomic_json(output / "episode_metadata.json", sorted(results, key=lambda row: row["episode"]["episode_id"]))
        _atomic_json(output / "progress.json", dict(status="complete", completed=len(results),
                                                   total=len(results), rows=100000, updated_unix=time.time()))
    except Exception as exc:
        _atomic_json(output / "progress.json", dict(status="failed", completed=len(results),
                                                   error=f"{type(exc).__name__}: {exc}", updated_unix=time.time()))
        raise
    return manifest

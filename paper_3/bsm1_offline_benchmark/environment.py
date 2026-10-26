"""Shared two-input BSM1 economic-control task, without added plant coupling.

The physical backend is the pinned ASM1/settler implementation. The 41-output
selection below is an explicit cost-sufficient adaptation, not a recovered Han
sensor ordering. Current influent (13 ASM1 concentrations and flow) makes the
common deployed input 57-dimensional after adding the two standard feedback
instruments. Full 145-state labels plus current influent are privileged.
"""

from __future__ import annotations

import copy

import pickle

from pathlib import Path

import numpy as np

from bsm1_repro.evaluation import Evaluator

from bsm1_repro.plant import StreamingPlant

from bsm1_repro.rewards import inventory, rates

from simulator.canonical import plant_state, restore_plant

from simulator.inputs import ASM1_NAMES, constant_input, load_official

ROOT = Path(__file__).resolve().parents[1]

ACTION_SCALE = np.array([92230.0, 240.0])

PARTICULATE_INDICES = (2, 3, 4, 5, 6)

EFFLUENT_INDICES = (0, 1, 8, 9, 10)

SENSOR_NAMES = (
    tuple(f"reactor{i}_{ASM1_NAMES[j]}" for i in range(1, 6) for j in PARTICULATE_INDICES)
    + tuple(f"settler_layer{i}_TSS" for i in range(1, 11))
    + tuple(f"effluent_{ASM1_NAMES[j]}" for j in EFFLUENT_INDICES)
    + ("reactor5_XND",)
)

DISTURBANCE_NAMES = tuple("inlet_" + name for name in ASM1_NAMES) + ("inlet_Q",)

OBSERVATION_NAMES = SENSOR_NAMES + ("reactor5_SO", "reactor2_SNO") + DISTURBANCE_NAMES

STATE_NAMES = (
    tuple(f"reactor{i}_{name}" for i in range(1, 6) for name in ASM1_NAMES)
    + tuple(f"settler_layer{i}_{name}" for name in ("SI", "SS", "SO", "SNO", "SNH", "SND", "SALK", "TSS")
            for i in range(1, 11))
)

COMPONENT_NAMES = ("aeration", "pumping", "mixing", "effluent_quality", "sludge_waste_x5", "sludge_inventory_change_x5")

SCHEMA = "canonical_bsm1_two_input_public57_privileged159_v1"

def state145(plant):
    """13 ASM1 entries per reactor and the first eight settler species/layers."""
    return np.concatenate([*[getattr(plant, f"reactor{i}").y0[:13] for i in range(1, 6)],
                           plant.settler.ys0[:80]]).astype(np.float64)

def sensor41(plant):
    return np.concatenate([
        *[getattr(plant, f"y_out{i}")[list(PARTICULATE_INDICES)] for i in range(1, 6)],
        plant.settler.ys0[70:80], plant.ys_eff[list(EFFLUENT_INDICES)], [plant.y_out5[11]],
    ]).astype(np.float64)

def physical_action(action):
    a = np.asarray(action, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("Expected two finite normalized commands [Qa/92230, KLa5/240]")
    return np.clip(a, 0, 1) * ACTION_SCALE

def economic_cost(components):
    c = np.asarray(components)
    return c[..., 3] + .3 * (c[..., 0] + c[..., 1] + c[..., 2] + c[..., 4] + c[..., 5])

class CanonicalBSM1Env:
    """Fifteen-minute held commands, one-minute physics, true current sensors.

    ``reset``/``step`` follow the Gym five-return convention, but do not require
    Gym. ``last_transition`` and snapshots are privileged logger interfaces.
    The baseline controller receives only observation57 and the previous executed action.
    Direct commanded KLa is used, as in the economic-control benchmark; there is
    no extra actuator lag, shared aeration budget, buffer, fault or inhibitor.
    """

    observation_dim = 57
    action_dim = 2
    state_dim = 145
    privileged_dim = 159

    def __init__(self, weather="dry", *, forcing=None, initial_state=None,
                 episode_days=14.0, decision_minutes=15, sensor_noise=True, seed=17,
                 return_sludge=18446.0, waste_sludge=385.0):
        if int(decision_minutes) != decision_minutes or decision_minutes < 1:
            raise ValueError("Decision interval must contain whole minutes")
        if episode_days <= 0 or not np.isclose(episode_days * 1440 / decision_minutes,
                                               round(episode_days * 1440 / decision_minutes)):
            raise ValueError("Episode must contain whole supervisory decisions")
        self.decision_minutes = int(decision_minutes)
        self.episode_days = float(episode_days)
        self.episode_steps = round(episode_days * 1440 / decision_minutes)
        self.weather = str(weather)
        self.sensor_noise = bool(sensor_noise)
        self.seed = int(seed)
        self.return_sludge = float(return_sludge)
        self.waste_sludge = float(waste_sludge)
        if not np.isfinite([self.return_sludge, self.waste_sludge]).all() or min(self.return_sludge, self.waste_sludge) <= 0:
            raise ValueError("Sludge return and waste flows must be finite and positive")
        if forcing is None:
            if weather == "constant":
                forcing = constant_input(max(episode_days, 1.0) + 1 / 1440)
            else:
                names = {"dry": "dry", "rain": "rain", "storm": "strm"}
                if weather not in names:
                    raise ValueError("Unknown official weather")
                forcing = load_official(input_path(f"forcing/Inf_{names[weather]}_2006.txt"))
        self.forcing = np.asarray(forcing, dtype=np.float64).copy()
        if (self.forcing.ndim != 2 or self.forcing.shape[1] != 22 or len(self.forcing) < 2
                or not np.isfinite(self.forcing).all() or self.forcing[0, 0] != 0
                or np.any(np.diff(self.forcing[:, 0]) <= 0)
                or self.forcing[-1, 0] < episode_days - 1e-10):
            raise ValueError("Expected finite 22-column forcing covering the episode, starting at zero")
        if np.any(self.forcing[:, 15] <= self.waste_sludge):
            raise ValueError("Influent flow must exceed the fixed waste flow")
        if initial_state is None:
            raise ValueError("Supply an initial state; the dissertation uses PaperPublic57Env with published_initial_state()")
        if "state" in initial_state and "plant" not in initial_state and "reactors" not in initial_state:
            initial_state = initial_state["state"]
        self.initial_state = copy.deepcopy(initial_state)
        self.reset()

    def _inlet(self, day):
        index = max(0, int(np.searchsorted(self.forcing[:, 0], day + 1e-12, side="right") - 1))
        return self.forcing[min(index, len(self.forcing) - 1), 1:].copy()

    def clean_observation(self):
        inlet = self._inlet(self.minute / 1440)
        return np.concatenate((sensor41(self.plant), [self.plant.y_out5[7], self.plant.y_out2[8]],
                               inlet[:13], [inlet[14]]))

    def observation(self):
        clean = self.clean_observation()
        if not self.sensor_noise:
            return clean
        # Fixed instrument law, identical for every controller and every split.
        # Re-reading a held observation never draws fresh independent evidence.
        sd = .01 * np.maximum(np.abs(clean), 1.0)
        sd[41], sd[42] = .05, .1
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.minute, 20261014]))
        return np.maximum(0, clean + rng.normal(size=57) * sd)

    def privileged(self):
        inlet = self._inlet(self.minute / 1440)
        return np.concatenate((state145(self.plant), inlet[:13], [inlet[14]]))

    def feedback_sensors(self):
        """Separate canonical PI instrument pair, not an extra learned input."""
        return np.array([self.plant.y_out2[8], self.plant.y_out5[7]], dtype=float)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = int(seed)
        self.plant = StreamingPlant()
        initial = self.initial_state.get("plant", self.initial_state)
        restore_plant(self.plant, initial)
        self.plant.ys_was = self.plant.ys_out.copy()
        self.plant.ys_was[14] = self.waste_sludge
        self.plant.settler.q_r = self.return_sludge
        self.plant.settler.q_w = self.waste_sludge
        self.minute = 0
        self.decisions = 0
        self.done = False
        self.previous_action = np.array([55338 / 92230, 84 / 240], dtype=float)
        self.evaluator = Evaluator(inventory(self.plant))
        self.last_transition = None
        return self.observation(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended")
        requested = np.asarray(action, dtype=float)
        command = physical_action(requested)
        bounded = command / ACTION_SCALE
        previous_observation, previous_state = self.observation(), state145(self.plant)
        previous_privileged = self.privileged()
        previous_clean = self.clean_observation()
        initial_inventory = inventory(self.plant)
        accum = dict.fromkeys(("AE_kWh_per_day", "PE_kWh_per_day", "ME_kWh_per_day",
                               "EQI_kgPU_per_day", "waste_kgSS_per_day"), 0.0)
        dt = 1 / 1440
        for _ in range(self.decision_minutes):
            day = self.minute / 1440
            self.plant.advance(self._inlet(day), dt, day, [0, 0, 240, 240, command[1]],
                               command[0], self.return_sludge, self.waste_sludge)
            current_rates = rates(self.plant)
            current_inventory = inventory(self.plant)
            self.evaluator.add(self.plant.ys_eff, current_rates, dt, current_inventory)
            for name in accum:
                accum[name] += current_rates[name] * dt
            self.minute += 1
        components = np.array([accum["AE_kWh_per_day"], accum["PE_kWh_per_day"],
                               accum["ME_kWh_per_day"], accum["EQI_kgPU_per_day"],
                               5 * accum["waste_kgSS_per_day"],
                               5 * (current_inventory - initial_inventory)])
        integrated_cost = float(economic_cost(components))
        cost = integrated_cost / (self.decision_minutes / 1440)
        self.decisions += 1
        self.done = self.decisions >= self.episode_steps
        next_observation, next_state = self.observation(), state145(self.plant)
        if not np.isfinite(np.concatenate((next_observation, next_state, components, [cost]))).all():
            raise FloatingPointError("Nonfinite canonical benchmark output")
        self.last_transition = dict(
            observation=previous_observation, next_observation=next_observation.copy(),
            clean_observation=previous_clean, next_clean_observation=self.clean_observation(),
            privileged=previous_privileged, next_privileged=self.privileged(),
            state=previous_state, next_state=next_state, action=bounded.copy(),
            requested_action=requested.copy(), physical_action=command.copy(),
            components=components, economic_cost=cost, cost=cost,
            integrated_economic_cost=integrated_cost, reward=-cost,
            previous_action=self.previous_action.copy(),
            time_start_days=(self.minute - self.decision_minutes) / 1440,
            time_end_days=self.minute / 1440,
            terminal=self.done,
        )
        self.previous_action = bounded.copy()
        return next_observation, -cost, False, self.done, {}

    def snapshot(self):
        return copy.deepcopy(dict(schema=SCHEMA, plant=plant_state(self.plant),
                                  ys_was=self.plant.ys_was, minute=self.minute,
                                  decisions=self.decisions, done=self.done,
                                  evaluator=self.evaluator, previous_action=self.previous_action,
                                  seed=self.seed, sensor_noise=self.sensor_noise,
                                  return_sludge=self.return_sludge, waste_sludge=self.waste_sludge))

    def restore(self, snapshot):
        value = copy.deepcopy(snapshot)
        if value["schema"] != SCHEMA:
            raise ValueError("Snapshot schema mismatch")
        restore_plant(self.plant, value["plant"])
        self.plant.ys_was = value["ys_was"]
        for name in ("minute", "decisions", "done", "evaluator", "previous_action", "seed", "sensor_noise"):
            setattr(self, name, value[name])
        for name in ("return_sludge", "waste_sludge"):
            if name in value:
                setattr(self, name, value[name])
        self.last_transition = None
        return self.observation()

    def summary(self):
        result = self.evaluator.summary()
        if self.evaluator.days > 0:
            result["economic_cost_per_day"] = result["EQI"] + .3 * result["OCI"]
            result["integrated_economic_cost"] = result["economic_cost_per_day"] * self.evaluator.days
        result["schema"] = SCHEMA
        return result

from work3_paths import input_path

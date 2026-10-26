"""Passive observation-prediction mismatch monitor for the existing AIF loop.

Pure NumPy monitoring; no model/simulator import occurs at module import. The
factory imports the unchanged recommended controller only when explicitly called.
An alarm suggests investigating mismatch. It diagnoses neither a new dynamic nor
model incapacity, changes no action, and never starts adaptation or data collection.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

THRESHOLD = 0.2490178992357052
PERSISTENCE = 4
ACTION_ATOL = 1e-7
SCHEMA = "passive_observation_mismatch_monitor_v1"
WRAPPER_SCHEMA = "passive_aif_observation_mismatch_wrapper_v1"
DETECTOR_PATH = Path(__file__).resolve().parents[1] / "configs/detector.json"

PUBLIC41_INDICES = np.array([13*r+j for r in range(5) for j in range(2,7)]
    + list(range(135,145)) + [65,75,95,105,115,63], dtype=np.int64)


def load_calibration(path=DETECTOR_PATH):
    """Read the existing frozen calibration; never fit or rewrite it."""
    path = Path(path).resolve()
    blob = path.read_bytes(); record = json.loads(blob)
    if (record.get("status") != "frozen" or record.get("threshold") != THRESHOLD
            or record.get("consecutive_steps") != PERSISTENCE or record.get("dev_residuals") != 4032
            or record.get("quantile") != .95 or record.get("method") != "linear"
            or record.get("performance_labels_used") is not False
            or record.get("calibration_cases") != [f"validation_p00{i}" for i in range(3)]):
        raise ValueError("The existing fixed DEV3 95th-percentile calibration is required")
    return dict(record, source_path=str(path), source_sha256=hashlib.sha256(blob).hexdigest())


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != tuple(shape) or not np.isfinite(result).all():
        raise ValueError(f"{name}: finite shape {tuple(shape)} required")
    return result


def _index(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < 0:
        raise ValueError(f"{name}: nonnegative integer required")
    return int(value)


def _numpy(value):
    """Copy an existing tensor/array without a model call or random draw."""
    if hasattr(value,"detach"):
        value = value.detach().cpu().numpy()
    return np.array(value,copy=True)


class ObservationMismatchMonitor:
    def __init__(self, train_public41_scale, *, calibration=None):
        self._scale = _array(train_public41_scale,(41,),"TRAIN public41 scale")
        if np.any(self._scale <= 0):
            raise ValueError("Strictly positive frozen TRAIN scales required")
        self._scale.setflags(write=False)
        self.calibration = copy.deepcopy(load_calibration() if calibration is None else calibration)
        if (self.calibration.get("threshold") != THRESHOLD
                or self.calibration.get("consecutive_steps") != PERSISTENCE
                or self.calibration.get("performance_labels_used") is not False
                or not isinstance(self.calibration.get("source_sha256"),str)
                or len(self.calibration["source_sha256"]) != 64):
            raise ValueError("Identified frozen threshold and four-observation persistence required")
        self._identity = dict(schema=SCHEMA,threshold=THRESHOLD,consecutive_steps=PERSISTENCE,
            detector_sha256=self.calibration["source_sha256"],
            train_public41_scale_sha256=hashlib.sha256(self._scale.astype("<f8").tobytes()).hexdigest(),
            public41_state_indices=PUBLIC41_INDICES.tolist(),
            residual_definition="sqrt(mean(((observed_next_public41-prior_weighted_prediction)/TRAIN_scale)^2))",
            comparison="strictly greater than threshold")
        self.completed_observations = 0
        self.consecutive_exceedances = 0
        self.persistent_prediction_mismatch = False
        self._pending = None
        self.last_result = None

    @property
    def pending_prediction(self):
        return copy.deepcopy(self._pending)

    @property
    def calibration_identity(self):
        return copy.deepcopy(self._identity)

    def record_prediction(self, next_public41_particles, prior_weights, selected_action, *, step):
        """Record the selected executed-action forecast before seeing its outcome."""
        step = _index(step,"step")
        if self._pending is not None:
            raise RuntimeError("Observe the pending prediction exactly once before recording another")
        if step != self.completed_observations:
            raise ValueError("Prediction step is a gap, duplicate or out of order")
        samples = np.array(next_public41_particles,dtype=np.float64,copy=True)
        if samples.ndim != 2 or samples.shape[1] != 41 or samples.shape[0] < 1 or not np.isfinite(samples).all():
            raise ValueError("Finite selected-action K-by-41 predictions required")
        weights = _array(prior_weights,(len(samples),),"prior weights")
        if np.any(weights < 0) or not np.isclose(weights.sum(),1.,atol=1e-10,rtol=0):
            raise ValueError("Normalized nonnegative prior weights required")
        action = _array(selected_action,(2,),"selected action")
        if np.any(action < 0) or np.any(action > 1):
            raise ValueError("Selected action must lie in the normalized actuator box")
        mean = np.einsum("k,kd->d",weights,samples)
        sd = np.sqrt(np.maximum(np.einsum("k,kd->d",weights,(samples-mean)**2),0.))
        if not np.isfinite(mean).all() or not np.isfinite(sd).all():
            raise ValueError("Prediction moments overflow; no valid forecast recorded")
        # Assignment occurs only after every validation, so invalid inputs do not
        # consume the pending slot or modify an existing alarm.
        self._pending = dict(step=step,public41_prediction_mean=mean.tolist(),
            prediction_particle_sd=sd.tolist(),selected_action=action.tolist(),particles=len(samples))
        return copy.deepcopy(self._pending)

    def preview_observation(self, actual_public41, executed_action, *, step):
        """Validate and compute a result without committing monitor state."""
        step = _index(step,"step")
        if self._pending is None:
            raise RuntimeError("An observation requires its prior recorded prediction")
        if step != self.completed_observations or step != self._pending["step"]:
            raise ValueError("Observation step is a gap, duplicate or out of order")
        observed = _array(actual_public41,(41,),"actual public41")
        action = _array(executed_action,(2,),"executed action")
        if np.any(action < 0) or np.any(action > 1):
            raise ValueError("Executed action must lie in the normalized actuator box")
        if not np.allclose(action,self._pending["selected_action"],atol=ACTION_ATOL,rtol=0):
            raise ValueError("Executed action differs from the forecast action; no matching prediction is available")
        residual = observed - np.asarray(self._pending["public41_prediction_mean"])
        normalized = residual/self._scale
        contributions = normalized**2/41.
        score = float(np.sqrt(np.mean(normalized**2)))
        if not np.isfinite(score) or not np.isfinite(contributions).all():
            raise ValueError("Residual score overflow; observation is invalid")
        exceeds = score > THRESHOLD
        consecutive = self.consecutive_exceedances+1 if exceeds else 0
        alarm = consecutive >= PERSISTENCE
        return dict(step=step,available_after_boundary=step+1,
            public41_residual_rmse=score,threshold=THRESHOLD,exceeds_threshold=bool(exceeds),
            consecutive_exceedances=consecutive,persistent_prediction_mismatch=bool(alarm),
            model_mismatch_suspected=bool(alarm),
            alarm_onset=bool(alarm and not self.persistent_prediction_mismatch),
            alarm_cleared=bool(self.persistent_prediction_mismatch and not alarm),
            public41_prediction_mean=copy.deepcopy(self._pending["public41_prediction_mean"]),
            observed_public41=observed.tolist(),public41_residual=residual.tolist(),
            public41_train_scaled_residual=normalized.tolist(),channel_squared_contributions=contributions.tolist(),
            prediction_particle_sd=copy.deepcopy(self._pending["prediction_particle_sd"]),
            executed_action=action.tolist(),
            interpretation="Persistent prediction mismatch suspected; cause and new dynamics are not diagnosed",
            timing="Available only after this transition is observed; it did not influence this transition's action",
            automatic_finetuning=False,automatic_action_change=False)

    def _commit(self, result):
        self.completed_observations += 1
        self.consecutive_exceedances = result["consecutive_exceedances"]
        self.persistent_prediction_mismatch = result["persistent_prediction_mismatch"]
        self.last_result = copy.deepcopy(result)
        self._pending = None
        return copy.deepcopy(result)

    def observe(self, actual_public41, executed_action, *, step):
        return self._commit(self.preview_observation(actual_public41,executed_action,step=step))

    def runtime_state(self):
        return dict(schema=SCHEMA,calibration_identity=self.calibration_identity,
            completed_observations=self.completed_observations,
            consecutive_exceedances=self.consecutive_exceedances,
            persistent_prediction_mismatch=self.persistent_prediction_mismatch,
            pending_prediction=self.pending_prediction,last_result=copy.deepcopy(self.last_result))

    def restore_runtime_state(self, state):
        """Validate the complete calibration/phase contract before changing state."""
        state = copy.deepcopy(state)
        if state.get("schema") != SCHEMA or state.get("calibration_identity") != self._identity:
            raise ValueError("Snapshot calibration, channel mapping or TRAIN normalization differs")
        completed = _index(state["completed_observations"],"completed observations")
        consecutive = _index(state["consecutive_exceedances"],"consecutive exceedances")
        alarm = state["persistent_prediction_mismatch"]
        if not isinstance(alarm,bool) or consecutive > completed or alarm != (consecutive >= PERSISTENCE):
            raise ValueError("Inconsistent snapshot persistence state")
        pending = state["pending_prediction"]
        if pending is not None:
            if _index(pending["step"],"pending step") != completed or _index(pending["particles"],"particles") < 1:
                raise ValueError("Snapshot pending prediction is out of sequence")
            _array(pending["public41_prediction_mean"],(41,),"pending mean")
            sd = _array(pending["prediction_particle_sd"],(41,),"pending SD")
            action = _array(pending["selected_action"],(2,),"pending action")
            if np.any(sd < 0) or np.any(action < 0) or np.any(action > 1):
                raise ValueError("Invalid pending forecast/action")
        last = state["last_result"]
        if (completed == 0) != (last is None):
            raise ValueError("Snapshot last observation does not match its completed count")
        if last is not None and (last.get("available_after_boundary") != completed
                or last.get("step") != completed-1 or last.get("consecutive_exceedances") != consecutive
                or last.get("persistent_prediction_mismatch") != alarm):
            raise ValueError("Snapshot last observation and persistence disagree")
        if last is not None:
            score=_array(last["public41_residual_rmse"],(),"last error score")
            if score < 0 or last.get("threshold") != THRESHOLD or last.get("exceeds_threshold") != (score > THRESHOLD):
                raise ValueError("Snapshot score and threshold state disagree")
            if bool(last["exceeds_threshold"]) != (consecutive > 0):
                raise ValueError("Snapshot score and consecutive count disagree")
            for key in ("public41_prediction_mean","observed_public41","public41_residual",
                        "public41_train_scaled_residual","channel_squared_contributions","prediction_particle_sd"):
                _array(last[key],(41,),"last "+key)
        self.completed_observations=completed;self.consecutive_exceedances=consecutive
        self.persistent_prediction_mismatch=alarm;self._pending=pending;self.last_result=last


class PassiveMismatchController:
    """Decorator for an existing selective-mean or compatible preference wrapper."""
    def __init__(self,base_controller,*,calibration=None):
        self.base_controller=base_controller
        native=base_controller.controller
        indices=_numpy(native.public_indices).astype(np.int64)
        if indices.shape != (55,) or not np.array_equal(indices[:41],PUBLIC41_INDICES):
            raise ValueError("The native public55/plant41 projection is required")
        scale=_numpy(native.model.state_scale)
        self.monitor=ObservationMismatchMonitor(scale[indices[:41]],calibration=calibration)
        if native.transition_count != 0 or native.pending is not None:
            raise ValueError("Wrap a fresh controller; restore a joint snapshot to resume")
        self.last_diagnostics={};self.last_monitor_result=None;self._failed=False

    def __getattr__(self,name):
        return getattr(self.base_controller,name)

    def _ready(self):
        if self._failed:
            raise RuntimeError("Wrapper is in a failed phase; preserve it or explicitly restore a valid joint snapshot")

    def act(self,public55,previous_action=None):
        self._ready();native=self.base_controller.controller
        if self.monitor.pending_prediction is not None or native.pending is not None:
            raise RuntimeError("Observe the pending action before calling act again")
        if native.transition_count != self.monitor.completed_observations:
            raise RuntimeError("Native/monitor transition sequence differs")
        step=int(native.transition_count)
        try:
            action=self.base_controller.act(public55,previous_action)
            pending=native.pending
            if pending is None:
                raise RuntimeError("Base controller did not retain its selected-action prediction")
            private=_numpy(pending["next_particles"])
            weights=_numpy(pending["weights"])
            if private.shape != (8,159) or weights.shape != (8,):
                raise ValueError("The native eight-particle selected next-state prediction is required")
            if not np.allclose(action,_numpy(pending["action"]),atol=ACTION_ATOL,rtol=0):
                raise ValueError("Returned action and native pending action differ")
            self.monitor.record_prediction(private[:,PUBLIC41_INDICES],weights,action,step=step)
            self.last_diagnostics=copy.deepcopy(self.base_controller.last_diagnostics)
            self.last_diagnostics["observation_mismatch_monitor"]=dict(
                phase="awaiting_observation",pending_step=step,
                persistent_prediction_mismatch=self.monitor.persistent_prediction_mismatch,
                last_observed_boundary=self.monitor.completed_observations)
            return action
        except BaseException:
            self._failed=True
            raise

    def observe(self,next_public55,executed_action,raw_cost):
        self._ready();native=self.base_controller.controller
        public=_array(next_public55,(55,),"next public55")
        # The monitor sees plant41 only: neither raw_cost nor hidden truth enters it.
        result=self.monitor.preview_observation(public[:41],executed_action,step=int(native.transition_count))
        if not np.isfinite(raw_cost):
            raise ValueError("Base controller requires a finite raw cost")
        try:
            native_result=self.base_controller.observe(next_public55,executed_action,raw_cost)
            if native.transition_count != self.monitor.completed_observations+1 or native.pending is not None:
                raise RuntimeError("Native observation did not complete exactly one pending transition")
            self.last_monitor_result=self.monitor._commit(result)
            self.last_diagnostics=copy.deepcopy(self.base_controller.last_diagnostics)
            self.last_diagnostics["observation_mismatch_monitor"]=copy.deepcopy(self.last_monitor_result)
            return native_result
        except BaseException:
            self._failed=True
            raise

    def runtime_state(self):
        return dict(schema=WRAPPER_SCHEMA,base_controller=copy.deepcopy(self.base_controller.runtime_state()),
            monitor=self.monitor.runtime_state(),failed_phase=self._failed,
            last_diagnostics=copy.deepcopy(self.last_diagnostics))

    def restore_runtime_state(self,state):
        state=copy.deepcopy(state)
        if state.get("schema") != WRAPPER_SCHEMA or not isinstance(state.get("failed_phase"),bool):
            raise ValueError("Compatible joint wrapper snapshot required")
        staged=ObservationMismatchMonitor(self.monitor._scale,calibration=self.monitor.calibration)
        staged.restore_runtime_state(state["monitor"])
        # Reject calibration drift before touching the native state or its RNG.
        try:
            self.base_controller.restore_runtime_state(state["base_controller"])
            native=self.base_controller.controller
            if native.transition_count != staged.completed_observations or ((native.pending is None) != (staged.pending_prediction is None)):
                raise ValueError("Restored native/monitor phases disagree")
            if native.pending is not None and not np.allclose(_numpy(native.pending["action"]),
                    staged.pending_prediction["selected_action"],atol=ACTION_ATOL,rtol=0):
                raise ValueError("Restored pending actions differ")
            self.monitor=staged;self.last_monitor_result=copy.deepcopy(staged.last_result)
            self.last_diagnostics=copy.deepcopy(state["last_diagnostics"]);self._failed=state["failed_phase"]
        except BaseException:
            self._failed=True
            raise


def make_controller(device="cuda",*,calibration_path=DETECTOR_PATH):
    """Explicitly construct and wrap the unchanged recommended controller."""
    from research_shared_model import recommended
    return PassiveMismatchController(recommended.make_controller(device=device),
                                     calibration=load_calibration(calibration_path))

"""Native revised-pair execution and exact trace validators, no six-arm orchestration."""
import time
import numpy as np
import torch
from work2.native.control import PublicHistory
from work2.native.environment import PaperEnv
from work2.native.evaluation import _synchronize
from work2.native.policy_preservation_control import policy_preservation_actions as adaptive_actions
from work2.native.reference_policy_control import ReferenceSelectionConfig,selection_weights
HIDDEN_METHODS=('revised_vfe','revised_efe')
def require(value,message):
    if not value: raise ValueError(message)

NATIVE_KEYS = (
    "actions",
    "rewards",
    "speed",
    "inverted",
    "conditions",
    "terminated",
    "truncated",
    "decision_s",
    "update_s",
    "update_complete",
    "simulator_s",
    "batch_size",
)


def validate_native(case, arrays, spec, method):
    n = len(arrays["rewards"])
    require(0 < n <= case["steps"], "native step count")
    for key in NATIVE_KEYS:
        value = arrays[key]
        require(
            value.shape == ((n, 6) if key == "actions" else (n,)) and np.isfinite(value).all(),
            f"finite native {key}",
        )
    require(
        arrays["observations"].shape == (n + 1, 17)
        and np.isfinite(arrays["observations"]).all()
        and (np.abs(arrays["actions"]) <= 1).all(),
        "public observations/bounded actions",
    )
    done = arrays["terminated"].astype(bool) | arrays["truncated"].astype(bool)
    require(
        not done[:-1].any()
        and (n == case["steps"] or arrays["terminated"][-1])
        and arrays["update_complete"].all(),
        "native ending and every actual update",
    )
    labels = np.full(n, case["initial_condition"])
    for event in case["events"]:
        labels[event["step"] :] = event["condition"]
    require(np.array_equal(arrays["conditions"], labels), "native transition schedule")
    require(method in HIDDEN_METHODS, "Only the two included revised methods are retained")
    if method in HIDDEN_METHODS:
        q, prior, ll = (arrays[k] for k in ("beliefs", "priors", "log_likelihood"))
        require(
            q.shape == (n + 1, 8) and prior.shape == ll.shape == (n, 8),
            "every actual transition needs a VFE update",
        )
        for values in (q, prior):
            require(
                np.isfinite(values).all()
                and (values >= 0).all()
                and np.allclose(values.sum(-1), 1, atol=2e-6, rtol=0),
                "categorical beliefs",
            )
        require(np.array_equal(q[0], np.full(8, 0.125)), "uniform initial physical belief")
        h = spec["config"]["hazard"]
        expected_prior = (1 - h) * q[:-1] + h / 7 * (q[:-1].sum(-1, keepdims=True) - q[:-1])
        require(np.allclose(prior, expected_prior, atol=2e-6, rtol=0), "one hazard step per action")
        logits = np.log(prior.astype(np.float64)) + ll
        posterior = np.exp(logits - logits.max(-1, keepdims=True))
        posterior /= posterior.sum(-1, keepdims=True)
        require(
            np.isfinite(ll).all() and np.allclose(q[1:], posterior, atol=2e-5, rtol=0),
            "normalized observed likelihood updates actual executed action",
        )
        k = 24
        for key in ("costs", "information", "information_se"):
            require(
                arrays[key].shape == (n, k) and np.isfinite(arrays[key]).all(),
                f"finite full candidate {key}",
            )
        require((arrays["information_se"] >= 0).all(), "nonnegative MI standard error")
        selected = arrays["selected"].astype(int)
        require(
            np.array_equal(selected, arrays["selected"])
            and ((0 <= selected) & (selected < 24)).all(),
            "eligible generated index",
        )
        weight = spec["config"]["information_weight"]
        objective = arrays["costs"] - weight * arrays["information"]
        if method == "revised_vfe":
            require(not arrays["information"].any(), "lambda0 skips information queries")
        selected_spec = spec["selection"]
        selection = ReferenceSelectionConfig(
            "soft", selected_spec["cost_scale"], selected_spec["temperature_multiplier"]
        )
        weights = np.stack(
            [selection_weights(torch.as_tensor(v), selection).numpy() for v in objective]
        )
        require(
            arrays["candidate_actions"].shape == (n, 24, 8, 6)
            and np.array_equal(
                arrays["actions"], arrays["candidate_actions"][np.arange(n), selected, 0]
            ),
            "actual action is selected generated trajectory prefix",
        )
        require(
            np.allclose(arrays["objective"], objective, atol=2e-6, rtol=1e-6)
            and np.allclose(arrays["weights"], weights, atol=2e-7, rtol=2e-6)
            and (weights[np.arange(n), selected] > 0).all(),
            "finite soft reference update",
        )
        require(
            np.allclose(arrays["ess"], 1 / np.square(weights).sum(-1), atol=2e-5, rtol=2e-6),
            "effective candidate count",
        )
    return True


def run_episodes(policies, cases, spec, method, *, completed_callback=None, progress_callback=None):
    """One actual prefix action and public observation update on every native step."""
    require(
        cases and len(cases) == len(policies) and len({c["case_id"] for c in cases}) == len(cases),
        "unique nonempty policy/case batch",
    )
    require(all(type(p) is type(policies[0]) for p in policies), "one policy family per batch")
    environments, histories, traces, done, results = [], [], [], set(), {}
    device = policies[0].device
    try:
        for policy, case in zip(policies, cases, strict=True):
            env = PaperEnv(
                "HalfCheetah-v5", case["initial_condition"], max_episode_steps=case["steps"]
            )
            environments.append(env)
            obs, _ = env.reset(seed=case["reset_seed"])
            histories.append(PublicHistory(obs, policy.history_length))
            trace = {key: [] for key in NATIVE_KEYS}
            trace["observations"] = [obs.copy()]
            if policy.controller is not None:
                trace["beliefs"] = [policy.controller.belief.cpu().numpy().copy()]
                trace["priors"], trace["log_likelihood"] = [], []
            traces.append(trace)
        schedules = [{e["step"]: e["condition"] for e in c["events"]} for c in cases]
        for step in range(max(c["steps"] for c in cases)):
            active = [i for i in range(len(cases)) if i not in done]
            for i in active:
                if step in schedules[i]:
                    environments[i].apply_condition(schedules[i][step])
            _synchronize(device)
            started = time.perf_counter()
            decisions = adaptive_actions(
                [policies[i] for i in active],
                [histories[i].tensors(device) for i in active],
                [step] * len(active),
                [
                    environments[i].condition.index
                    if getattr(policies[i], "receives_truth", False)
                    else None
                    for i in active
                ],
            )
            _synchronize(device)
            decision_s, stepped = time.perf_counter() - started, []
            for i, (action, details) in zip(active, decisions, strict=True):
                started = time.perf_counter()
                obs, reward, terminated, truncated, info = environments[i].step(action)
                values = dict(
                    actions=np.asarray(action).copy(),
                    rewards=reward,
                    speed=float(info.get("x_velocity", 0)),
                    inverted=environments[i].diagnostics()["inverted"],
                    conditions=environments[i].condition.index,
                    terminated=terminated,
                    truncated=truncated,
                    decision_s=decision_s,
                    update_s=0.0,
                    update_complete=False,
                    simulator_s=time.perf_counter() - started,
                    batch_size=len(active),
                )
                values.update({k: v for k, v in details.items() if v is not None and k != "prior"})
                traces[i]["observations"].append(obs.copy())
                for key, value in values.items():
                    traces[i].setdefault(key, []).append(value)
                stepped.append((i, action, obs, reward, details))
            started = time.perf_counter()
            try:
                for i, action, obs, reward, details in stepped:
                    histories[i].append(action, obs, reward)
                    inference = policies[i].observe(action, obs, histories[i].tensors(device))
                    if policies[i].controller is not None:
                        traces[i]["beliefs"].append(inference["belief"].copy())
                        traces[i]["log_likelihood"].append(inference["log_likelihood"].copy())
                        traces[i]["priors"].append(details["prior"].copy())
                    traces[i]["update_complete"][-1] = True
                _synchronize(device)
            finally:
                duration = time.perf_counter() - started
                for i, *_ in stepped:
                    traces[i]["update_s"][-1] = duration
            for i in active:
                trace = traces[i]
                if (
                    trace["terminated"][-1]
                    or trace["truncated"][-1]
                    or step + 1 == cases[i]["steps"]
                ):
                    arrays = {k: np.asarray(v) for k, v in trace.items()}
                    validate_native(cases[i], arrays, spec, method)
                    if completed_callback:
                        completed_callback(cases[i], arrays)
                    results[cases[i]["case_id"]] = arrays
                    done.add(i)
            if progress_callback and (step + 1) % 100 == 0:
                progress_callback(step + 1)
            if len(done) == len(cases):
                break
        return results
    except BaseException as error:
        error.paper2_partial_traces = {
            cases[i]["case_id"]: {k: np.asarray(v) for k, v in trace.items()}
            for i, trace in enumerate(traces)
            if i not in done
        }
        raise
    finally:
        for env in environments:
            env.close()

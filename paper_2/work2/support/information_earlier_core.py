"""Native original 120-case information study logging; two included arms only."""
import time
import numpy as np
import torch
from work2.native.control import PublicHistory,act_batch
from work2.native.evaluation import decision_seed,_synchronize
from work2.native.environment import PaperEnv
from work2.native.protocol import episode_metrics

@torch.no_grad()
def policy_actions(policies,histories,steps,oracle_conditions):
    if any(t is not None for t in oracle_conditions): raise ValueError('No evaluator truth in the included pair')
    if len(policies)==1: return [policies[0].act(histories[0],steps[0])]
    for policy,step in zip(policies,steps,strict=True):
        policy.controller.generator.manual_seed(decision_seed(policy.seed,step))
    return act_batch([p.controller for p in policies],histories,oracle_conditions)

def run_episodes(
    policies,
    cases,
    env_id="HalfCheetah-v5",
    reference_speeds=None,
    progress_callback=None,
    completed_callback=None,
):
    """Synchronous physical cases with batched inference; B>1 is throughput, not B=1 latency."""
    environments, histories, traces, done, results = [], [], [], set(), {}
    if len(policies) != len(cases) or not cases:
        raise ValueError("Each nonempty case needs its own policy state")
    device = policies[0].device
    try:
        for policy, case in zip(policies, cases, strict=True):
            if policy.method == "expert" and case["events"]:
                raise ValueError("The expert capability evaluator is stationary")
            env = PaperEnv(env_id, case["initial_condition"], max_episode_steps=case["steps"])
            environments.append(env)
            observation, _ = env.reset(seed=case["reset_seed"])
            histories.append(PublicHistory(observation, policy.history_length))
            trace = {
                key: []
                for key in (
                    "actions",
                    "rewards",
                    "speed",
                    "inverted",
                    "conditions",
                    "decision_s",
                    "update_s",
                    "simulator_s",
                    "terminated",
                    "truncated",
                    "selected",
                    "costs",
                    "information",
                    "batch_size",
                )
            }
            trace["observations"] = [observation.copy()]
            if policy.controller:
                trace.update(
                    beliefs=[policy.controller.belief.cpu().numpy().copy()],
                    log_likelihood=[],
                    priors=[],
                )
            traces.append(trace)
        schedules = [
            {event["step"]: event["condition"] for event in case["events"]} for case in cases
        ]
        for step in range(max(c["steps"] for c in cases)):
            active = [i for i in range(len(cases)) if i not in done]
            for i in active:
                if step in schedules[i]:
                    environments[i].apply_condition(schedules[i][step])
            _synchronize(device)
            started = time.perf_counter()
            public = [histories[i].tensors(device) for i in active]
            decisions = policy_actions(
                [policies[i] for i in active],
                public,
                [step] * len(active),
                [
                    environments[i].condition.index if policies[i].method == "oracle" else None
                    for i in active
                ],
            )
            _synchronize(device)
            decision_duration = time.perf_counter() - started
            transitions = []
            for i, (action, _details) in zip(active, decisions, strict=True):
                sim_started = time.perf_counter()
                transition = environments[i].step(action)
                sim_duration = time.perf_counter() - sim_started
                transitions.append((transition, sim_duration))
            update_started = time.perf_counter()
            inferences = []
            for i, (action, _), ((observation, reward, _, _, _), _) in zip(
                active, decisions, transitions, strict=True
            ):
                histories[i].append(action, observation, reward)
                inferences.append(
                    policies[i].observe(action, observation, histories[i].tensors(device))
                )
            _synchronize(device)
            update_duration = time.perf_counter() - update_started
            for i, (action, details), (transition, sim_duration), inference in zip(
                active, decisions, transitions, inferences, strict=True
            ):
                observation, reward, terminated, truncated, info = transition
                trace = traces[i]
                trace["observations"].append(observation.copy())
                values = {
                    "actions": np.asarray(action).copy(),
                    "rewards": reward,
                    "speed": float(info.get("x_velocity", 0)),
                    "inverted": environments[i].diagnostics()["inverted"],
                    "conditions": environments[i].condition.index,
                    "terminated": terminated,
                    "truncated": truncated,
                    "decision_s": decision_duration,
                    "update_s": update_duration,
                    "simulator_s": sim_duration,
                    "batch_size": len(active),
                }
                for key, value in values.items():
                    trace[key].append(value)
                if inference:
                    trace["beliefs"].append(inference["belief"].copy())
                    trace["log_likelihood"].append(inference["log_likelihood"].copy())
                    trace["priors"].append(details["prior"].copy())
                    for key in ("selected", "costs", "information"):
                        trace[key].append(details[key])
                if terminated or truncated or step + 1 == cases[i]["steps"]:
                    arrays = {key: np.asarray(value) for key, value in trace.items() if len(value)}
                    metrics = episode_metrics(
                        cases[i],
                        arrays["rewards"],
                        arrays["speed"],
                        arrays["inverted"],
                        reference_speeds,
                        arrays.get("beliefs"),
                    )
                    times = arrays["decision_s"] + arrays["update_s"]
                    metrics["latency"] = {
                        "mean_s": float(times.mean()),
                        "median_s": float(np.median(times)),
                        "p95_s": float(np.quantile(times, 0.95)),
                        "deadline_s": environments[i].unwrapped.dt,
                        "deadline_miss_fraction": float(
                            (times > environments[i].unwrapped.dt).mean()
                        ),
                        "amortized_controller_s": float((times / arrays["batch_size"]).mean()),
                        "simulator_mean_s": float(arrays["simulator_s"].mean()),
                        "maximum_batch_size": len(cases),
                        "scope": (
                            "single-environment complete controller, simulator excluded"
                            if len(cases) == 1
                            else "batch response time; amortized throughput is not B=1 latency"
                        ),
                    }
                    results[cases[i]["case_id"]] = (arrays, metrics)
                    done.add(i)
                    if completed_callback:
                        completed_callback(cases[i], arrays, metrics)
            if progress_callback and (step + 1) % 100 == 0:
                progress_callback(step + 1)
            if len(done) == len(cases):
                break
        return results
    except BaseException as error:
        error.paper2_partial_traces = {
            cases[i]["case_id"]: {
                key: np.asarray(value) for key, value in trace.items() if len(value)
            }
            for i, trace in enumerate(traces)
            if i not in done
        }
        raise
    finally:
        for env in environments:
            env.close()

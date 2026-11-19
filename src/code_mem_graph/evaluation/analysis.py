"""Offline, failure-inclusive analysis of a predeclared experiment schedule."""
from __future__ import annotations

from collections import Counter
from math import ceil
from statistics import mean

from code_mem_graph.serialization import fingerprint
from code_mem_graph.tokens import PayloadCounter

from .experiment import Experiment
from .models import Trace
from .report import capture_blockers, score_run
from .accounting import validate_trace_usage


def analyze_experiment(experiment: Experiment, traces: list[Trace]) -> dict:
    if len({trace.run_id for trace in traces}) != len(traces):
        raise ValueError("duplicate trial run ID")
    validate_trace_usage(traces)
    scheduled = {entry["run_id"]: entry for entry in experiment.schedule()}
    observed: dict[str, dict] = {}
    counter = PayloadCounter(experiment.encoding)
    profiles = set()
    for trace in traces:
        if trace.run_id in observed:
            raise ValueError("duplicate trial run ID")
        if trace.run_id not in scheduled:
            raise ValueError("trace is outside the predeclared trial schedule")
        expected = scheduled[trace.run_id]
        if (trace.policy.model_dump(mode="json") != expected["policy"] or trace.task_id != expected["task_id"]
                or trace.replicate != expected["replicate"] or trace.cache_state != expected["cache_state"]):
            raise ValueError("trace does not match its scheduled policy/task/replicate/cache state")
        profile = fingerprint(trace.agent.model_dump(mode="json"))
        if experiment.agent is not None and profile != fingerprint(experiment.agent.model_dump(mode="json")):
            raise ValueError("trace agent differs from the predeclared profile")
        profiles.add(profile)
        observed[trace.run_id] = score_run(experiment.suite, trace, counter)
    if len(profiles) > 1:
        raise ValueError("experiment traces must use the same controlled agent profile")
    runs = list(observed.values())
    if runs and all(run["usage_complete"] for run in runs):
        basis = "reported_model_total_tokens"
    elif runs and all(run["model_total_tokens_estimate"] is not None for run in runs):
        basis = "model_total_tokens_estimate"
    else:
        basis = None
    policies = []
    by_policy: dict[str, dict] = {}
    for policy in experiment.policies():
        planned = [entry for entry in scheduled.values() if entry["policy"] == policy.model_dump(mode="json")]
        present = [observed[entry["run_id"]] for entry in planned if entry["run_id"] in observed]
        reviewed = all(run["status"] not in {"unreviewed","unverified","disputed"} for run in present)
        complete = len(present) == len(planned)
        accepted = sum(run["accepted"] for run in present)
        total = sum(run[basis] for run in present) if basis is not None and present else None
        elapsed = sorted(run["elapsed_ms"] for run in present)
        value = {
            "policy": policy.model_dump(mode="json"), "scheduled": len(planned), "observed": len(present),
            "status_counts": dict(Counter(run["status"] for run in present)), "accepted": accepted,
            "acceptance_rate": accepted / len(present) if present and reviewed else None,
            "complete": complete, "all_reviews_complete": bool(present) and reviewed,
            "token_basis": basis, "observed_tokens_spent": total,
            "tokens_per_accepted_result": total / accepted if total is not None and accepted and complete and reviewed else None,
            "mean_run_tokens": total / len(present) if total is not None and present else None,
            "latency_ms": {"mean": mean(elapsed), "p50": elapsed[ceil(len(elapsed) * .5) - 1],
                           "p95": elapsed[ceil(len(elapsed) * .95) - 1], "quantile_method": "nearest_rank"} if elapsed else None,
        }
        policies.append(value)
        by_policy[policy.name] = value
    baseline = by_policy["native"]
    comparisons = []
    for candidate in policies[1:]:
        name = candidate["policy"]["name"]
        pairs = []
        for replicate in map(str, range(1, experiment.replicates + 1)):
            expected = [entry for entry in scheduled.values() if entry["replicate"] == replicate]
            native_id = next(entry["run_id"] for entry in expected if entry["policy"]["name"] == "native")
            candidate_id = next(entry["run_id"] for entry in expected if entry["policy"]["name"] == name)
            native_run, candidate_run = observed.get(native_id), observed.get(candidate_id)
            scored = (native_run is not None and candidate_run is not None
                      and native_run["status"] not in {"unreviewed","unverified","disputed"}
                      and candidate_run["status"] not in {"unreviewed","unverified","disputed"})
            pairs.append({"replicate": replicate, "native_run": native_id, "candidate_run": candidate_id,
                          "quality_delta": int(candidate_run["accepted"]) - int(native_run["accepted"]) if scored else None,
                          "token_delta": candidate_run[basis] - native_run[basis]
                          if basis and native_run is not None and candidate_run is not None else None})
        native_cost, cost = baseline["tokens_per_accepted_result"], candidate["tokens_per_accepted_result"]
        comparisons.append({"policy_name": name, "pairs": pairs,
                            "observed_quality_regressions": sum(pair["quality_delta"] == -1 for pair in pairs),
                            "quality_status": "regression_observed" if any(pair["quality_delta"] == -1 for pair in pairs)
                            else "unknown" if any(pair["quality_delta"] is None for pair in pairs) else "no_regression_observed",
                            "token_saving_per_accepted_result": 1 - cost / native_cost
                            if native_cost is not None and cost is not None and native_cost > 0 else None})
    missing = sorted(scheduled.keys() - observed.keys())
    pending = [run["run_id"] for run in runs if run["status"] in {"unreviewed","unverified","disputed"}]
    reasons = []
    if experiment.agent is None:
        reasons.append("agent_profile_not_predeclared")
    if missing:
        reasons.append("scheduled_runs_missing")
    if pending:
        reasons.append("reviews_incomplete")
    if any(run["agent_kind"] != "agent" for run in runs):
        reasons.append("scripted_runs_do_not_measure_agent_performance")
    if basis is None:
        reasons.append("model_usage_unavailable")
    reasons.extend(sorted({reason for run in runs for reason in capture_blockers(run["capture"])}))
    reasons.extend(sorted({reason for run in runs for reason in run["usage_blockers"]}))
    reasons.extend(sorted({reason for run in runs for reason in run["validation_blockers"]}))
    reasons.extend(sorted({reason for run in runs for reason in run["coordination_blockers"]}))
    if len({(run["coordination"] or {}).get("registration_digest") for run in runs}) > 1:
        reasons.append("cohort_registrations_differ")
    for comparison in comparisons:
        comparison["eligible_for_agent_comparison"] = not reasons
        if reasons:
            comparison["token_saving_per_accepted_result"] = None
    return {
        "schema_version": 1, "experiment_id": experiment.digest, "snapshot": experiment.snapshot.digest,
        "split": experiment.split, "scheduled_runs": len(scheduled), "observed_runs": len(runs),
        "missing_run_ids": missing, "unreviewed_run_ids": pending,
        "readiness": "observed_tradeoffs_only" if not reasons else "incomplete", "readiness_reasons": reasons,
        "policies": policies, "comparisons": comparisons, "tokenizer": counter.description,
        "calibration_status": "not_calibrated", "default_selected": None, "runs": runs,
        "notes": [
            "Token spending includes failed/aborted/limit/rejected attempts; unknown usage is not zero.",
            "No confidence interval, quality non-inferiority claim, or population optimum is established here.",
            "Calibration and independent held-out workloads require a predeclared statistical design before default selection.",
            "Build/generation costs and host framing require separate accounting before overall cost claims.",
        ],
    }

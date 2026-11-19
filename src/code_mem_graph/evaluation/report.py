from __future__ import annotations

from collections import Counter
from typing import Callable

from code_mem_graph.tokens import PayloadCounter as PayloadCounter

from .models import ModelEvent, Suite, ToolEvent, Trace, fingerprint
from .accounting import usage_diagnostics, validate_trace_usage
from .validation import validation_summary


def capture_blockers(capture: dict | None) -> list[str]:
    """Known capture limitations disqualify efficiency claims in every report path."""
    if capture is None:
        return ["capture_undeclared"]
    reasons = []
    if capture["request_capture"] == "visible_only":
        reasons.append("only_visible_requests_captured")
    for field, reason in [
        ("provider_usage_available", "provider_usage_unavailable"),
        ("model_version_pinned", "provider_model_version_unpinned"),
        ("host_tools_enforced", "host_tool_isolation_not_enforced"),
        ("provider_token_budget_enforced", "provider_token_budget_not_enforced"),
    ]:
        if not capture[field]:
            reasons.append(reason)
    return reasons


def coordination_blockers(coordination: dict | None) -> list[str]:
    if coordination is None:
        return ["coordination_undeclared"]
    reasons = list(coordination.get("deviations", []))
    if not coordination.get("registration_digest"):
        reasons.append("cohort_registration_missing")
    return sorted(set(reasons))


def score_run(suite: Suite, trace: Trace, count: Callable[[object], int]) -> dict:
    validate_trace_usage([trace])
    if trace.suite_digest != suite.digest:
        raise ValueError(f"{trace.run_id}: suite fingerprint mismatch")
    task = suite.task(trace.task_id)
    revision = next(repo.revision for repo in suite.repositories if repo.id == task.repository_id)
    if trace.repository_revision != revision:
        raise ValueError(f"{trace.run_id}: repository revision mismatch")
    criterion_ids = {criterion.id for criterion in task.criteria}
    if set(trace.review.criteria) - criterion_ids:
        raise ValueError(f"{trace.run_id}: review contains unknown criterion IDs")
    if set(trace.review.acceptance_checks) - set(task.acceptance_checks):
        raise ValueError(f"{trace.run_id}: review contains unknown acceptance check IDs")
    tools = [event for event in trace.events if isinstance(event, ToolEvent)]
    turns = [event for event in trace.events if isinstance(event, ModelEvent)]
    met = {key for key, value in trace.review.criteria.items() if value == "met"}
    missing = sorted(criterion_ids - met)
    missing_critical = [criterion.id for criterion in task.criteria if criterion.critical and criterion.id not in met]
    incomplete_checks = [key for key in task.acceptance_checks if trace.review.acceptance_checks.get(key) != "passed"]
    within_limits = (len(tools) <= suite.protocol.max_tool_calls and len(turns) <= suite.protocol.max_model_turns
                     and trace.elapsed_ms <= suite.protocol.max_elapsed_ms)
    validation = validation_summary(suite,trace)
    accepted = (trace.outcome == "completed" and within_limits and trace.review.verdict == "accepted"
                and not missing and not incomplete_checks and not trace.review.unsupported_high_impact_claims
                and validation["status"] == "verified")
    if not within_limits or trace.outcome == "limit_exceeded":
        status = "limit_exceeded"
    elif trace.outcome != "completed":
        status = trace.outcome
    elif trace.review.verdict == "unreviewed":
        status = "unreviewed"
    elif validation["status"] in {"unverified","unreviewed"}:
        status = "unverified"
    elif validation["status"] == "disputed":
        status = "disputed"
    else:
        status = "accepted" if accepted else "rejected"
    diagnostics = [usage_diagnostics(turn.usage, turn.usage_capture) for turn in turns]
    usages = [result["observed_usage"] for result in diagnostics if result["observed_usage"] is not None]
    usage_complete = bool(turns) and all(result["recorded_scope_complete"] for result in diagnostics)
    provider_complete = bool(turns) and all(result["provider_complete"] for result in diagnostics)
    reported_input = sum(usage["input_tokens"] for usage in usages) if usages else None
    reported_output = sum(usage["output_tokens"] for usage in usages) if usages else None
    estimated_input = sum(count(turn.request) for turn in turns) if turns else None
    estimated_output = sum(count(turn.response) for turn in turns) if turns and all(turn.response_observed for turn in turns) else None
    return {
        "run_id": trace.run_id, "task_id": trace.task_id, "repository_id": task.repository_id,
        "repository_revision": revision, "policy": trace.policy.model_dump(),
        "policy_id": fingerprint(trace.policy.model_dump()), "agent_id": fingerprint(trace.agent.model_dump()),
        "agent": trace.agent.model_dump(), "review": trace.review.model_dump(exclude={"validation"}),
        "answer_validation":validation,"validation_blockers":validation["blockers"],
        "capture": trace.capture.model_dump() if trace.capture else None, "termination_reason": trace.termination_reason,
        "control_decisions": [event.model_dump(mode="json") for event in trace.events if event.kind == "control"],
        "coordination": trace.coordination, "coordination_blockers": coordination_blockers(trace.coordination),
        "agent_kind": trace.agent.kind, "replicate": trace.replicate, "cache_state": trace.cache_state,
        "status": status, "accepted": accepted, "reviewer": trace.review.reviewer,
        "coverage": len(met) / len(criterion_ids), "missing_criteria": missing,
        "missing_critical_criteria": missing_critical, "incomplete_checks": incomplete_checks,
        "unsupported_high_impact_claims": trace.review.unsupported_high_impact_claims,
        "elapsed_ms": trace.elapsed_ms, "tool_calls": len(tools), "model_turns": len(turns),
        "tool_request_tokens_estimate": sum(count(event.request) for event in tools),
        "tool_response_tokens_estimate": sum(count(event.response) for event in tools),
        "model_input_tokens_estimate": estimated_input, "model_output_tokens_estimate": estimated_output,
        "model_total_tokens_estimate": (estimated_input + estimated_output)
        if estimated_input is not None and estimated_output is not None else None,
        "usage_complete": usage_complete, "usage_turns": len(usages),
        "usage_completeness_basis": "at recorded scope; not proof of complete provider-call accounting",
        "provider_usage_complete": provider_complete, "usage_accounting": diagnostics,
        "usage_blockers": sorted({reason for result in diagnostics for reason in result["blockers"]}),
        "observed_input_tokens": reported_input, "observed_output_tokens": reported_output,
        "observed_cached_input_tokens": sum(usage["cached_input_tokens"] for usage in usages if "cached_input_tokens" in usage)
        if any("cached_input_tokens" in usage for usage in usages) else None,
        "observed_cache_write_input_tokens": sum(usage["cache_write_input_tokens"] for usage in usages
                                                  if "cache_write_input_tokens" in usage)
        if any("cache_write_input_tokens" in usage for usage in usages) else None,
        "observed_reasoning_output_tokens": sum(usage["reasoning_output_tokens"] for usage in usages
                                                  if "reasoning_output_tokens" in usage)
        if any("reasoning_output_tokens" in usage for usage in usages) else None,
        "detail_usage_turns": {key: sum(key in usage for usage in usages)
                               for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens")},
        "reported_model_total_tokens": (reported_input + reported_output) if usage_complete else None,
    }


def build_report(suite: Suite, traces: list[Trace], counter: PayloadCounter, *, baseline: str) -> dict:
    ids = [trace.run_id for trace in traces]
    if len(ids) != len(set(ids)):
        raise ValueError("run IDs must be unique")
    validate_trace_usage(traces)
    runs = [score_run(suite, trace, counter) for trace in traces]
    blocks: dict[tuple, list[dict]] = {}
    for run in runs:
        key = (run["task_id"], run["agent_id"], run["cache_state"], run["replicate"])
        blocks.setdefault(key, []).append(run)
    comparisons = []
    for block in blocks.values():
        policies = [run["policy_id"] for run in block]
        if len(policies) != len(set(policies)):
            raise ValueError("duplicate policy runs in a paired comparison block")
        references = [run for run in block if run["policy"]["name"] == baseline]
        if len(references) > 1:
            raise ValueError("baseline policy is ambiguous within a comparison block")
        for candidate in block:
            if candidate in references:
                continue
            reference = references[0] if references else None
            blockers = sorted(set(capture_blockers(candidate["capture"]) +
                                  candidate["usage_blockers"] + candidate["coordination_blockers"] + candidate["validation_blockers"] +
                                  (capture_blockers(reference["capture"]) + reference["usage_blockers"] +
                                   reference["coordination_blockers"] + reference["validation_blockers"] if reference else [])))
            if reference and (candidate["coordination"] or {}).get("registration_digest") != (reference["coordination"] or {}).get("registration_digest"):
                blockers.append("cohort_registrations_differ")
            reason = None
            if reference is None:
                reason = "no_matching_baseline"
            elif candidate["agent_kind"] != "agent":
                reason = "scripted_runs_do_not_measure_agent_performance"
            elif not reference["accepted"] or not candidate["accepted"]:
                reason = "both_runs_must_be_accepted"
            elif blockers:
                reason = "incomplete_capture"
            basis = None
            saving = None
            if reason is None:
                basis = ("reported_model_total_tokens" if reference["usage_complete"] and candidate["usage_complete"]
                         else "model_total_tokens_estimate")
                ref_tokens, candidate_tokens = reference[basis], candidate[basis]
                if ref_tokens is None or candidate_tokens is None or ref_tokens == 0:
                    reason = "insufficient_model_usage"
                else:
                    saving = 1 - candidate_tokens / ref_tokens
            comparisons.append({
                "candidate_run": candidate["run_id"], "baseline_run": reference["run_id"] if reference else None,
                "eligible": reason is None, "reason": reason, "token_basis": basis, "token_saving_fraction": saving,
                "capture_blockers": blockers,
            })
    return {
        "schema_version": 1, "suite_id": suite.id, "suite_digest": suite.digest,
        "protocol": suite.protocol.model_dump(), "baseline_policy_name": baseline,
        "tokenizer": counter.description, "runs": runs, "status_counts": dict(Counter(run["status"] for run in runs)),
        "comparisons": comparisons, "calibration_status": "not_calibrated",
        "notes": [
            "Tool payload estimates are diagnostic and are not added to model totals a second time.",
            "Cached input is a subset of provider input and is not added again.",
            "Missing provider usage is unknown, not zero; estimates do not include unrecorded host framing.",
            "This report does not select a default budget or establish statistical superiority.",
        ],
    }

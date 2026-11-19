from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import Contract, DirectoryEvidence, Evidence, KnowledgeMap
from .sources import EvidenceUnavailable, capture_directory, capture_file, policy_digest


class Assessment(Contract):
    item_id: str
    freshness: Literal["fresh", "stale", "unknown"]
    reasons: list[str]
    review_required: bool
    conflict_ids: list[str] = Field(default_factory=list)


def _probe(root: Path, evidence: Evidence) -> tuple[str, str | None]:
    if evidence.policy_digest != policy_digest():
        return "stale", f"evidence_policy_changed:{evidence.path}"
    try:
        if isinstance(evidence, DirectoryEvidence):
            current = capture_directory(root, evidence.path, recursive=evidence.recursive)
        else:
            current = capture_file(root, evidence.path, start_line=evidence.start_line, end_line=evidence.end_line)
    except EvidenceUnavailable as exc:
        status = "stale" if exc.reason in {"missing", "wrong_kind", "invalid_line_range", "repository_excluded"} else "unknown"
        return status, f"{exc.reason}:{evidence.path}"
    if current.digest != evidence.digest:
        return "stale", f"source_changed:{evidence.path}"
    return "fresh", None


def assess_map(knowledge: KnowledgeMap, root: Path) -> list[Assessment]:
    """Compute applicability from disk and transitive dependencies; never rewrite reviews."""
    knowledge = KnowledgeMap.model_validate(knowledge.model_dump(mode="json"))
    items = {item.id: item for item in knowledge.items}
    components = {component.id: component
                  for layer in (knowledge.generated, knowledge.curated) for component in layer.components}
    evidence_cache: dict[str, tuple[str, str | None]] = {}
    results: dict[str, Assessment] = {}
    remaining = {key: len(item.dependencies) for key, item in items.items()}
    dependents: dict[str, list[str]] = {key: [] for key in items}
    for item in items.values():
        for dependency in item.dependencies:
            dependents[dependency.item_id].append(item.id)
    pending = [key for key, count in remaining.items() if count == 0]
    while pending:
        item = items[pending.pop()]
        statuses, reasons = [], []
        if item.lifecycle == "retired":
            statuses.append("stale")
            reasons.append("item_retired")
        if any(components[key].lifecycle == "retired" for key in item.component_ids):
            statuses.append("stale")
            reasons.append("component_retired")
        if knowledge.refresh.status == "failed" and item.origin == "generated":
            statuses.append("unknown")
            reasons.append("refresh_failed")
        for evidence in item.evidence:
            identity = evidence.model_dump_json()
            if identity not in evidence_cache:
                evidence_cache[identity] = _probe(root, evidence)
            status, reason = evidence_cache[identity]
            statuses.append(status)
            if reason is not None:
                reasons.append(reason)
        for dependency in item.dependencies:
            parent = results[dependency.item_id]
            if items[dependency.item_id].content_digest != dependency.content_digest:
                statuses.append("stale")
                reasons.append(f"dependency_version_changed:{dependency.item_id}")
            statuses.append(parent.freshness)
            if parent.freshness != "fresh":
                reasons.append(f"dependency_{parent.freshness}:{dependency.item_id}")
            if parent.review_required:
                reasons.append(f"dependency_requires_review:{dependency.item_id}")
        if not item.evidence and not item.dependencies:
            statuses.append("unknown")
            reasons.append("no_verifiable_source")
        state = "stale" if "stale" in statuses else "unknown" if "unknown" in statuses else "fresh"
        review_required = (item.review.status == "rejected"
                           or any(results[dep.item_id].review_required for dep in item.dependencies)
                           or (item.basis in {"inferred", "maintainer"}
                               and (item.review.status != "accepted" or state != "fresh")))
        results[item.id] = Assessment(item_id=item.id, freshness=state, reasons=reasons,
                                      review_required=review_required)
        for dependent in dependents[item.id]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                pending.append(dependent)
    # A resolution only applies to the exact, still-fresh claims that were resolved.
    for layer in (knowledge.generated, knowledge.curated):
        for conflict in layer.conflicts:
            unresolved = (conflict.status == "unresolved"
                          or any(conflict.resolved_versions.get(key) != items[key].content_digest
                                 or results[key].freshness != "fresh" for key in conflict.item_ids))
            if unresolved:
                for key in conflict.item_ids:
                    data = results[key].model_dump()
                    data["conflict_ids"].append(conflict.id)
                    data["review_required"] = True
                    results[key] = Assessment.model_validate(data)
    # Review blockers from conflicts must propagate through the evidence graph too.
    blocked = [key for key, result in results.items() if result.review_required]
    while blocked:
        for dependent in dependents[blocked.pop()]:
            if not results[dependent].review_required:
                data = results[dependent].model_dump()
                data["review_required"] = True
                data["reasons"].append("dependency_requires_review")
                results[dependent] = Assessment.model_validate(data)
                blocked.append(dependent)
    return [results[item.id] for item in knowledge.items]


def assess_gaps(
    knowledge: KnowledgeMap,
    assessments: dict[str, Assessment],
    *,
    scope_ids: set[str] | None = None,
    applicable_components: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Compute effective gap status from stored support and accepted fresh contributions."""
    items = {item.id: item for item in knowledge.items}
    all_ready = {
        item.id
        for item in knowledge.items
        if item.lifecycle == "active"
        and assessments[item.id].freshness == "fresh"
        and not assessments[item.id].review_required
    }
    scoped_ready = all_ready if scope_ids is None else all_ready & scope_ids
    addressed: dict[str, list[str]] = {}
    for item in knowledge.items:
        for gap_id in item.addresses_gap_ids:
            addressed.setdefault(gap_id, []).append(item.id)
    results: dict[str, dict[str, object]] = {}
    for layer in (knowledge.generated, knowledge.curated):
        for gap in layer.gaps:
            stored_ready = (
                [key for key in gap.resolved_by if key in scoped_ready]
                if gap.status == "resolved"
                else []
            )
            addressed_by = addressed.get(gap.id, [])
            accepted_ready = [
                key
                for key in addressed_by
                if key in all_ready
                and items[key].review.status == "accepted"
                and (
                    applicable_components is None
                    or key in scoped_ready
                    or not items[key].component_ids
                    or bool(applicable_components.intersection(items[key].component_ids))
                )
            ]
            effective_resolved_by = list(dict.fromkeys([*stored_ready, *accepted_ready]))
            results[gap.id] = {
                "effective_status": "resolved" if effective_resolved_by else "open",
                "addressed_by": addressed_by,
                "effective_resolved_by": effective_resolved_by,
            }
    return results

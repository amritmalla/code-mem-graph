"""Explicit maintainer inspection, refresh, and version-bound review operations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import yaml

from code_mem_graph.serialization import fingerprint

from .build import BuildError, _atomic_write, _build_lock, _existing, build_repository, load_repository_map, render_brief
from .freshness import assess_gaps, assess_map
from .models import DirectoryEvidence, ItemDependency, KnowledgeItem, KnowledgeMap, ReviewRecord, review_item
from .sources import EvidenceUnavailable, capture_directory, capture_file


def status_report(root: Path, document: KnowledgeMap | None = None) -> dict:
    document = document or load_repository_map(root)
    assessments = {entry.item_id: entry for entry in assess_map(document, root)}
    active = [item for item in document.items if item.lifecycle == "active"]
    gap_states = assess_gaps(document, assessments)
    gaps, conflicts = [], []
    for layer in (document.generated, document.curated):
        for gap in layer.gaps:
            info = gap_states[gap.id]
            row = {**gap.model_dump(mode="json"), "effective_status": info["effective_status"]}
            if info["addressed_by"]:
                row["addressed_by"] = info["addressed_by"]
                row["effective_resolved_by"] = info["effective_resolved_by"]
            gaps.append(row)
        for conflict in layer.conflicts:
            unresolved = any(conflict.id in assessments[key].conflict_ids for key in conflict.item_ids)
            conflicts.append({**conflict.model_dump(mode="json"), "effective_status": "unresolved" if unresolved else "resolved"})
    return {
        "repository_id": document.repository_id, "map_digest": fingerprint(document.model_dump(mode="json")),
        "refresh": document.refresh.model_dump(), "active_claims": len(active),
        "retired_claims": len(document.items) - len(active),
        "source_status": {state: sum(assessments[item.id].freshness == state for item in active)
                          for state in ("fresh", "stale", "unknown")},
        "review_queue": [item.id for item in active if assessments[item.id].review_required],
        "assessments": [assessments[item.id].model_dump() for item in document.items],
        "gaps": gaps, "conflicts": conflicts,
    }


def item_report(root: Path, document: KnowledgeMap, item_id: str) -> dict:
    item = next((entry for entry in document.items if entry.id == item_id), None)
    if item is None:
        raise ValueError("unknown knowledge item ID")
    assessments = {entry.item_id: entry for entry in assess_map(document, root)}
    assessment = assessments[item_id]
    current_evidence = []
    for reference in item.evidence:
        try:
            current = (capture_directory(root, reference.path, recursive=reference.recursive)
                       if isinstance(reference, DirectoryEvidence) else
                       capture_file(root, reference.path, start_line=reference.start_line, end_line=reference.end_line))
            current_evidence.append({"stored": reference.model_dump(), "current": current.model_dump()})
        except EvidenceUnavailable as exc:
            current_evidence.append({"stored": reference.model_dump(), "current": None, "reason": exc.reason})
    result = {
        "item": item.model_dump(mode="json"),
        "content_digest": item.content_digest,
        "map_digest": fingerprint(document.model_dump(mode="json")),
        "assessment": assessment.model_dump(),
        "current_evidence": current_evidence,
        "dependents": [entry.id for entry in document.items if any(dep.item_id == item_id for dep in entry.dependencies)],
    }
    if item.addresses_gap_ids:
        gap_states = assess_gaps(document, assessments)
        result["addressed_gaps"] = [
            {
                "id": gid,
                "effective_status": gap_states[gid]["effective_status"],
                "effective_resolved_by": gap_states[gid]["effective_resolved_by"],
            }
            for gid in item.addresses_gap_ids
        ]
    return result


def show_item(root: Path, item_id: str) -> dict:
    return item_report(root, load_repository_map(root), item_id)


def refresh_repository(root: Path) -> dict:
    previous = load_repository_map(root)
    document = build_repository(root, expected_digest=fingerprint(previous.model_dump(mode="json")))
    old = {item.id: item for item in previous.generated.items}
    new = {item.id: item for item in document.generated.items}
    report = status_report(root, document)
    report["changes"] = {
        "added": [key for key in new if key not in old],
        "changed": [key for key in new if key in old and new[key].content_digest != old[key].content_digest],
        "retired": [key for key in new if new[key].lifecycle == "retired"
                    and (key not in old or old[key].lifecycle != "retired")],
        "reactivated": [key for key in new if key in old and old[key].lifecycle == "retired" and new[key].lifecycle == "active"],
        "curated_preserved": document.curated == previous.curated,
    }
    return report


def _edit(root: Path, transform: Callable[[KnowledgeMap], KnowledgeMap]) -> KnowledgeMap:
    with _build_lock(root):
        raw = _existing(root, "map.yaml")
        if raw is None:
            raise ValueError("build knowledge before editing or reviewing it")
        current = KnowledgeMap.model_validate(yaml.safe_load(raw.decode("utf-8")))
        brief_raw = _existing(root, "README.md")
        updated = transform(current)
        updated = KnowledgeMap.model_validate(updated.model_dump(mode="json"))
        brief = render_brief(updated, root)
        _atomic_write(root, "map.yaml", yaml.safe_dump(updated.model_dump(mode="json"), sort_keys=False,
                                                       allow_unicode=True), expected=raw)
        try:
            _atomic_write(root, "README.md", brief, expected=brief_raw)
        except Exception as exc:
            raise BuildError("knowledge map saved, but its Markdown view could not be updated; rerun the build") from exc
        return updated


def record_review(root: Path, item_id: str, *, expected_digest: str, reviewer: str,
                  status: Literal["accepted", "rejected"], note: str | None = None,
                  refresh_evidence: bool = False) -> KnowledgeMap:
    def transform(document):
        items = {item.id: item for item in document.items}
        if item_id not in items:
            raise ValueError("unknown knowledge item ID")
        item = items[item_id]
        if item.content_digest != expected_digest:
            raise ValueError("claim version changed; inspect it again before reviewing")
        if item.lifecycle == "retired":
            raise ValueError("retired claims cannot receive a current review")
        data = item.model_dump(mode="json")
        data["review"] = ReviewRecord().model_dump(mode="json")
        if refresh_evidence:
            if item.origin != "maintainer":
                raise ValueError("rebuild generated facts; only curated evidence can be explicitly recaptured")
            data["evidence"] = [(capture_directory(root, ref.path, recursive=ref.recursive)
                                 if isinstance(ref, DirectoryEvidence) else
                                 capture_file(root, ref.path, start_line=ref.start_line, end_line=ref.end_line)).model_dump(mode="json")
                                for ref in item.evidence]
            data["dependencies"] = [{"item_id": dep.item_id, "content_digest": items[dep.item_id].content_digest}
                                    for dep in item.dependencies]
        candidate = KnowledgeItem.model_validate(data)
        replacement = review_item(candidate, reviewer=reviewer, status=status, note=note)
        new_data = document.model_dump(mode="json")
        layer = "curated" if item.origin == "maintainer" else "generated"
        new_data[layer]["items"] = [replacement.model_dump(mode="json") if entry["id"] == item_id else entry
                                     for entry in new_data[layer]["items"]]
        updated = KnowledgeMap.model_validate(new_data)
        if status == "accepted":
            assessments = {entry.item_id: entry for entry in assess_map(updated, root)}
            attestation = item.basis == "maintainer" and not candidate.evidence and not candidate.dependencies
            if assessments[item_id].freshness != "fresh" and not attestation:
                raise ValueError("review acceptance requires current evidence; rebuild facts or explicitly recapture curated evidence")
            if any(assessments[dep.item_id].review_required for dep in candidate.dependencies):
                raise ValueError("review the blocked evidence dependencies first")
        return updated
    return _edit(root, transform)


def resolve_conflict(root: Path, conflict_id: str, *, expected_map_digest: str,
                     reviewer: str, explanation: str) -> KnowledgeMap:
    def transform(document):
        if fingerprint(document.model_dump(mode="json")) != expected_map_digest:
            raise ValueError("knowledge map version changed; inspect the conflict again")
        items = {item.id: item for item in document.items}
        assessments = {entry.item_id: entry for entry in assess_map(document, root)}
        data = document.model_dump(mode="json")
        for layer in ("generated", "curated"):
            for conflict in data[layer]["conflicts"]:
                if conflict["id"] == conflict_id:
                    if any(assessments[key].freshness != "fresh" for key in conflict["item_ids"]):
                        raise ValueError("conflict resolution requires current evidence for every competing claim")
                    conflict.update(status="resolved", resolution=explanation, resolved_by=reviewer,
                                    resolved_versions={key: items[key].content_digest for key in conflict["item_ids"]})
                    # An explicit human resolution is also durable authored knowledge.
                    # Keep its text if a later extractor changes conflict membership.
                    prefix = "conflict-resolution/" + fingerprint(conflict_id)[:20] + "/"
                    record_id = prefix + fingerprint({"versions": conflict["resolved_versions"],
                                                      "reviewer": reviewer, "explanation": explanation})[:20]
                    record = review_item(KnowledgeItem(
                        id=record_id, topic="decision", title=f"Resolution: {conflict_id}", statement=explanation,
                        origin="maintainer", basis="maintainer", author=reviewer,
                        dependencies=[ItemDependency(item_id=key, content_digest=items[key].content_digest)
                                      for key in conflict["item_ids"]]), reviewer=reviewer, status="accepted")
                    for prior in data["curated"]["items"]:
                        if prior["id"].startswith(prefix) and prior["id"] != record_id:
                            prior["lifecycle"] = "retired"
                    data["curated"]["items"] = [entry for entry in data["curated"]["items"] if entry["id"] != record_id]
                    data["curated"]["items"].append(record.model_dump(mode="json"))
                    return KnowledgeMap.model_validate(data)
        raise ValueError("unknown conflict ID")
    return _edit(root, transform)

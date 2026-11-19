"""Readable maintainer contribution workflow with version guards and durable gap links."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
import yaml

from code_mem_graph.serialization import fingerprint

from .build import load_repository_map
from .models import (
    Contract,
    DirectoryEvidence,
    Facet,
    Identity,
    ItemDependency,
    KnowledgeItem,
    KnowledgeMap,
    OnboardingQuestion,
    ReviewRecord,
    SourcePath,
    Text,
    Topic,
)
from .sources import capture_directory, capture_file
from .workflow import _edit, item_report

CONFLICT_RESOLUTION_PREFIX = "conflict-resolution/"
ContributionBasis = Literal["documented", "inferred", "maintainer"]


class ContributionFileEvidence(Contract):
    kind: Literal["file"] = "file"
    path: SourcePath
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_range(self):
        if self.path == ".":
            raise ValueError("file evidence must name a file")
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("line ranges require both start_line and end_line")
        if self.start_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line precedes start_line")
        return self


class ContributionDirectoryEvidence(Contract):
    kind: Literal["directory"] = "directory"
    path: SourcePath
    recursive: bool = False


ContributionEvidence = Annotated[
    ContributionFileEvidence | ContributionDirectoryEvidence,
    Field(discriminator="kind"),
]


class ContributionInput(Contract):
    """Compact maintainer authoring input; fingerprints and dependency versions are captured internally."""

    id: Identity
    topic: Topic
    title: Text
    statement: Text
    author: Text
    basis: ContributionBasis = "maintainer"
    component_ids: list[Identity] = Field(default_factory=list)
    facets: list[Facet] = Field(default_factory=list)
    onboarding: list[OnboardingQuestion] = Field(default_factory=list)
    critical: bool = False
    evidence: list[ContributionEvidence] = Field(default_factory=list)
    dependencies: list[Identity] = Field(default_factory=list)
    addresses_gap_ids: list[Identity] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value):
        if not isinstance(value, list):
            return value
        normalized = []
        for entry in value:
            if isinstance(entry, str):
                normalized.append({"kind": "file", "path": entry})
            elif isinstance(entry, dict) and "kind" not in entry:
                normalized.append({"kind": "file", **entry})
            else:
                normalized.append(entry)
        return normalized

    @model_validator(mode="after")
    def valid_contribution(self):
        if self.id.startswith(CONFLICT_RESOLUTION_PREFIX):
            raise ValueError("conflict-resolution/ item IDs are reserved for conflict resolution")
        if self.basis == "documented" and not self.evidence:
            raise ValueError("documented contributions require direct source evidence")
        if self.basis == "inferred" and not (self.evidence or self.dependencies):
            raise ValueError("inferred contributions require source evidence or knowledge dependencies")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("duplicate component references")
        if len(self.facets) != len(set(self.facets)):
            raise ValueError("duplicate orientation facets")
        if len(self.onboarding) != len(set(self.onboarding)):
            raise ValueError("duplicate onboarding questions")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("duplicate item dependencies")
        if self.id in self.dependencies:
            raise ValueError("a contribution cannot depend on itself")
        if len(self.addresses_gap_ids) != len(set(self.addresses_gap_ids)):
            raise ValueError("duplicate gap references")
        evidence_keys = [
            (
                entry.kind,
                entry.path,
                getattr(entry, "start_line", None),
                getattr(entry, "end_line", None),
                getattr(entry, "recursive", None),
            )
            for entry in self.evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("duplicate evidence references")
        return self


def load_contribution(path: Path) -> ContributionInput:
    """Parse and strictly validate a YAML or JSON maintainer contribution file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("contribution file must contain a YAML or JSON mapping")
    return ContributionInput.model_validate(raw)


def contribution_from_item(item: KnowledgeItem) -> dict:
    """Convert an active curated item into editable hash-free contribution fields."""
    if item.origin != "maintainer":
        raise ValueError("only curated maintainer items can be exported as contributions")
    if item.id.startswith(CONFLICT_RESOLUTION_PREFIX):
        raise ValueError("conflict-resolution/ items are managed through resolve-conflict")
    if item.lifecycle == "retired":
        raise ValueError("retired curated items cannot be exported for update")
    if item.basis not in {"documented", "inferred", "maintainer"}:
        raise ValueError("observed items require choosing documented, inferred, or maintainer basis")
    evidence: list[dict] = []
    for ref in item.evidence:
        if isinstance(ref, DirectoryEvidence):
            entry: dict = {"kind": "directory", "path": ref.path}
            if ref.recursive:
                entry["recursive"] = True
            evidence.append(entry)
        else:
            entry = {"path": ref.path}
            if ref.start_line is not None and ref.end_line is not None:
                entry["start_line"] = ref.start_line
                entry["end_line"] = ref.end_line
            evidence.append(entry)
    return {
        "id": item.id,
        "topic": item.topic,
        "title": item.title,
        "statement": item.statement,
        "basis": item.basis,
        "author": item.author,
        "component_ids": list(item.component_ids),
        "facets": list(item.facets),
        "onboarding": list(item.onboarding),
        "critical": item.critical,
        "evidence": evidence,
        "dependencies": [dep.item_id for dep in item.dependencies],
        "addresses_gap_ids": list(item.addresses_gap_ids),
    }


def render_contribution_template(
    *,
    output_format: Literal["yaml", "json"] = "yaml",
    item: KnowledgeItem | None = None,
    item_id: str | None = None,
    topic: Topic | None = None,
    author: str | None = None,
    basis: ContributionBasis | None = None,
    addresses_gap_ids: list[str] | None = None,
) -> str:
    """Render a starter contribution template or export an existing curated item."""
    if item is not None:
        data = contribution_from_item(item)
        if output_format == "json":
            return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    starter = ContributionInput(
        id=item_id or "architecture/example-rationale",
        topic=topic or "architecture",
        title="Concise title for this maintainer explanation",
        statement=(
            "Explain the architectural decision, convention, terminology, or workflow here.\n"
            "Replace this placeholder statement and cite real repository paths before adding."
        ),
        basis=basis or "maintainer",
        author=author or "your-name",
        component_ids=["project"],
        facets=[],
        onboarding=[],
        critical=False,
        evidence=[ContributionFileEvidence(path="README.md", start_line=1, end_line=10)],
        dependencies=[],
        addresses_gap_ids=list(addresses_gap_ids or []),
    )
    data = {
        "id": starter.id,
        "topic": starter.topic,
        "title": starter.title,
        "statement": starter.statement,
        "basis": starter.basis,
        "author": starter.author,
        "component_ids": list(starter.component_ids),
        "facets": list(starter.facets),
        "onboarding": list(starter.onboarding),
        "critical": starter.critical,
        "evidence": [{"path": "README.md", "start_line": 1, "end_line": 10}],
        "dependencies": list(starter.dependencies),
        "addresses_gap_ids": list(starter.addresses_gap_ids),
    }
    if output_format == "json":
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    header = (
        "# code-mem-graph maintainer contribution template\n"
        "# Topics: overview | architecture | domain | conventions | workflow | decision\n"
        "# Basis: documented (requires evidence) | inferred (evidence or dependencies) | maintainer (may be source-free)\n"
        "# Evidence: repository-relative POSIX file paths (optional start_line/end_line) or {kind: directory, path: ..., recursive: true}.\n"
        "# Fingerprints and dependency versions are captured automatically; contributions start unreviewed.\n"
        "# Use `code-mem-graph knowledge add <file> --root <repo>` then inspect and `knowledge review`.\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _materialize_item(
    root: Path, document: KnowledgeMap, contribution: ContributionInput,
    *, existing: KnowledgeItem | None = None,
) -> KnowledgeItem:
    components = {
        entry.id: entry
        for layer in (document.generated, document.curated)
        for entry in layer.components
    }
    items = {item.id: item for item in document.items}
    gaps = {gap.id for layer in (document.generated, document.curated) for gap in layer.gaps}

    for component_id in contribution.component_ids:
        component = components.get(component_id)
        if component is None:
            raise ValueError(f"unknown component ID: {component_id}")
        if component.lifecycle == "retired":
            raise ValueError(f"component is retired and cannot be assigned: {component_id}")

    dependencies: list[ItemDependency] = []
    for dep_id in contribution.dependencies:
        dependency = items.get(dep_id)
        if dependency is None:
            raise ValueError(f"unknown dependency item ID: {dep_id}")
        if dependency.lifecycle == "retired":
            raise ValueError(f"dependency item is retired: {dep_id}")
        dependencies.append(ItemDependency(item_id=dep_id, content_digest=dependency.content_digest))

    for gap_id in contribution.addresses_gap_ids:
        if gap_id not in gaps:
            raise ValueError(f"unknown knowledge gap ID: {gap_id}")

    captured_evidence = [
        capture_directory(root, ref.path, recursive=ref.recursive)
        if isinstance(ref, ContributionDirectoryEvidence)
        else capture_file(root, ref.path, start_line=ref.start_line, end_line=ref.end_line)
        for ref in contribution.evidence
    ]

    # The compact authoring format does not edit structured P03 metadata. Keep
    # it on updates, and validate its evidence against the proposed contribution
    # before publishing anything instead of silently discarding either field.
    relationships = list(existing.relationships) if existing is not None else []
    command_checks = list(existing.command_checks) if existing is not None else []
    required_paths = {path for edge in relationships for path in (edge.source, edge.target)}
    required_paths.update(check.document for check in command_checks)
    required_paths.update(check.manifest for check in command_checks if check.manifest is not None)
    missing_paths = required_paths - {reference.path for reference in captured_evidence}
    if missing_paths:
        raise ValueError(
            "updates preserve existing relationships and command checks; retain evidence for: "
            + ", ".join(sorted(missing_paths))
        )

    return KnowledgeItem(
        id=contribution.id,
        topic=contribution.topic,
        facets=list(contribution.facets),
        onboarding=list(contribution.onboarding),
        title=contribution.title,
        statement=contribution.statement,
        origin="maintainer",
        basis=contribution.basis,
        author=contribution.author,
        component_ids=list(contribution.component_ids),
        evidence=captured_evidence,
        dependencies=dependencies,
        relationships=relationships,
        command_checks=command_checks,
        addresses_gap_ids=list(contribution.addresses_gap_ids),
        critical=contribution.critical,
        lifecycle="active",
        review=ReviewRecord(),
    )


def add_contribution(
    root: Path,
    contribution: ContributionInput,
    *,
    expected_map_digest: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Create a new unreviewed curated contribution with captured evidence and dependency versions."""
    contribution = ContributionInput.model_validate(contribution.model_dump(mode="json"))

    def transform(document: KnowledgeMap) -> KnowledgeMap:
        if expected_map_digest is not None and fingerprint(document.model_dump(mode="json")) != expected_map_digest:
            raise ValueError("knowledge map version changed; inspect the current map and retry")
        if any(item.id == contribution.id for item in document.generated.items):
            raise ValueError(f"cannot overwrite generated knowledge item: {contribution.id}")
        if any(item.id == contribution.id for item in document.curated.items):
            raise ValueError(
                f"knowledge item ID already exists: {contribution.id}; use update to revise a curated contribution"
            )
        new_item = _materialize_item(root, document, contribution)
        data = document.model_dump(mode="json")
        data["curated"]["items"].append(new_item.model_dump(mode="json"))
        return KnowledgeMap.model_validate(data)

    if dry_run:
        current = load_repository_map(root)
        proposed = transform(current)
        report = item_report(root, proposed, contribution.id)
        proposed_digest = report["map_digest"]
        report["map_digest"] = fingerprint(current.model_dump(mode="json"))
        report["proposed_map_digest"] = proposed_digest
        report["action"] = "add"
        report["dry_run"] = True
        return report

    updated = _edit(root, transform)
    report = item_report(root, updated, contribution.id)
    report["action"] = "add"
    report["dry_run"] = False
    return report


def update_contribution(
    root: Path,
    contribution: ContributionInput,
    *,
    expected_digest: str,
    expected_map_digest: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Revise an existing curated contribution with version protection; resets prior review."""
    contribution = ContributionInput.model_validate(contribution.model_dump(mode="json"))
    previous_meta: dict[str, str] = {}

    def transform(document: KnowledgeMap) -> KnowledgeMap:
        if expected_map_digest is not None and fingerprint(document.model_dump(mode="json")) != expected_map_digest:
            raise ValueError("knowledge map version changed; inspect the current map and retry")
        if any(item.id == contribution.id for item in document.generated.items):
            raise ValueError(f"generated knowledge items cannot be updated as maintainer contributions: {contribution.id}")
        existing = next((item for item in document.curated.items if item.id == contribution.id), None)
        if existing is None:
            raise ValueError(
                f"unknown curated knowledge item ID: {contribution.id}; use add to create a new contribution"
            )
        if existing.lifecycle == "retired":
            raise ValueError(f"retired curated item cannot be updated: {contribution.id}")
        if existing.content_digest != expected_digest:
            raise ValueError("claim version changed; inspect it again before updating")
        previous_meta["previous_content_digest"] = existing.content_digest
        previous_meta["previous_review_status"] = existing.review.status
        replacement = _materialize_item(root, document, contribution, existing=existing)
        data = document.model_dump(mode="json")
        data["curated"]["items"] = [
            replacement.model_dump(mode="json") if entry["id"] == contribution.id else entry
            for entry in data["curated"]["items"]
        ]
        return KnowledgeMap.model_validate(data)

    if dry_run:
        current = load_repository_map(root)
        proposed = transform(current)
        report = item_report(root, proposed, contribution.id)
        proposed_digest = report["map_digest"]
        report["map_digest"] = fingerprint(current.model_dump(mode="json"))
        report["proposed_map_digest"] = proposed_digest
        report["action"] = "update"
        report["dry_run"] = True
        report.update(previous_meta)
        return report

    updated = _edit(root, transform)
    report = item_report(root, updated, contribution.id)
    report["action"] = "update"
    report["dry_run"] = False
    report.update(previous_meta)
    return report


def retire_contribution(
    root: Path,
    item_id: str,
    *,
    expected_digest: str,
    expected_map_digest: str | None = None,
) -> dict:
    """Retire an active curated contribution while preserving its historical review record."""
    if item_id.startswith(CONFLICT_RESOLUTION_PREFIX):
        raise ValueError("conflict-resolution/ items are managed through resolve-conflict")

    def transform(document: KnowledgeMap) -> KnowledgeMap:
        if expected_map_digest is not None and fingerprint(document.model_dump(mode="json")) != expected_map_digest:
            raise ValueError("knowledge map version changed; inspect the current map and retry")
        if any(item.id == item_id for item in document.generated.items):
            raise ValueError(f"generated knowledge items cannot be retired as maintainer contributions: {item_id}")
        existing = next((item for item in document.curated.items if item.id == item_id), None)
        if existing is None:
            raise ValueError(f"unknown curated knowledge item ID: {item_id}")
        if existing.lifecycle == "retired":
            raise ValueError(f"curated knowledge item is already retired: {item_id}")
        if existing.content_digest != expected_digest:
            raise ValueError("claim version changed; inspect it again before retiring")
        data = document.model_dump(mode="json")
        for entry in data["curated"]["items"]:
            if entry["id"] == item_id:
                entry["lifecycle"] = "retired"
                break
        return KnowledgeMap.model_validate(data)

    updated = _edit(root, transform)
    report = item_report(root, updated, item_id)
    report["action"] = "retire"
    return report

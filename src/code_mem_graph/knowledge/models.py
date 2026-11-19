from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_serializer, model_validator,
)

from code_mem_graph.serialization import fingerprint

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identity = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._/-]*$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Topic = Literal["overview", "architecture", "domain", "conventions", "workflow", "decision"]
Facet = Literal["purpose", "stack", "entrypoints", "components", "constraints", "workflow"]
OnboardingQuestion = Literal["purpose", "connection", "prerequisites", "responsibilities", "conventions", "validation"]


def portable_path(value: str) -> str:
    """Require one unambiguous, portable repository-relative POSIX spelling."""
    if value == ".":
        return value
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if not value or any(ord(char) < 32 or char in '\\<>:"|?*' for char in value):
        raise ValueError("source path must be a portable repository-relative POSIX path")
    for part in value.split("/"):
        if part in {"", ".", ".."} or part.endswith((" ", ".")) or part.split(".")[0].upper() in reserved:
            raise ValueError("source path contains an unsafe or noncanonical component")
    return value


SourcePath = Annotated[str, AfterValidator(portable_path)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class FileEvidence(Contract):
    kind: Literal["file"] = "file"
    path: SourcePath
    digest: Digest
    policy_digest: Digest
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


class DirectoryEvidence(Contract):
    kind: Literal["directory"] = "directory"
    path: SourcePath
    digest: Digest
    policy_digest: Digest
    recursive: bool = False


Evidence = Annotated[FileEvidence | DirectoryEvidence, Field(discriminator="kind")]


class ItemDependency(Contract):
    item_id: Identity
    content_digest: Digest


class ReviewRecord(Contract):
    status: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"
    reviewer: Text | None = None
    content_digest: Digest | None = None
    note: Text | None = None

    @model_validator(mode="after")
    def bound_review(self):
        if self.status == "unreviewed":
            if self.reviewer is not None or self.content_digest is not None:
                raise ValueError("an unreviewed item cannot carry a scored review")
        elif self.reviewer is None or self.content_digest is None:
            raise ValueError("a scored review requires a reviewer and content digest")
        return self


class Generator(Contract):
    name: Text
    version: Text


class LocationRelationship(Contract):
    """A documented location or a statically resolved module declaration."""

    kind: Literal["documented_location", "static_import"]
    source: SourcePath
    target: SourcePath

    @model_validator(mode="after")
    def distinct_locations(self):
        if self.source == self.target:
            raise ValueError("a location relationship requires distinct endpoints")
        return self


class CommandCheck(Contract):
    document: SourcePath
    manager: Literal["npm", "pnpm", "yarn", "bun"]
    script: Text
    directory: SourcePath | None = None
    manifest: SourcePath | None = None
    status: Literal["declared_script", "script_not_declared", "scope_unknown"]
    scope_basis: Literal["document_directory", "literal_cd", "explicit_directory_option", "unknown"]
    similar_names: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_scope(self):
        if self.status == "scope_unknown":
            if self.manifest is not None or self.scope_basis != "unknown":
                raise ValueError("unknown command scope cannot assert a manifest")
        elif self.directory is None or self.manifest != (
            "package.json" if self.directory == "." else self.directory + "/package.json"
        ) or self.scope_basis == "unknown":
            raise ValueError("command comparison requires a scoped package manifest")
        return self


class KnowledgeItem(Contract):
    id: Identity
    topic: Topic
    facets: list[Facet] = Field(default_factory=list)
    onboarding: list[OnboardingQuestion] = Field(default_factory=list)
    title: Text
    statement: Text
    origin: Literal["generated", "maintainer"]
    basis: Literal["observed", "documented", "inferred", "maintainer"]
    author: Text | None = None
    generator: Generator | None = None
    component_ids: list[Identity] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    dependencies: list[ItemDependency] = Field(default_factory=list)
    relationships: list[LocationRelationship] = Field(default_factory=list)
    command_checks: list[CommandCheck] = Field(default_factory=list)
    addresses_gap_ids: list[Identity] = Field(default_factory=list)
    critical: bool = False
    lifecycle: Literal["active", "retired"] = "active"
    review: ReviewRecord = Field(default_factory=ReviewRecord)

    @model_serializer(mode="wrap")
    def serialize_optional_onboarding(self, handler):
        # Keep historical claim/review/map digests stable when this additive
        # annotation is absent. Adding a real annotation changes the claim hash.
        data = handler(self)
        if not data.get("onboarding"):
            data.pop("onboarding", None)
        if not data.get("relationships"):
            data.pop("relationships", None)
        if not data.get("command_checks"):
            data.pop("command_checks", None)
        if not data.get("addresses_gap_ids"):
            data.pop("addresses_gap_ids", None)
        return data

    @property
    def content_digest(self) -> str:
        # Retirement is a separate applicability signal; it does not rewrite a claim.
        return fingerprint(self.model_dump(mode="json", exclude={"review", "lifecycle"}))

    @model_validator(mode="after")
    def valid_claim(self):
        if self.origin == "generated":
            if self.generator is None or self.author is not None or self.basis == "maintainer":
                raise ValueError("generated items require a generator, not a maintainer attribution")
        elif self.author is None or self.generator is not None:
            raise ValueError("maintainer items require an author, not a generator")
        if self.basis in {"observed", "documented"} and not self.evidence:
            raise ValueError("observed/documented claims require direct source evidence")
        if self.basis == "inferred" and not (self.evidence or self.dependencies):
            raise ValueError("inferences require evidence or versioned item dependencies")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("duplicate component references")
        if len(self.facets) != len(set(self.facets)):
            raise ValueError("duplicate orientation facets")
        if len(self.onboarding) != len(set(self.onboarding)):
            raise ValueError("duplicate onboarding questions")
        edges = [(edge.kind, edge.source, edge.target) for edge in self.relationships]
        if len(edges) != len(set(edges)):
            raise ValueError("duplicate location relationships")
        cited = {reference.path for reference in self.evidence}
        if any(check.document not in cited or (check.manifest is not None and check.manifest not in cited)
               for check in self.command_checks):
            raise ValueError("command comparison requires document and manifest evidence")
        if any(edge.source not in cited or edge.target not in cited for edge in self.relationships):
            raise ValueError("relationship endpoints require direct source evidence")
        dependency_ids = [dep.item_id for dep in self.dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("duplicate item dependencies")
        if len(self.addresses_gap_ids) != len(set(self.addresses_gap_ids)):
            raise ValueError("duplicate gap references")
        if self.review.status != "unreviewed" and self.review.content_digest != self.content_digest:
            raise ValueError("review does not match this claim version")
        return self


class Component(Contract):
    id: Identity
    name: Text
    paths: list[SourcePath] = Field(min_length=1)
    parent_id: Identity | None = None
    lifecycle: Literal["active", "retired"] = "active"


class KnowledgeGap(Contract):
    id: Identity
    topic: Topic
    description: Text
    component_ids: list[Identity] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "open"
    resolved_by: list[Identity] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_resolution(self):
        if (self.status == "resolved") != bool(self.resolved_by):
            raise ValueError("only resolved gaps must identify supporting knowledge items")
        return self


class KnowledgeConflict(Contract):
    id: Identity
    item_ids: list[Identity] = Field(min_length=2)
    description: Text
    status: Literal["unresolved", "resolved"] = "unresolved"
    resolution: Text | None = None
    resolved_by: Text | None = None
    resolved_versions: dict[Identity, Digest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def explicit_resolution(self):
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("a conflict must reference distinct items")
        if self.status == "resolved":
            if self.resolution is None or self.resolved_by is None or set(self.resolved_versions) != set(self.item_ids):
                raise ValueError("resolved conflicts require an explanation, resolver, and all claim versions")
        elif self.resolution is not None or self.resolved_by is not None or self.resolved_versions:
            raise ValueError("unresolved conflicts cannot carry a resolution")
        return self


class KnowledgeLayer(Contract):
    components: list[Component] = Field(default_factory=list)
    items: list[KnowledgeItem] = Field(default_factory=list)
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    conflicts: list[KnowledgeConflict] = Field(default_factory=list)


class RefreshState(Contract):
    status: Literal["ready", "failed"] = "ready"
    error: Text | None = None

    @model_validator(mode="after")
    def failure_reason(self):
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("only a failed refresh must include an error")
        return self


def _check_cycles(edges: dict[str, list[str]], label: str) -> None:
    # Iterative topological traversal also handles deep maps without recursion limits.
    remaining = {key: len(deps) for key, deps in edges.items()}
    dependents: dict[str, list[str]] = {key: [] for key in edges}
    for key, deps in edges.items():
        for dep in deps:
            dependents[dep].append(key)
    ready = [key for key, count in remaining.items() if count == 0]
    visited = 0
    while ready:
        visited += 1
        for dependent in dependents[ready.pop()]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
    if visited != len(edges):
        raise ValueError(f"{label} must not contain cycles")


class KnowledgeMap(Contract):
    schema_version: Literal[1]
    repository_id: Identity
    generated: KnowledgeLayer = Field(default_factory=KnowledgeLayer)
    curated: KnowledgeLayer = Field(default_factory=KnowledgeLayer)
    refresh: RefreshState = Field(default_factory=RefreshState)

    @property
    def items(self) -> list[KnowledgeItem]:
        return [*self.generated.items, *self.curated.items]

    @model_validator(mode="after")
    def valid_graph(self):
        for field in ("components", "items", "gaps", "conflicts"):
            entries = [*getattr(self.generated, field), *getattr(self.curated, field)]
            ids = [entry.id for entry in entries]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{field} IDs must be unique across generated and curated layers")
        for layer, origin in ((self.generated, "generated"), (self.curated, "maintainer")):
            if any(item.origin != origin for item in layer.items):
                raise ValueError(f"{origin} layer contains an item with the wrong origin")
        components = {entry.id: entry for entry in [*self.generated.components, *self.curated.components]}
        items = {item.id: item for item in self.items}
        gaps = {gap.id for layer in (self.generated, self.curated) for gap in layer.gaps}
        for component in components.values():
            if component.parent_id is not None and component.parent_id not in components:
                raise ValueError("unknown parent component")
        _check_cycles({key: [entry.parent_id] if entry.parent_id else []
                       for key, entry in components.items()}, "component hierarchy")
        for item in items.values():
            if set(item.component_ids) - components.keys():
                raise ValueError("unknown component reference")
            if any(dep.item_id not in items for dep in item.dependencies):
                raise ValueError("unknown item dependency")
            if set(item.addresses_gap_ids) - gaps:
                raise ValueError("unknown gap reference")
        _check_cycles({key: [dep.item_id for dep in item.dependencies] for key, item in items.items()},
                      "item evidence dependencies")
        for layer in (self.generated, self.curated):
            for gap in layer.gaps:
                if set(gap.component_ids) - components.keys() or set(gap.resolved_by) - items.keys():
                    raise ValueError("gap references unknown components or items")
            for conflict in layer.conflicts:
                if set(conflict.item_ids) - items.keys():
                    raise ValueError("conflict references unknown items")
        return self


def review_item(item: KnowledgeItem, *, reviewer: str,
                status: Literal["accepted", "rejected"], note: str | None = None) -> KnowledgeItem:
    """Explicitly bind a human/maintainer review to one complete claim version."""
    data = item.model_dump(mode="json")
    data["review"] = ReviewRecord(status=status, reviewer=reviewer,
                                 content_digest=item.content_digest, note=note).model_dump(mode="json")
    return KnowledgeItem.model_validate(data)

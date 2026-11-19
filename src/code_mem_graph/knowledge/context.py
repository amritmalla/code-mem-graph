"""Explainable task selection over recorded knowledge and eligible source paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Callable, Literal

from pydantic import Field

from code_mem_graph.serialization import fingerprint

from .build import load_repository_map
from .compact import is_navigation_fact
from .bundle import Scope, pack_bundle
from .models import Identity, SourcePath, Text
from .orient import OrientRequest
from .freshness import assess_map
from .relationships import is_import_record
from .sources import (
    EvidenceUnavailable,
    capture_file,
    capture_directory,
    discover_sources,
)
from .scoping import (
    ApplicationBoundaryError,
    InstructionIndexIncomplete,
    applicable_instructions,
    beneath,
    instruction_directory,
    instruction_paths,
    manifest_paths,
    peer_item,
    peer_paths,
)

_STOP = set(
    "a an the and or to of in on for with from by as is are be this that it its "
    "please add change update implement fix work code project repository new how where should can".split()
)


def terms(text: str) -> set[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return set(re.findall(r"[^\W_]+", text.casefold())) - _STOP


def _test_path(path: str) -> bool:
    parts = Path(path).parts
    return (
        bool({"tests", "test", "__tests__"}.intersection(parts))
        or Path(path).name.startswith("test_")
        or re.search(r"\.(?:test|spec)\.[cm]?[jt]sx?$", Path(path).name, re.I)
        is not None
    )


class ContextRequest(OrientRequest):
    task: Text
    application: SourcePath | None = Field(
        default=None,
        description="Eligible manifest directory relative to the repository, or . for the whole repository. Filters candidates and independent peer-package claims; required evidence is retained.",
    )
    paths: list[SourcePath] = Field(default_factory=list)
    component_ids: list[Identity] = Field(default_factory=list)
    item_ids: list[Identity] = Field(default_factory=list)


def task_context(
    root: Path,
    request: ContextRequest,
    *,
    transport: Literal["cli", "mcp"] = "cli",
    focused: bool = False,
    observe: Callable[[dict], None] | None = None,
) -> dict:
    document = load_repository_map(root)
    items = {item.id: item for item in document.items}
    components = {
        entry.id: entry
        for layer in (document.generated, document.curated)
        for entry in layer.components
    }
    if set(request.item_ids) - items.keys():
        raise ValueError("unknown requested knowledge item ID")
    if set(request.component_ids) - components.keys():
        raise ValueError("unknown requested component ID")
    query = set() if focused else terms(request.task)
    inventory_error = None
    try:
        inventory = discover_sources(root)
    except EvidenceUnavailable as exc:
        inventory = []
        inventory_error = exc.reason

    package_roots = {
        str(PurePosixPath(path).parent) for path in manifest_paths(inventory)
    }
    application = request.application
    if application is not None:
        if inventory_error:
            raise ApplicationBoundaryError(
                "application boundary cannot be checked; inspect source access and refresh knowledge"
            )
        if not any(
            component.lifecycle == "active" and application in component.paths
            for component in components.values()
        ):
            raise ApplicationBoundaryError(
                "unknown application directory; refresh knowledge or choose an active component path"
            )
        if application != "." and application not in package_roots:
            raise ApplicationBoundaryError(
                "application must name an eligible manifest directory; refresh knowledge or select the repository root"
            )
        capture_directory(root, application, recursive=True)
        if any(not beneath(path, application) for path in request.paths):
            raise ApplicationBoundaryError(
                "path hint is outside the application boundary"
            )
        if any(
            peer_item(items[key], components, application, package_roots)
            for key in request.item_ids
        ):
            raise ApplicationBoundaryError(
                "requested knowledge item belongs to another application"
            )
        if any(
            peer_paths(components[key].paths, application, package_roots)
            for key in request.component_ids
        ):
            raise ApplicationBoundaryError(
                "requested component belongs to another application"
            )

    matches = []
    for path in [] if focused else inventory:
        if application is not None and not beneath(path, application):
            continue
        explicit = any(beneath(path, hint) for hint in request.paths)
        overlap = terms(path) & query
        if explicit or overlap:
            matches.append(
                (
                    not explicit,
                    -len(terms(Path(path).stem) & query),
                    -len(overlap),
                    path,
                )
            )
    candidates = []
    for explicit_rank, _, _, path in sorted(matches):
        reason = "explicit_path" if not explicit_rank else "filename_or_path_overlap"
        try:
            reference = capture_file(root, path).model_dump(mode="json")
            candidates.append(
                {
                    "source": reference,
                    "basis": "path_match",
                    "reason": reason,
                    "role": "test_candidate"
                    if _test_path(path)
                    else "source_candidate",
                    "status": "readable",
                }
            )
        except EvidenceUnavailable as exc:
            candidates.append(
                {
                    "path": path,
                    "basis": "path_match",
                    "reason": reason,
                    "status": "unavailable",
                    "error": exc.reason,
                }
            )
    selected_components = set(request.component_ids)
    relationship_locations = request.paths or [entry[-1] for entry in matches]

    def touches(location):
        return any(
            beneath(location, target) or beneath(target, location)
            for target in relationship_locations
        )

    for path in [*request.paths, *(entry[-1] for entry in matches)]:
        covering = [
            (len(prefix), component.id)
            for component in components.values()
            for prefix in component.paths
            if prefix != "." and beneath(path, prefix)
        ]
        if covering:
            # Overlapping declared scopes can carry distinct mandatory policies.
            selected_components.update(key for _, key in covering)
    pending = list(selected_components)
    while pending:
        parent = components[pending.pop()].parent_id
        if parent is not None and parent not in selected_components:
            selected_components.add(parent)
            pending.append(parent)
    scoped = {key for key in selected_components if components[key].paths != ["."]}
    reasons, ranked, global_ids = {}, [], []
    for key, item in items.items():
        why = []
        explicit = key in request.item_ids
        if peer_item(item, components, application, package_roots):
            continue
        if (
            item.evidence
            and item.evidence[0].kind == "file"
            and instruction_paths([item.evidence[0].path])
            and not item.command_checks
            and not explicit
        ):
            continue
        if is_import_record(item):
            if explicit or (
                not focused
                and item.lifecycle == "active"
                and any(
                    touches(edge.source) or touches(edge.target)
                    for edge in item.relationships
                )
            ):
                reasons[key] = [
                    "explicit_item_id" if explicit else "static_import_location_match"
                ]
                ranked.append((not explicit, True, True, 0, key))
            continue
        if (
            application is not None
            and not explicit
            and is_navigation_fact(item)
            and any(
                prefix != "." and prefix != application and beneath(application, prefix)
                for cid in item.component_ids
                for prefix in components[cid].paths
            )
        ):
            continue
        if explicit:
            why.append("explicit_item_id")
        if item.lifecycle == "active":
            if not focused and any(touches(edge.target) for edge in item.relationships):
                why.append("documented_location_match")
            if any(
                beneath(ref.path, hint)
                or (ref.kind == "directory" and beneath(hint, ref.path))
                for ref in item.evidence
                for hint in request.paths
                if ref.path != "."
            ):
                why.append("path_hint_evidence")
            if scoped.intersection(item.component_ids) or set(
                request.component_ids
            ).intersection(item.component_ids):
                why.append("component_scope")
            overlap = query & terms(
                " ".join(
                    [
                        key,
                        item.title,
                        item.statement,
                        *(ref.path for ref in item.evidence),
                    ]
                )
            )
            if overlap:
                why.append("recorded_text_overlap")
            global_item = not item.component_ids or any(
                components[cid].paths == ["."] for cid in item.component_ids
            )
            if global_item and (
                item.critical
                or (
                    not focused
                    and (
                        "constraints" in item.facets
                        or item.topic in {"conventions", "workflow"}
                    )
                )
            ):
                global_ids.append(key)
                reasons[key] = ["repository_constraint_or_workflow"]
        if why:
            reasons[key] = why
            ranked.append(
                (
                    not explicit,
                    "path_hint_evidence" not in why,
                    "component_scope" not in why,
                    -len(query & terms(item.title + " " + item.statement)),
                    key,
                )
            )
    seeds = [entry[-1] for entry in sorted(ranked)]
    selected = set(seeds)
    # A task touching evidence also needs explanations/decisions derived from that evidence.
    pending = [] if focused else list(seeds)
    dependents: dict[str, list[str]] = {key: [] for key in items}
    for item in items.values():
        for dependency in item.dependencies:
            dependents[dependency.item_id].append(item.id)
    while pending:
        for key in dependents[pending.pop()]:
            if (
                key not in selected
                and items[key].lifecycle == "active"
                and not is_import_record(items[key])
                and not peer_item(items[key], components, application, package_roots)
            ):
                selected.add(key)
                pending.append(key)
                reasons[key] = ["depends_on_task_matched_knowledge"]
    order = list(
        dict.fromkeys([*seeds, *(key for key in items if key in selected), *global_ids])
    )
    # One incident-record step from the original locations/selected guidance.
    # Newly discovered endpoints never recursively flood the rest of the graph.
    relationship_items = [key for key in order if items[key].relationships]
    if relationship_items and not focused:
        assessments = {entry.item_id: entry for entry in assess_map(document, root)}
        candidate_by_path = {
            row.get("source", {}).get("path", row.get("path")): row
            for row in candidates
        }
        for key in relationship_items:
            assessment = assessments[key]
            if assessment.freshness != "fresh" or assessment.review_required:
                continue
            for edge in items[key].relationships:
                for path in (edge.source, edge.target):
                    if path not in inventory or (
                        application is not None and not beneath(path, application)
                    ):
                        continue
                    if path not in candidate_by_path:
                        try:
                            candidate_by_path[path] = {
                                "source": capture_file(root, path).model_dump(
                                    mode="json"
                                ),
                                "basis": "evidenced_relationship",
                                "reason": "recorded_location_relationship",
                                "role": "test_candidate"
                                if _test_path(path)
                                else "source_candidate",
                                "status": "readable",
                            }
                            candidates.append(candidate_by_path[path])
                        except EvidenceUnavailable:
                            continue
                    candidate_by_path[path].setdefault(
                        "relationship_item_ids", []
                    ).append(key)
        for row in candidates:
            if "relationship_item_ids" in row:
                row["relationship_item_ids"] = sorted(set(row["relationship_item_ids"]))
    links = {key: set() for key in items}
    for layer in (document.generated, document.curated):
        for conflict in layer.conflicts:
            for key in conflict.item_ids:
                links[key].update(conflict.item_ids)

    def proof_closure(keys):
        found = set()
        pending = list(keys)
        while pending:
            key = pending.pop()
            if key in found:
                continue
            found.add(key)
            pending.extend(dep.item_id for dep in items[key].dependencies)
            pending.extend(links[key])
        return found

    indexed = {
        item.evidence[0].path: key
        for key, item in items.items()
        if item.lifecycle == "active" and instruction_directory(item) is not None
    }
    required_instructions = []
    while True:
        proof = proof_closure(order)
        targets = {
            application or ".",
            *request.paths,
            *(entry[-1] for entry in matches),
        }
        for key in proof:
            if instruction_directory(items[key]) is None and not is_navigation_fact(
                items[key]
            ):
                targets.update(
                    reference.path
                    for reference in items[key].evidence
                    if reference.kind == "file"
                )
        broad = {
            path
            for path in request.paths
            if any(candidate.startswith(path + "/") for candidate in inventory)
        }
        for key in request.component_ids:
            for path in components[key].paths:
                broad.add(
                    application if application and beneath(application, path) else path
                )
        live_required = applicable_instructions(
            instruction_paths(inventory), targets, broad
        )
        missing = set(live_required) - indexed.keys()
        if missing:
            raise InstructionIndexIncomplete(
                "Applicable directory instructions are not indexed; run knowledge refresh before requesting context: "
                + ", ".join(sorted(missing))
            )
        required_paths = applicable_instructions(list(indexed), targets, broad)
        required_instructions = [indexed[path] for path in sorted(required_paths)]
        constraint_components = {
            cid for key in proof for cid in items[key].component_ids
        }
        if application is not None:
            constraint_components.update(
                key
                for key, component in components.items()
                if application in component.paths
            )
        pending = list(constraint_components)
        while pending:
            parent = components[pending.pop()].parent_id
            if parent is not None and parent not in constraint_components:
                constraint_components.add(parent)
                pending.append(parent)
        constraints = [
            key
            for key, item in items.items()
            if item.lifecycle == "active"
            and item.critical
            and instruction_directory(item) is None
            and (
                not item.component_ids
                or constraint_components.intersection(item.component_ids)
            )
        ]
        extra = [
            key for key in [*constraints, *required_instructions] if key not in order
        ]
        if not extra:
            break
        for key in extra:
            reasons[key] = [
                "applicable_directory_instruction"
                if key in required_instructions
                else "applicable_component_constraint"
            ]
        order.extend(extra)
    # Match instruction prose only after its directory applicability is known.
    for key in required_instructions:
        if key not in seeds and query & terms(
            items[key].title + " " + items[key].statement
        ):
            seeds.append(key)
    metadata = {
        "selector": "recorded-text-and-paths-v3",
        "calibrated": False,
        "task_digest": fingerprint(request.task),
        "matched_claims": len(seeds),
        "matched_components": len(scoped),
        "candidate_locations": len(candidates),
        "inventory_status": "not_requested"
        if focused
        else "unknown"
        if inventory_error
        else "checked",
        "inventory_reason": inventory_error,
        "unavailable_path_hints": sum(
            not any(beneath(path, hint) for path in inventory) for hint in request.paths
        ),
        "dependency_kind": "knowledge_evidence_not_runtime_call_graph",
    }
    if relationship_items:
        metadata.update(
            relationship_selection="one_incident_record_step_from_original_locations_and_selected_guidance",
            relationship_item_ids=relationship_items,
            relationship_limits="documented locations and supported static imports; no runtime binding, causal change chain or test-coverage claim",
        )
    if application is not None:
        metadata.update(
            application=application,
            boundary="source_candidates_and_independent_claims",
            cross_boundary_proof_ids=sorted(
                key
                for key in proof
                if peer_item(items[key], components, application, package_roots)
            ),
            cross_boundary_relationship_targets=sorted(
                {
                    edge.target
                    for key in relationship_items
                    for edge in items[key].relationships
                    if not beneath(edge.target, application)
                }
            ),
        )
    if required_instructions or inventory_error:
        metadata.update(
            instruction_discovery="unknown"
            if inventory_error
            else "eligible_catalog_checked",
            required_instruction_ids=required_instructions,
        )
    scope = Scope(
        item_ids=order,
        component_ids=[key for key in components if key in selected_components],
        required_facets=() if focused else ("constraints", "workflow"),
        reasons=reasons,
        candidates=candidates,
        seed_ids=seeds,
        metadata=metadata,
        mandatory_ids=required_instructions,
        unresolved=not bool(seeds) or inventory_error is not None,
        focus_ids=[
            key
            for key in order
            if not is_navigation_fact(items[key]) or key in request.item_ids
        ],
    )
    return pack_bundle(
        document,
        root,
        budget=request.budget,
        encoding=request.encoding,
        transport=transport,
        scope=scope,
        observe=observe,
        representation=request.representation,
        detail=request.detail,
    )

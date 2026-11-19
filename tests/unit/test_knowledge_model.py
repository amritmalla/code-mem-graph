from copy import deepcopy
import json
from pathlib import Path
import stat
from types import SimpleNamespace

from click.testing import CliRunner
from pydantic import ValidationError
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.freshness import assess_map
from code_mem_graph.knowledge.io import load_map
from code_mem_graph.knowledge.models import (
    Component, FileEvidence, Generator, ItemDependency, KnowledgeConflict, KnowledgeGap,
    KnowledgeItem, KnowledgeLayer, KnowledgeMap, ReviewRecord, portable_path, review_item,
)
from code_mem_graph.knowledge.refresh import refresh_generated
from code_mem_graph.knowledge.sources import (
    EvidenceUnavailable, capture_directory, capture_file, evidence_policy, policy_digest,
)


@pytest.fixture
def source_root(tmp_path):
    (tmp_path / "app.py").write_text("def health():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("A fictional service.\n", encoding="utf-8")
    return tmp_path


def claim(root, *, item_id="service/health", basis="observed"):
    return KnowledgeItem(
        id=item_id, topic="architecture", title="Health function", statement="app.py implements health.",
        origin="generated", basis=basis, generator=Generator(name="fixture", version="1"),
        evidence=[capture_file(root, "app.py")],
    )


def map_for(*items, curated=None, components=None, conflicts=None):
    return KnowledgeMap(schema_version=1, repository_id="test-project",
                        generated=KnowledgeLayer(items=list(items), components=components or [],
                                                 conflicts=conflicts or []),
                        curated=curated or KnowledgeLayer())


def child_of(parent):
    return KnowledgeItem(id="architecture/boundary", topic="architecture", title="Boundary",
                         statement="Health is the service boundary.", origin="generated", basis="inferred",
                         generator=Generator(name="fixture", version="1"),
                         dependencies=[ItemDependency(item_id=parent.id, content_digest=parent.content_digest)])


@pytest.mark.parametrize("path", [
    "", "/etc/passwd", "../app.py", "src/../app.py", "src//app.py", "./app.py",
    "C:/app.py", "C:app.py", "src\\app.py", "//server/share", "src/CON.py",
    "src/NUL", "src/app.py.", "src /app.py ", "src/a:stream.py", "a\n.py",
])
def test_nonportable_or_escaping_paths_rejected(path):
    with pytest.raises(ValueError):
        portable_path(path)


@pytest.mark.parametrize("path", ["README.md", "src/parcel/app.py", ".github/workflows/test.yml", "src/नमस्ते.py", "."])
def test_canonical_relative_paths_accepted(path):
    assert portable_path(path) == path


def test_file_ranges_and_whole_file_hashing(source_root):
    evidence = capture_file(source_root, "app.py", start_line=1, end_line=1)
    assert len(evidence.digest) == 64 and evidence.policy_digest == policy_digest()
    with pytest.raises(EvidenceUnavailable, match="invalid_line_range"):
        capture_file(source_root, "app.py", start_line=1, end_line=3)
    with pytest.raises(ValidationError, match="both"):
        FileEvidence(path="app.py", digest=evidence.digest, policy_digest=evidence.policy_digest, start_line=1)
    with pytest.raises(ValidationError, match="precedes"):
        capture_file(source_root, "app.py", start_line=2, end_line=1)
    (source_root / "app.py").write_text("def health():\n    return 'changed'\n", encoding="utf-8")
    assert capture_file(source_root, "app.py", start_line=1, end_line=1).digest != evidence.digest


@pytest.mark.parametrize("path", [".env", ".env.example.md", ".aws/config", ".git/config", ".mcp.json",
                                  "settings.yaml", "secrets.json", "node_modules/x/app.py",
                                  ".project-knowledge/brief.md", "dist/app.py"])
def test_excluded_files_not_read(source_root, path, monkeypatch):
    def fail_read(*args, **kwargs):
        raise AssertionError("excluded content was read")
    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(EvidenceUnavailable, match="excluded"):
        capture_file(source_root, path)


def test_membership_policy_and_recursive_scope(source_root):
    (source_root / "src").mkdir()
    (source_root / "src/a.ts").write_text("export const a = 1;", encoding="utf-8")
    (source_root / "node_modules").mkdir()
    (source_root / "node_modules/ignored.py").write_text("ignore", encoding="utf-8")
    direct = capture_directory(source_root, ".")
    recursive = capture_directory(source_root, ".", recursive=True)
    (source_root / "src/b.ts").write_text("export const b = 2;", encoding="utf-8")
    assert capture_directory(source_root, ".").digest == direct.digest
    assert capture_directory(source_root, ".", recursive=True).digest != recursive.digest
    current = capture_directory(source_root, ".", recursive=True)
    (source_root / "src/a.ts").write_text("changed content", encoding="utf-8")
    (source_root / ".env").write_text("SECRET=fixture", encoding="utf-8")
    (source_root / "empty").mkdir()
    assert capture_directory(source_root, ".", recursive=True).digest == current.digest
    (source_root / "src/a.ts").rename(source_root / "src/renamed.ts")
    assert capture_directory(source_root, ".", recursive=True).digest != current.digest


@pytest.mark.parametrize("path", ["pyproject.toml", "web/package.json", "web/tsconfig.json", "Cargo.toml",
                                  "go.mod", "pom.xml", ".github/workflows/test.yaml"])
def test_allowlisted_manifests_and_workflows(source_root, path):
    file = source_root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("fixture", encoding="utf-8")
    assert capture_file(source_root, path).path == path


def test_reparse_points_rejected_before_reading(source_root, monkeypatch):
    original = Path.lstat
    target = source_root / "app.py"
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024, raising=False)

    def metadata(path, *args, **kwargs):
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=1024)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail("linked content read"))
    with pytest.raises(EvidenceUnavailable, match="unsafe_link"):
        capture_file(source_root, "app.py")
    with pytest.raises(EvidenceUnavailable, match="unsafe_link"):
        capture_directory(source_root, ".", recursive=True)


def test_symlink_outside_root_is_not_evidence(source_root, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("private fixture", encoding="utf-8")
    try:
        (source_root / "linked.py").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks requires platform privileges")
    with pytest.raises(EvidenceUnavailable, match="unsafe_link"):
        capture_file(source_root, "linked.py")


def test_source_edits_deletion_and_policy_change(source_root):
    item = claim(source_root)
    document = map_for(item)
    assert assess_map(document, source_root)[0].freshness == "fresh"
    (source_root / "app.py").write_text("# working tree edit", encoding="utf-8")
    result = assess_map(document, source_root)[0]
    assert result.freshness == "stale" and "source_changed:app.py" in result.reasons
    (source_root / "app.py").unlink()
    assert "missing:app.py" in assess_map(document, source_root)[0].reasons
    data = item.model_dump(mode="json")
    data["evidence"][0]["policy_digest"] = "0" * 64
    assert "evidence_policy_changed:app.py" in assess_map(map_for(KnowledgeItem.model_validate(data)), source_root)[0].reasons


def test_membership_additions_and_deletions_invalidate(source_root):
    data = claim(source_root).model_dump(mode="json")
    data["evidence"] = [capture_directory(source_root, ".").model_dump(mode="json")]
    document = map_for(KnowledgeItem.model_validate(data))
    assert assess_map(document, source_root)[0].freshness == "fresh"
    (source_root / "new.py").write_text("# new", encoding="utf-8")
    assert assess_map(document, source_root)[0].freshness == "stale"
    (source_root / "new.py").unlink()
    assert assess_map(document, source_root)[0].freshness == "fresh"
    (source_root / "app.py").unlink()
    assert assess_map(document, source_root)[0].freshness == "stale"


def test_unreadable_evidence_unknown_and_known_changes_dominate(source_root, monkeypatch):
    item = claim(source_root)
    data = item.model_dump(mode="json")
    data["evidence"].append(capture_file(source_root, "README.md").model_dump(mode="json"))
    document = map_for(KnowledgeItem.model_validate(data))
    original = Path.read_bytes

    def read(path):
        if path.name == "README.md":
            raise PermissionError("fixture denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert assess_map(document, source_root)[0].freshness == "unknown"
    (source_root / "app.py").write_text("# changed", encoding="utf-8")
    assert assess_map(document, source_root)[0].freshness == "stale"


def test_unreadable_directory_never_returns_partial_digest(source_root, monkeypatch):
    item = claim(source_root).model_dump(mode="json")
    item["evidence"] = [capture_directory(source_root, ".", recursive=True).model_dump(mode="json")]
    document = map_for(KnowledgeItem.model_validate(item))
    original = Path.iterdir

    def interrupted(path):
        yield next(original(path))
        raise PermissionError("partial scan")

    monkeypatch.setattr(Path, "iterdir", interrupted)
    assert assess_map(document, source_root)[0].freshness == "unknown"


def test_dependency_versions_staleness_and_review_propagate(source_root):
    parent = claim(source_root)
    child = review_item(child_of(parent), reviewer="fixture-reviewer", status="accepted")
    grandchild_data = child_of(child).model_dump(mode="json")
    grandchild_data["id"] = "architecture/overview"
    grandchild = KnowledgeItem.model_validate(grandchild_data)
    document = map_for(parent, child, grandchild)
    results = assess_map(document, source_root)
    assert [result.freshness for result in results] == ["fresh"] * 3
    assert results[1].review_required is False
    (source_root / "app.py").write_text("# changed", encoding="utf-8")
    assert all(result.freshness == "stale" for result in assess_map(document, source_root))
    parent_data = parent.model_dump(mode="json")
    parent_data["statement"] = "A revised claim using the same evidence."
    updated = KnowledgeItem.model_validate(parent_data)
    results = assess_map(map_for(updated, child), source_root)
    assert f"dependency_version_changed:{parent.id}" in results[1].reasons


def test_review_rejection_is_independent_of_freshness(source_root):
    parent = review_item(claim(source_root), reviewer="fixture", status="rejected")
    child = review_item(child_of(parent), reviewer="fixture", status="accepted")
    results = assess_map(map_for(parent, child), source_root)
    assert all(result.freshness == "fresh" for result in results)
    assert all(result.review_required for result in results)


@pytest.mark.parametrize("field,value", [
    ("statement", "Changed claim."), ("title", "Changed title"), ("critical", True),
    ("generator", {"name": "fixture", "version": "2"}),
])
def test_review_is_bound_to_complete_claim_version(source_root, field, value):
    item = review_item(claim(source_root), reviewer="fixture", status="accepted")
    data = item.model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError, match="claim version"):
        KnowledgeItem.model_validate(data)
    assert item.review.status == "accepted"


def test_no_evidence_cannot_masquerade_as_fact(source_root):
    data = claim(source_root).model_dump(mode="json")
    data["evidence"] = []
    for basis in ("observed", "documented", "inferred"):
        data["basis"] = basis
        with pytest.raises(ValidationError):
            KnowledgeItem.model_validate(data)
    assertion = KnowledgeItem(id="decision/intent", topic="decision", title="Intent", statement="Maintainer intent.",
                              origin="maintainer", basis="maintainer", author="fixture")
    assertion = review_item(assertion, reviewer="fixture", status="accepted")
    result = assess_map(map_for(curated=KnowledgeLayer(items=[assertion])), source_root)[0]
    assert result.freshness == "unknown"
    assert "no_verifiable_source" in result.reasons
    assert assertion.review.status == "accepted"


@pytest.mark.parametrize("change", [
    lambda data: data["generated"]["items"].append(deepcopy(data["generated"]["items"][0])),
    lambda data: data["generated"]["items"][0].update(component_ids=["missing"]),
    lambda data: data["generated"]["items"][0].update(dependencies=[{"item_id": "missing", "content_digest": "0" * 64}]),
    lambda data: data["curated"]["items"].append(data["generated"]["items"].pop()),
    lambda data: data.update(extra="invalid"),
])
def test_graph_references_and_ownership_validated(source_root, change):
    data = map_for(claim(source_root)).model_dump(mode="json")
    change(data)
    with pytest.raises(ValidationError):
        KnowledgeMap.model_validate(data)


def test_cycles_and_dangling_conflicts_are_rejected(source_root):
    parent = claim(source_root)
    child = child_of(parent)
    parent_data = parent.model_dump(mode="json")
    parent_data["dependencies"] = [{"item_id": child.id, "content_digest": child.content_digest}]
    with pytest.raises(ValidationError, match="cycles"):
        map_for(KnowledgeItem.model_validate(parent_data), child)
    with pytest.raises(ValidationError, match="cycles"):
        map_for(parent, components=[Component(id="a", name="A", paths=["src"], parent_id="b"),
                                    Component(id="b", name="B", paths=["src"], parent_id="a")])
    with pytest.raises(ValidationError, match="unknown items"):
        map_for(parent, conflicts=[KnowledgeConflict(id="conflict/a", item_ids=[parent.id, "missing"], description="Conflict")])


def test_unresolved_and_expired_conflict_resolutions(source_root):
    left, right = claim(source_root), claim(source_root, item_id="service/alternative")
    conflict = KnowledgeConflict(id="conflict/runtime", item_ids=[left.id, right.id], description="Competing claims")
    results = assess_map(map_for(left, right, conflicts=[conflict]), source_root)
    assert all(result.freshness == "fresh" and result.review_required for result in results)
    assert all(result.conflict_ids == [conflict.id] for result in results)
    data = conflict.model_dump(mode="json")
    data.update(status="resolved", resolution="Fixture resolution", resolved_by="fixture",
                resolved_versions={item.id: item.content_digest for item in [left, right]})
    resolved = KnowledgeConflict.model_validate(data)
    assert not any(result.conflict_ids for result in assess_map(map_for(left, right, conflicts=[resolved]), source_root))
    changed = right.model_dump(mode="json")
    changed["statement"] = "Revised competing claim"
    assert all(result.conflict_ids for result in assess_map(
        map_for(left, KnowledgeItem.model_validate(changed), conflicts=[resolved]), source_root))
    (source_root / "app.py").unlink()
    assert all(result.conflict_ids for result in assess_map(map_for(left, right, conflicts=[resolved]), source_root))


def test_refresh_preserves_curated_data_and_unchanged_reviews(source_root):
    prior = review_item(claim(source_root), reviewer="fixture", status="accepted")
    authored = KnowledgeItem(id="decision/reason", topic="decision", title="Reason", statement="Maintainer explanation.",
                            origin="maintainer", basis="maintainer", author="fixture",
                            dependencies=[ItemDependency(item_id=prior.id, content_digest=prior.content_digest)])
    curated = KnowledgeLayer(items=[review_item(authored, reviewer="fixture", status="accepted")],
                             gaps=[KnowledgeGap(id="workflow/approvers", topic="workflow", description="Unknown approvers")])
    current = map_for(prior, curated=curated)
    before = current.model_dump(mode="json")
    incoming = KnowledgeLayer(items=[claim(source_root)])
    merged = refresh_generated(current, incoming)
    assert merged.curated == current.curated
    assert merged.generated.items[0].review == prior.review
    assert current.model_dump(mode="json") == before
    merged.curated.items.clear()  # Caller mutation of a returned container must not modify the original.
    assert current.model_dump(mode="json") == before


def test_refresh_resets_changed_reviews_and_rejects_generator_review_authority(source_root):
    prior = review_item(claim(source_root), reviewer="fixture", status="accepted")
    changed = prior.model_dump(mode="json")
    changed.update(statement="Revised claim", review=ReviewRecord().model_dump())
    new = review_item(KnowledgeItem.model_validate(changed), reviewer="generator-pretending-to-review", status="accepted")
    merged = refresh_generated(map_for(prior), KnowledgeLayer(items=[new]))
    assert merged.generated.items[0].review.status == "unreviewed"
    initial = refresh_generated(map_for(), KnowledgeLayer(items=[prior]))
    assert initial.generated.items[0].review.status == "unreviewed"


def test_retirement_preserves_links_and_explicit_gaps_conflicts(source_root):
    parent = claim(source_root)
    other = claim(source_root, item_id="service/alternative")
    dependent = child_of(parent).model_dump(mode="json")
    dependent.update(id="decision/explanation", origin="maintainer", author="fixture", generator=None)
    curated = KnowledgeLayer(items=[KnowledgeItem.model_validate(dependent)])
    current = map_for(parent, other, curated=curated,
                      conflicts=[KnowledgeConflict(id="conflict/health", item_ids=[parent.id, other.id], description="Conflict")])
    current_data = current.model_dump(mode="json")
    current_data["generated"]["gaps"] = [KnowledgeGap(id="workflow/unknown", topic="workflow", description="Unknown").model_dump()]
    current = KnowledgeMap.model_validate(current_data)
    merged = refresh_generated(current, KnowledgeLayer())
    assert all(item.lifecycle == "retired" for item in merged.generated.items)
    assert merged.generated.gaps == current.generated.gaps
    assert merged.generated.conflicts == current.generated.conflicts
    assert merged.curated == curated
    assert all(result.freshness == "stale" for result in assess_map(merged, source_root))


def test_failed_refresh_keeps_prior_fingerprints_and_cannot_certify_freshness(source_root):
    parent = claim(source_root)
    current = map_for(parent, child_of(parent))
    failed = refresh_generated(current, None, error="Fixture parser failed")
    assert failed.generated == current.generated
    assert failed.refresh.status == "failed"
    assert all(result.freshness == "unknown" for result in assess_map(failed, source_root))
    (source_root / "app.py").write_text("# changed", encoding="utf-8")
    assert all(result.freshness == "stale" for result in assess_map(failed, source_root))
    recovered = refresh_generated(failed, current.generated)
    assert recovered.refresh.status == "ready"
    assert assess_map(recovered, source_root)[0].freshness == "stale"


def test_bad_refresh_rejected_without_mutating_original(source_root):
    authored = KnowledgeItem(id="decision/intent", topic="decision", title="Intent", statement="Maintainer statement.",
                            origin="maintainer", basis="maintainer", author="fixture")
    current = map_for(curated=KnowledgeLayer(items=[authored]))
    before = current.model_dump(mode="json")
    with pytest.raises(ValidationError, match="unique"):
        refresh_generated(current, KnowledgeLayer(items=[claim(source_root, item_id=authored.id)]))
    assert current.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="reason"):
        refresh_generated(current, None)
    with pytest.raises(ValueError, match="successful"):
        refresh_generated(current, KnowledgeLayer(), error="failure")


def test_cli_schema_map_validation_and_stale_reporting(source_root, tmp_path):
    runner = CliRunner()
    schema = runner.invoke(main, ["knowledge", "schema"])
    assert schema.exit_code == 0, schema.output
    assert json.loads(schema.output)["properties"]["schema_version"]["const"] == 1
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(map_for(claim(source_root)).model_dump(mode="json")), encoding="utf-8")
    args = ["knowledge", "validate", str(path)]
    assert json.loads(runner.invoke(main, args).output)["assessments"] is None
    result = runner.invoke(main, args + ["--root", str(source_root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["assessments"][0]["freshness"] == "fresh"
    (source_root / "app.py").unlink()
    stale = runner.invoke(main, args + ["--root", str(source_root)])
    assert stale.exit_code == 0  # Stale knowledge is valid data, explicitly labeled.
    assert json.loads(stale.output)["assessments"][0]["freshness"] == "stale"
    path.write_text("schema_version: 999", encoding="utf-8")
    assert runner.invoke(main, args).exit_code != 0


@pytest.mark.parametrize("name", ["python_service", "typescript_app"])
def test_representative_fixture_roundtrip_and_current_sources(name):
    root = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge" / name
    document = load_map(root / "map.yaml")
    assert KnowledgeMap.model_validate_json(document.model_dump_json()) == document
    results = assess_map(document, root)
    assert all(result.freshness == "fresh" for result in results if result.item_id not in {"domain/parcel"})
    if name == "typescript_app":
        assert all(result.conflict_ids == ["workflow/node-runtime"] for result in results[:2])
        assert results[-1].freshness == "unknown"
    else:
        assert results[2].review_required is True
        assert document.generated.gaps[0].status == "open"
    assert evidence_policy()["version"] == "repository-evidence-v2"

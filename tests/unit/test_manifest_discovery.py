"""Nested manifest facts remain scoped, cited and distinct from fixture data."""

import json

import pytest
import yaml

from code_mem_graph.knowledge.build import build_repository, load_repository_map
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.freshness import assess_map
from code_mem_graph.knowledge.models import KnowledgeItem, review_item


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_colon_separated_node_script_names_are_retained_without_command_bodies(
    tmp_path,
):
    path = "frontend/package.json"
    write(
        tmp_path,
        path,
        json.dumps(
            {
                "scripts": {
                    "test:ui": "SECRET_SENTINEL=1 do-not-run",
                    "test:e2e": "external command",
                    "build:production": "another external command",
                }
            }
        ),
    )
    document = build_repository(tmp_path, repository_id="script-names")
    facts = manifest_items(document, path)
    assert {item.title for item in facts} == {
        "Declared script: test:ui",
        "Declared script: test:e2e",
        "Declared script: build:production",
    }
    assert all(item.component_ids == [scope(document, "frontend").id] for item in facts)
    assert "SECRET_SENTINEL" not in document.model_dump_json()


def scope(document, path):
    return next(
        component
        for component in document.generated.components
        if component.paths == [path]
    )


def manifest_items(document, path):
    return [
        item
        for item in document.generated.items
        if item.lifecycle == "active"
        and item.id.startswith("manifest/")
        and any(reference.path == path for reference in item.evidence)
    ]


@pytest.mark.parametrize("directory", ["backend", "services/payments/api"])
def test_nested_python_runtime_and_fastapi_entry_are_bound_to_manifest_directory(
    tmp_path, directory
):
    path = directory + "/pyproject.toml"
    write(
        tmp_path,
        path,
        '[project]\nname="payments"\nrequires-python=">=3.14,<4.0"\n'
        'description="Nested description is not the whole repository purpose"\n'
        'dependencies=["fastapi>=0.100", "private-lib @ https://name:secret@private.example/pkg"]\n'
        '[project.scripts]\nworker="payments.worker:main"\n'
        '[tool.pytest.ini_options]\ntestpaths=["tests"]\n'
        '[tool.fastapi]\nentrypoint="payments.main:app"\n',
    )
    document = build_repository(tmp_path, repository_id="nested-python")
    component = scope(document, directory)
    facts = manifest_items(document, path)
    assert facts and all(item.component_ids == [component.id] for item in facts)
    text = "\n".join(item.statement for item in facts)
    assert (
        ">=3.14,<4.0" in text and "tool.fastapi.entrypoint = payments.main:app" in text
    )
    assert "payments.worker:main" in text and "pytest configuration" in text
    assert "private-lib" in text and "name:secret" not in text
    assert not any(item.id == "project/purpose" for item in document.items)
    for item in facts:
        assert item.basis == "observed" and item.evidence[0].path == path
    if "/" in directory:
        assert component.parent_id == scope(document, directory.rsplit("/", 1)[0]).id
    else:
        assert component.parent_id == "project"
    assert build_repository(tmp_path) == document


def test_node_siblings_keep_runtime_scripts_entries_and_dependencies_in_separate_scopes(
    tmp_path,
):
    write(
        tmp_path,
        "package.json",
        json.dumps({"scripts": {"root-check": "do not execute"}}),
    )
    for directory, runtime, dependency in [
        ("apps/web", ">=20", "react"),
        ("apps/admin", ">=22", "admin-ui"),
    ]:
        write(
            tmp_path,
            directory + "/package.json",
            json.dumps(
                {
                    "engines": {"node": runtime},
                    "main": "src/main.ts",
                    "scripts": {"test": "SECRET_DO_NOT_COPY=1 run-tests"},
                    "dependencies": {
                        dependency: "https://name:secret@private.example/pkg"
                    },
                }
            ),
        )
    document = build_repository(tmp_path, repository_id="sibling-apps")
    for directory in ("apps/web", "apps/admin"):
        component = scope(document, directory)
        assert component.parent_id == scope(document, "apps").id
        facts = manifest_items(document, directory + "/package.json")
        assert len(facts) == 4
        assert all(item.component_ids == [component.id] for item in facts)
    assert all(
        item.component_ids == ["project"]
        for item in manifest_items(document, "package.json")
    )
    assert (
        "SECRET_DO_NOT_COPY" not in document.model_dump_json()
        and "name:secret" not in document.model_dump_json()
    )
    web = scope(document, "apps/web")
    before_id = web.id
    payload = task_context(
        tmp_path,
        ContextRequest(
            task="quuxxyz",
            paths=["apps/web/src/main.ts"],
            budget=20000,
            encoding="cl100k_base",
        ),
    )
    returned = [item for group in ("items", "cautions") for item in payload[group]]
    assert any(
        "apps/web/package.json" in [ref["path"] for ref in item["evidence"]]
        for item in returned
    )
    assert not any(
        "apps/admin/package.json" in [ref["path"] for ref in item["evidence"]]
        for item in returned
    )
    assert scope(build_repository(tmp_path), "apps/web").id == before_id


@pytest.mark.parametrize(
    "directory",
    [
        "tests/service",
        "apps/web/__tests__/fixture",
        "packages/core/fixtures/sample",
        "examples/demo",
        "products/SAMPLES/demo",
        "services/api/__mocks__/backend",
        "vendor/package",
    ],
)
def test_fixture_and_example_manifests_are_not_promoted_to_project_facts(
    tmp_path, directory
):
    # Invalid fixture manifests must not abort extraction of the actual project.
    write(tmp_path, directory + "/pyproject.toml", "[project\nnot valid TOML")
    write(tmp_path, directory + "/package.json", "not valid JSON")
    write(
        tmp_path,
        "services/live/pyproject.toml",
        '[project]\nname="live"\nrequires-python=">=3.11"\n',
    )
    document = build_repository(tmp_path, repository_id="fixture-exclusions")
    assert manifest_items(document, "services/live/pyproject.toml")
    assert not manifest_items(document, directory + "/pyproject.toml")
    assert not manifest_items(document, directory + "/package.json")


def test_nested_manifests_cannot_reopen_ignored_sources_or_expose_unsupported_entry_values(
    tmp_path,
):
    write(tmp_path, ".gitignore", "services/private/\n")
    write(tmp_path, ".knowledgeignore", "frontend/secret/\n")
    for directory in ("services/private", "frontend/secret"):
        write(tmp_path, directory + "/pyproject.toml", "[invalid")
    write(
        tmp_path,
        "services/public/pyproject.toml",
        '[project]\nname="public"\n'
        '[tool.fastapi]\nentrypoint="TOKEN_DO_NOT_COPY=$(execute)"\n',
    )
    document = build_repository(tmp_path, repository_id="ignore-boundaries")
    facts = manifest_items(document, "services/public/pyproject.toml")
    assert len(facts) == 1 and "TOKEN_DO_NOT_COPY" not in document.model_dump_json()
    assert not any(
        "private/pyproject" in reference.path or "secret/pyproject" in reference.path
        for item in document.items
        for reference in item.evidence
    )


def test_nested_manifest_add_edit_and_removal_invalidate_facts_and_preserve_maintainer_knowledge(
    tmp_path,
):
    write(tmp_path, "README.md", "A repository with a separately packaged service.\n")
    first = build_repository(tmp_path, repository_id="manifest-refresh")
    authored = review_item(
        KnowledgeItem(
            id="decision/ownership",
            topic="decision",
            title="Maintainer ownership",
            statement="The team owns this service.",
            origin="maintainer",
            basis="maintainer",
            author="maintainer",
        ),
        reviewer="maintainer",
        status="accepted",
    )
    data = first.model_dump(mode="json")
    data["curated"]["items"] = [authored.model_dump(mode="json")]
    (tmp_path / ".project-knowledge/map.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    path = "services/worker/pyproject.toml"
    write(tmp_path, path, '[project]\nname="worker"\nrequires-python=">=3.11"\n')
    assert any(
        entry.freshness == "stale"
        for entry in assess_map(load_repository_map(tmp_path), tmp_path)
    )
    added = build_repository(tmp_path)
    runtime = next(
        item for item in manifest_items(added, path) if item.id.endswith("/runtime")
    )
    write(tmp_path, path, '[project]\nname="worker"\nrequires-python=">=3.12"\n')
    assert (
        next(
            entry
            for entry in assess_map(added, tmp_path)
            if entry.item_id == runtime.id
        ).freshness
        == "stale"
    )
    edited = build_repository(tmp_path)
    updated = next(
        item for item in manifest_items(edited, path) if item.id == runtime.id
    )
    assert (
        updated.content_digest != runtime.content_digest
        and ">=3.12" in updated.statement
    )
    (tmp_path / path).unlink()
    removed = build_repository(tmp_path)
    assert not manifest_items(removed, path)
    assert (
        next(item for item in removed.items if item.id == runtime.id).lifecycle
        == "retired"
    )
    assert removed.curated.items == [authored]

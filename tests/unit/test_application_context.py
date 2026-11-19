import json

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import build_repository, load_repository_map
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.models import (
    ItemDependency,
    KnowledgeConflict,
    KnowledgeItem,
)
from code_mem_graph.knowledge.orient import mcp_result
from code_mem_graph.knowledge.scoping import InstructionIndexIncomplete
from code_mem_graph.knowledge.sources import capture_file
from code_mem_graph.mcp import handle_context, handle_item
from code_mem_graph.tokens import PayloadCounter


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    write(tmp_path, "package.json", json.dumps({"scripts": {"check": "do-not-run"}}))
    write(
        tmp_path,
        "README.md",
        "An application family.\n\n[Shared architecture](docs/architecture.md)\n",
    )
    write(
        tmp_path, "docs/architecture.md", "# Architecture\n\nShared widget guidance.\n"
    )
    for app, node in [("web", ">=20"), ("admin", ">=22")]:
        write(
            tmp_path,
            f"apps/{app}/package.json",
            json.dumps({"engines": {"node": node}, "scripts": {"test": "do-not-run"}}),
        )
        write(tmp_path, f"apps/{app}/src/widget.py", "def widget(): return True\n")
    for path, text in [
        ("AGENTS.md", "Repository-wide instructions."),
        ("apps/CLAUDE.md", "Parent application instructions."),
        ("apps/web/AGENTS.md", "Web-only instructions."),
        ("apps/admin/AGENTS.md", "Admin-only instructions."),
        ("apps/web/src/payments/CLAUDE.md", "Payments instructions."),
        (
            "apps/web/src/other/AGENTS.md",
            "Unrelated widget instructions for the other folder.",
        ),
        ("apps/web/tests/AGENTS.md", "Test-directory instructions."),
        ("examples/demo/AGENTS.md", "Example-only instructions."),
    ]:
        write(tmp_path, path, text + "\n")
    write(tmp_path, "apps/web/src/payments/widget.py", "def widget(): return True\n")
    write(tmp_path, "apps/web/src/other/other.py", "def other(): return True\n")
    write(
        tmp_path, "apps/web/tests/widget.test.ts", "export const widgetTest = true;\n"
    )
    build_repository(tmp_path, repository_id="app-context")
    return tmp_path


def request(**changes):
    return ContextRequest(
        **(
            dict(
                task="widget runtime test",
                application="apps/web",
                budget=100000,
                encoding="cl100k_base",
            )
            | changes
        )
    )


def entries(payload):
    return {
        item["id"]: item for group in ("items", "cautions") for item in payload[group]
    }


def guide_paths(payload):
    return {
        item["evidence"][0]["path"]
        for item in entries(payload).values()
        if item["id"].endswith("/agent-guidance")
    }


def test_application_filters_peer_candidates_and_facts_but_keeps_shared_guidance(
    project,
):
    payload = task_context(project, request())
    assert payload["scope"]["application"] == "apps/web"
    assert payload["source_candidates"]
    assert all(
        row["source"]["path"].startswith("apps/web/")
        for row in payload["source_candidates"]
    )
    assert not any(
        item["evidence"][0]["path"].startswith("apps/admin/")
        for item in entries(payload).values()
    )
    assert any(
        item["evidence"][0]["path"] == "docs/architecture.md"
        for item in entries(payload).values()
    )
    assert payload["scope"]["cross_boundary_proof_ids"] == []


def test_parent_and_matching_descendant_instructions_are_required_without_unrelated_rules(
    project,
):
    payload = task_context(project, request())
    assert guide_paths(payload) == {
        "AGENTS.md",
        "apps/CLAUDE.md",
        "apps/web/AGENTS.md",
        "apps/web/src/payments/CLAUDE.md",
        "apps/web/tests/AGENTS.md",
    }
    assert payload["coverage"]["instructions"]["status"] == "included"
    assert not payload["coverage"]["instructions"]["omitted_ids"]
    assert all(
        "Applies only within" in item["statement"]
        for item in entries(payload).values()
        if item["id"].endswith("/agent-guidance")
        and item["evidence"][0]["path"] != "AGENTS.md"
    )


def test_focused_item_read_inherits_ancestors_without_expanding_all_package_facts(
    project,
):
    document = load_repository_map(project)
    item = next(
        item
        for item in document.items
        if item.id.endswith("/runtime")
        and item.evidence[0].path == "apps/web/package.json"
    )
    payload = task_context(
        project,
        request(task="Inspect requested item", item_ids=[item.id]),
        focused=True,
    )
    assert guide_paths(payload) == {"AGENTS.md", "apps/CLAUDE.md", "apps/web/AGENTS.md"}
    assert set(entries(payload)) == {
        item.id,
        *payload["coverage"]["instructions"]["required_ids"],
    }


def test_unscoped_path_hints_remain_priorities_not_application_filters(project):
    payload = task_context(
        project, request(application=None, paths=["apps/web/src/widget.py"])
    )
    assert any(
        row["source"]["path"].startswith("apps/admin/")
        for row in payload["source_candidates"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"application": "apps/missing"},
        {"application": "apps"},
        {"paths": ["apps/admin/src/widget.py"]},
        {"application": "../outside"},
    ],
)
def test_unknown_invalid_or_contradictory_boundaries_are_rejected(project, changes):
    with pytest.raises(ValueError):
        task_context(project, request(**changes))


def test_explicit_peer_item_is_rejected(project):
    document = load_repository_map(project)
    item = next(
        item
        for item in document.items
        if item.id.endswith("/runtime")
        and item.evidence[0].path == "apps/admin/package.json"
    )
    with pytest.raises(ValueError, match="another application"):
        task_context(project, request(item_ids=[item.id]))


def test_new_applicable_instruction_requires_refresh_without_writing_the_map(project):
    saved = (project / ".project-knowledge/map.yaml").read_bytes()
    write(project, "apps/web/src/payments/AGENTS.md", "New payment policy.\n")
    with pytest.raises(InstructionIndexIncomplete, match="knowledge refresh"):
        task_context(project, request())
    assert (project / ".project-knowledge/map.yaml").read_bytes() == saved
    build_repository(project)
    assert "apps/web/src/payments/AGENTS.md" in guide_paths(
        task_context(project, request())
    )


def test_stale_or_removed_instructions_are_cautions_and_keep_context_partial(project):
    (project / "apps/web/AGENTS.md").unlink()
    payload = task_context(project, request())
    assert (
        payload["status"] == "partial"
        and payload["coverage"]["instructions"]["status"] == "unavailable"
    )
    assert any(
        item["evidence"][0]["path"] == "apps/web/AGENTS.md"
        and item["freshness"] == "stale"
        for item in payload["cautions"]
    )


def test_empty_instruction_file_can_be_indexed(project):
    write(project, "apps/web/src/payments/AGENTS.md", "")
    build_repository(project)
    assert "apps/web/src/payments/AGENTS.md" in guide_paths(
        task_context(project, request())
    )


def test_cross_application_dependency_and_conflict_proof_remain_explicit(project):
    document = load_repository_map(project)
    data = document.model_dump(mode="json")
    components = {
        component.paths[0]: component.id for component in document.generated.components
    }
    other = KnowledgeItem(
        id="admin/proof",
        topic="conventions",
        facets=["constraints"],
        title="Peer evidence",
        statement="A peer constraint needed to interpret this decision.",
        origin="maintainer",
        basis="documented",
        author="fixture",
        component_ids=[components["apps/admin"]],
        evidence=[capture_file(project, "apps/admin/src/widget.py")],
    )
    selected = KnowledgeItem(
        id="web/decision",
        topic="decision",
        title="Widget decision",
        statement="A scoped decision with peer evidence.",
        origin="maintainer",
        basis="documented",
        author="fixture",
        component_ids=[components["apps/web"]],
        evidence=[capture_file(project, "apps/web/src/widget.py")],
        dependencies=[
            ItemDependency(item_id=other.id, content_digest=other.content_digest)
        ],
    )
    data["curated"]["items"] = [
        other.model_dump(mode="json"),
        selected.model_dump(mode="json"),
    ]
    data["curated"]["conflicts"] = [
        KnowledgeConflict(
            id="cross/conflict",
            item_ids=[other.id, selected.id],
            description="A retained cross-application disagreement.",
        ).model_dump(mode="json")
    ]
    (project / ".project-knowledge/map.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    payload = task_context(project, request(item_ids=[selected.id]))
    assert {other.id, selected.id} <= entries(payload).keys()
    assert other.id in payload["scope"]["cross_boundary_proof_ids"]
    assert "apps/admin/AGENTS.md" in guide_paths(payload)
    assert payload["conflicts"]
    assert all(
        row["source"]["path"].startswith("apps/web/")
        for row in payload["source_candidates"]
    )


@pytest.mark.parametrize("transport", ["cli", "mcp"])
def test_tight_cap_discloses_omitted_required_instructions(project, transport):
    seen = []
    task_context(project, request(), transport=transport, observe=seen.append)
    payload = task_context(
        project, request(budget=seen[0]["size"]["tokens"]), transport=transport
    )
    assert payload["status"] == "partial" and payload["omitted"]["mandatory_claims"] > 0
    assert payload["coverage"]["instructions"]["omitted_ids"]
    wire = (
        mcp_result(payload).model_dump(mode="json", by_alias=True, exclude_none=True)
        if transport == "mcp"
        else payload
    )
    assert (
        PayloadCounter("cl100k_base")(wire)
        == payload["size"]["tokens"]
        <= payload["size"]["budget"]
    )


@pytest.mark.asyncio
async def test_cli_and_mcp_expose_application_and_missing_instruction_refresh(project):
    result = CliRunner().invoke(
        main,
        [
            "context",
            "--root",
            str(project),
            "--application",
            "apps/web",
            "--task",
            "widget",
            "--budget",
            "100000",
            "--encoding",
            "cl100k_base",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["scope"]["application"] == "apps/web"
    response = await handle_context(project, request().model_dump(mode="json"))
    assert not response.isError
    document = load_repository_map(project)
    item = next(
        item
        for item in document.items
        if item.id.endswith("/runtime")
        and item.evidence[0].path == "apps/web/package.json"
    )
    response = await handle_item(
        project,
        {
            "item_id": item.id,
            "application": "apps/web",
            "budget": 100000,
            "encoding": "cl100k_base",
        },
    )
    assert not response.isError
    assert guide_paths(json.loads(response.content[0].text)) == {
        "AGENTS.md",
        "apps/CLAUDE.md",
        "apps/web/AGENTS.md",
    }
    write(project, "apps/web/src/payments/AGENTS.md", "New rule.\n")
    response = await handle_context(project, request().model_dump(mode="json"))
    assert response.isError and "knowledge refresh" in response.content[0].text

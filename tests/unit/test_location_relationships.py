import json

import pytest
from pydantic import ValidationError

from code_mem_graph.knowledge.build import build_repository, load_repository_map
from code_mem_graph.knowledge.compact import expand_bundle
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.freshness import assess_map
from code_mem_graph.knowledge.models import LocationRelationship
from code_mem_graph.knowledge.orient import OrientRequest, orientation
from code_mem_graph.knowledge.relationships import is_import_record
from code_mem_graph.tokens import PayloadCounter


def write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    for path, content in {
        "package.json": "{}",
        "README.md": "A test repository.\n",
        "AGENTS.md": "Root guidance.\n",
        "apps/web/package.json": '{"scripts":{"test":"do-not-run"}}',
        "apps/web/tsconfig.json": json.dumps(
            {"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["./src/*"]}}}
        ),
        "apps/web/AGENTS.md": "Web guidance.\n",
        "apps/web/src/form.tsx": "import { schema } from './schema';\nimport { api } from '@/lib/client';\nexport const form = schema;\n",
        "apps/web/src/schema.ts": "export const schema = {};\n",
        "apps/web/src/lib/client.ts": "export { transport as api } from './transport';\n",
        "apps/web/src/lib/transport.ts": "export const transport = {};\n",
        "apps/web/src/form.test.tsx": "import { form } from './form';\nexport const check = form;\n",
        "apps/peer/package.json": "{}",
        "apps/peer/AGENTS.md": "Peer guidance.\n",
        "apps/peer/model.ts": "export const model = {};\n",
    }.items():
        write(tmp_path, path, content)
    build_repository(tmp_path, repository_id="relationships")
    return tmp_path


def context(root, **changes):
    return task_context(
        root,
        ContextRequest(
            **{
                "task": "unrelated",
                "paths": ["apps/web/src/form.tsx"],
                "application": "apps/web",
                "budget": 100000,
                "encoding": "cl100k_base",
                **changes,
            }
        ),
    )


def records(document):
    return {
        item.evidence[0].path: item for item in document.items if is_import_record(item)
    }


def candidate_paths(payload):
    return {
        row.get("source", {}).get("path", row.get("path"))
        for row in payload["source_candidates"]
    }


def test_static_edges_retain_import_ranges_alias_config_and_reverse_test_navigation(
    project,
):
    indexed = records(load_repository_map(project))
    form = indexed["apps/web/src/form.tsx"]
    assert {edge.target for edge in form.relationships} == {
        "apps/web/src/schema.ts",
        "apps/web/src/lib/client.ts",
    }
    assert form.evidence[0].start_line == 1 and form.evidence[0].end_line == 2
    assert "apps/web/tsconfig.json" in {ref.path for ref in form.evidence}
    payload = context(project)
    assert {
        "apps/web/src/schema.ts",
        "apps/web/src/lib/client.ts",
        "apps/web/src/form.test.tsx",
    } <= candidate_paths(payload)
    assert "apps/web/src/lib/transport.ts" not in candidate_paths(payload)
    test = next(
        row
        for row in payload["source_candidates"]
        if row["source"]["path"].endswith("form.test.tsx")
    )
    assert (
        test["basis"] == "evidenced_relationship" and test["role"] == "test_candidate"
    )
    assert test["relationship_item_ids"]
    assert payload["scope"]["selector"] == "recorded-text-and-paths-v3"


def test_import_index_does_not_flood_orientation_and_focused_reads_keep_edges(project):
    doc = load_repository_map(project)
    item = records(doc)["apps/web/src/form.tsx"]
    payload = orientation(project, OrientRequest(budget=100000, encoding="cl100k_base"))
    assert not any(row["id"] == item.id for row in payload["items"])
    focused = task_context(
        project,
        ContextRequest(
            task="inspect",
            item_ids=[item.id],
            application="apps/web",
            budget=100000,
            encoding="cl100k_base",
        ),
        focused=True,
    )
    assert next(row for row in focused["items"] if row["id"] == item.id)[
        "relationships"
    ]
    assert focused["source_candidates"] == []


def test_documented_workflow_locations_are_not_causal_edges(tmp_path):
    write(tmp_path, "pyproject.toml", '[project]\nname="example"\n')
    write(
        tmp_path,
        "README.md",
        "A project.\n\n## Migrations\n\nAfter changing `app/models.py`, create a revision. See `app/revisions`.\n\n```bash\nuv run alembic revision --autogenerate\n```\n",
    )
    write(tmp_path, "app/models.py", "class Example: pass\n")
    write(tmp_path, "app/revisions/first.py", "revision = 'one'\n")
    doc = build_repository(tmp_path, repository_id="workflow")
    item = next(item for item in doc.items if "Migrations" in item.title)
    assert {(edge.kind, edge.source, edge.target) for edge in item.relationships} == {
        ("documented_location", "README.md", "app/models.py"),
        ("documented_location", "README.md", "app/revisions"),
    }
    payload = context(tmp_path, application=None, paths=[], task="migrations revision")
    assert "app/models.py" in candidate_paths(payload)
    assert all(edge.kind != "static_import" for edge in item.relationships)


def test_python_modules_and_relative_imports_are_resolved_without_execution(tmp_path):
    for path, content in {
        "backend/pyproject.toml": '[project]\nname="service"\n',
        "backend/app/models.py": "raise RuntimeError('never execute')\n",
        "backend/app/routes/items.py": "from app.models import Item\nfrom ..models import Other\n",
        "backend/app/bad.py": "from ......app.models import Item\n",
    }.items():
        write(tmp_path, path, content)
    doc = build_repository(tmp_path, repository_id="python")
    indexed = records(doc)
    assert [
        edge.target for edge in indexed["backend/app/routes/items.py"].relationships
    ] == ["backend/app/models.py"]
    assert "backend/app/bad.py" not in indexed


def test_ambiguous_targets_comments_dynamic_imports_and_invalid_syntax_do_not_invent_edges(
    project,
):
    write(project, "apps/web/src/schema.tsx", "export const schema = {};\n")
    write(
        project,
        "apps/web/src/noise.ts",
        "// import './schema';\nconst text = \"import './schema'\";\nimport('./form');\n",
    )
    write(project, "apps/web/src/broken.ts", "import { from './form';\n")
    write(project, "apps/web/src/ambiguous.ts", "import { x } from './schema';\n")
    indexed = records(build_repository(project))
    assert [edge.target for edge in indexed["apps/web/src/form.tsx"].relationships] == [
        "apps/web/src/lib/client.ts"
    ]
    assert (
        not {
            "apps/web/src/noise.ts",
            "apps/web/src/broken.ts",
            "apps/web/src/ambiguous.ts",
        }
        & indexed.keys()
    )


def test_config_inheritance_or_invalid_nearest_config_does_not_guess_aliases(project):
    config = project / "apps/web/tsconfig.json"
    data = json.loads(config.read_bytes())
    data["extends"] = "./unknown.json"
    config.write_text(json.dumps(data), encoding="utf-8")
    write(project, "apps/web/src/tsconfig.json", "{invalid configuration")
    indexed = records(build_repository(project))
    assert [edge.target for edge in indexed["apps/web/src/form.tsx"].relationships] == [
        "apps/web/src/schema.ts"
    ]


def test_jsonc_aliases_preserve_strings_and_support_comments_and_trailing_commas(
    project,
):
    write(
        project,
        "apps/web/tsconfig.json",
        """{
      // Common tsconfig syntax.
      "compilerOptions": {
        "paths": {"@/*": ["./src/*",],}, /* inline comment */
        "custom": "https://example.invalid/value,}",
      },
    }""",
    )
    indexed = records(build_repository(project))
    assert "apps/web/src/lib/client.ts" in {
        edge.target for edge in indexed["apps/web/src/form.tsx"].relationships
    }


def test_alias_fallbacks_with_multiple_existing_targets_are_unknown(project):
    write(project, "apps/web/src/alternate/client.ts", "export const api = {};\n")
    write(
        project,
        "apps/web/tsconfig.json",
        json.dumps(
            {"compilerOptions": {"paths": {"@/*": ["./src/*", "./src/alternate/*"]}}}
        ),
    )
    # Both candidates for @/lib/client must exist before rejecting its mapping.
    write(project, "apps/web/src/alternate/lib/client.ts", "export const api = {};\n")
    indexed = records(build_repository(project))
    assert [edge.target for edge in indexed["apps/web/src/form.tsx"].relationships] == [
        "apps/web/src/schema.ts"
    ]


@pytest.mark.parametrize("change", ["target", "config", "membership"])
def test_stale_edges_are_cautions_and_do_not_expand_candidates(project, change):
    if change == "target":
        write(
            project, "apps/web/src/schema.ts", "export const schema = {changed:true};\n"
        )
    elif change == "config":
        write(project, "apps/web/tsconfig.json", "{}")
    else:
        write(project, "apps/web/src/new.ts", "export const added = true;\n")
    doc = load_repository_map(project)
    item = records(doc)["apps/web/src/form.tsx"]
    assert (
        next(
            row for row in assess_map(doc, project) if row.item_id == item.id
        ).freshness
        == "stale"
    )
    payload = context(project)
    assert item.id in {row["id"] for row in payload["cautions"]}
    assert "apps/web/src/schema.ts" not in candidate_paths(payload)


def test_cross_boundary_import_proof_is_cited_and_guidance_retained_without_peer_candidates(
    project,
):
    write(
        project,
        "apps/web/src/form.tsx",
        "import { model } from '../../peer/model';\nexport const form = model;\n",
    )
    build_repository(project)
    payload = context(project)
    assert "apps/peer/model.ts" not in candidate_paths(payload)
    assert (
        "apps/peer/model.ts" in payload["scope"]["cross_boundary_relationship_targets"]
    )
    assert any(
        row["id"].endswith("/agent-guidance")
        and row["evidence"][0]["path"] == "apps/peer/AGENTS.md"
        for row in payload["items"]
    )


def test_relationships_survive_compact_focused_wire_and_validate_endpoint_evidence(
    project,
):
    payload = context(project, representation="compact", detail="focused")
    expanded = expand_bundle(payload)
    assert any(row.get("relationships") for row in expanded["items"])
    assert PayloadCounter("cl100k_base")(payload) == payload["size"]["tokens"]
    item = records(load_repository_map(project))["apps/web/src/form.tsx"]
    data = item.model_dump(mode="json")
    data["relationships"][0]["target"] = "missing.ts"
    with pytest.raises(ValidationError, match="endpoints"):
        type(item).model_validate(data)
    with pytest.raises(ValidationError, match="distinct"):
        LocationRelationship(kind="static_import", source="same.ts", target="same.ts")


def test_import_cycles_are_source_edges_not_invalid_knowledge_dependency_cycles(
    project,
):
    write(
        project,
        "apps/web/src/schema.ts",
        "import { form } from './form';\nexport const schema = form;\n",
    )
    doc = build_repository(project)
    assert {"apps/web/src/form.tsx", "apps/web/src/schema.ts"} <= records(doc).keys()
    plain = next(item for item in doc.items if not item.relationships)
    assert "relationships" not in plain.model_dump(mode="json")

import json
from pathlib import Path
import stat
from types import SimpleNamespace

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import BuildError, build_repository, load_repository_map
from code_mem_graph.knowledge.compact import expand_bundle
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.contributions import (
    ContributionInput,
    add_contribution,
    load_contribution,
    render_contribution_template,
    retire_contribution,
    update_contribution,
)
from code_mem_graph.knowledge.io import load_map
from code_mem_graph.knowledge.models import (
    CommandCheck, Component, KnowledgeGap, KnowledgeItem, KnowledgeMap, LocationRelationship,
)
from code_mem_graph.knowledge.sources import capture_file
from code_mem_graph.knowledge.orient import OrientRequest, orientation
from code_mem_graph.knowledge.workflow import record_review, refresh_repository, show_item, status_report
from code_mem_graph.mcp import build_server, handle_context, handle_item, handle_orient
from code_mem_graph.onboarding import inspect_repository
from code_mem_graph.serialization import fingerprint
from code_mem_graph.tokens import PayloadCounter
from mcp.types import ListToolsRequest


def _write_files(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    root = _write_files(
        tmp_path / "py_target",
        {
            "README.md": (
                "# Billing Service\n\n"
                "A fictional Python service for ledger checks.\n\n"
                "## Architecture\n\n"
                "The ledger module lives under `src/billing/ledger.py`.\n"
            ),
            "pyproject.toml": (
                '[project]\nname = "billing-fixture"\nrequires-python = ">=3.11"\n'
                'dependencies = ["pydantic>=2.0"]\n\n'
                '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
            ),
            "src/billing/__init__.py": "",
            "src/billing/ledger.py": (
                "def calculate_balance(entries: list[int]) -> int:\n"
                "    # Pure calculation with no I/O.\n"
                "    return sum(entries)\n"
            ),
            "tests/test_ledger.py": (
                "def test_balance():\n"
                "    assert True\n"
            ),
        },
    )
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--root", str(root), "--repository-id", "billing-fixture"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def typescript_repo(tmp_path: Path) -> Path:
    root = _write_files(
        tmp_path / "ts_target",
        {
            "README.md": (
                "# Orders Web\n\n"
                "Requires Node 20 for local UI development.\n\n"
                "## Testing\n\n"
                "Run `yarn test:e2e` to execute end-to-end checks.\n"
            ),
            "package.json": json.dumps(
                {
                    "name": "orders-web-fixture",
                    "engines": {"node": ">=22"},
                    "scripts": {"test-e2e": "echo fixture", "lint": "echo lint"},
                    "dependencies": {"react": "19.0.0"},
                },
                indent=2,
            ),
            "tsconfig.json": json.dumps({"compilerOptions": {"strict": True}}, indent=2),
            "src/orders/OrdersView.tsx": (
                "export function OrdersView() {\n"
                "  return 'orders';\n"
                "}\n"
            ),
            "src/orders/OrdersView.test.tsx": "export const spec = true;\n",
        },
    )
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--root", str(root), "--repository-id", "orders-web-fixture"])
    assert result.exit_code == 0, result.output
    return root


def _write_contrib(path: Path, data: dict, *, as_json: bool = False) -> Path:
    text = json.dumps(data, indent=2) if as_json else yaml.safe_dump(data, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return path


def test_template_command_generates_valid_starter_and_exports_curated_item(python_repo: Path, tmp_path: Path):
    runner = CliRunner()
    yaml_out = tmp_path / "starter.yaml"
    created = runner.invoke(
        main,
        [
            "knowledge",
            "template",
            "--id",
            "architecture/ledger-isolation",
            "--topic",
            "architecture",
            "--author",
            "alex-maintainer",
            "--basis",
            "inferred",
            "--gap",
            "knowledge/architecture-rationale",
            "--output",
            str(yaml_out),
        ],
    )
    assert created.exit_code == 0, created.output
    parsed = load_contribution(yaml_out)
    assert parsed.id == "architecture/ledger-isolation"
    assert parsed.author == "alex-maintainer"
    assert parsed.basis == "inferred"
    assert parsed.addresses_gap_ids == ["knowledge/architecture-rationale"]

    # Refuses to overwrite an existing output file.
    overwrite = runner.invoke(main, ["knowledge", "template", "--output", str(yaml_out)])
    assert overwrite.exit_code != 0

    # JSON stdout format is also valid ContributionInput, and Python API template matches.
    json_out = runner.invoke(main, ["knowledge", "template", "--format", "json"])
    assert json_out.exit_code == 0, json_out.output
    assert ContributionInput.model_validate(json.loads(json_out.output)).basis == "maintainer"
    assert yaml.safe_load(render_contribution_template())["basis"] == "maintainer"

    # Cannot combine --from-item with starter overrides or export a generated item.
    conflict_opts = runner.invoke(
        main, ["knowledge", "template", "--root", str(python_repo), "--from-item", "project/purpose", "--author", "x"]
    )
    assert conflict_opts.exit_code != 0
    generated_export = runner.invoke(
        main, ["knowledge", "template", "--root", str(python_repo), "--from-item", "project/purpose"]
    )
    assert generated_export.exit_code != 0
    assert "only curated maintainer items" in generated_export.output


def test_python_repo_create_inspect_review_retrieve_update_and_refresh_lifecycle(python_repo: Path, tmp_path: Path):
    runner = CliRunner()
    doc = load_repository_map(python_repo)
    runtime_item = next(item for item in doc.generated.items if item.id.endswith("/runtime"))
    src_component = next(c for c in doc.generated.components if c.paths == ["src"])

    # Gap is initially open in generated layer, status, doctor, and orient.
    initial_status = status_report(python_repo)
    assert any(g["id"] == "knowledge/architecture-rationale" and g["effective_status"] == "open" for g in initial_status["gaps"])
    assert "knowledge/architecture-rationale" in inspect_repository(python_repo)["open_gaps"]

    contrib_file = _write_contrib(
        tmp_path / "rationale.yaml",
        {
            "id": "architecture/ledger-purity",
            "topic": "architecture",
            "title": "Ledger calculation stays pure",
            "statement": "calculate_balance performs deterministic summation without external I/O so unit tests stay isolated.",
            "author": "priya-maintainer",
            "basis": "inferred",
            "component_ids": [src_component.id],
            "facets": ["entrypoints"],
            "onboarding": ["responsibilities"],
            "evidence": [
                {"path": "src/billing/ledger.py", "start_line": 1, "end_line": 3},
                {"kind": "directory", "path": "src", "recursive": True},
            ],
            "dependencies": [runtime_item.id],
            "addresses_gap_ids": ["knowledge/architecture-rationale"],
        },
    )

    # Dry-run previews captured fingerprints/dependency versions without touching map or README.
    map_before = (python_repo / ".project-knowledge/map.yaml").read_bytes()
    brief_before = (python_repo / ".project-knowledge/README.md").read_bytes()
    preview_res = runner.invoke(main, ["knowledge", "add", str(contrib_file), "--root", str(python_repo), "--dry-run"])
    assert preview_res.exit_code == 0, preview_res.output
    preview = json.loads(preview_res.output)
    assert preview["dry_run"] is True
    assert preview["item"]["review"]["status"] == "unreviewed"
    assert preview["assessment"]["review_required"] is True
    assert len(preview["item"]["evidence"]) == 2
    assert preview["item"]["dependencies"][0]["content_digest"] == runtime_item.content_digest
    assert preview["addressed_gaps"][0]["effective_status"] == "open"
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == map_before
    assert (python_repo / ".project-knowledge/README.md").read_bytes() == brief_before

    # Add contribution for real.
    added_res = runner.invoke(
        main,
        [
            "knowledge",
            "add",
            str(contrib_file),
            "--root",
            str(python_repo),
            "--expected-map-digest",
            initial_status["map_digest"],
        ],
    )
    assert added_res.exit_code == 0, added_res.output
    added = json.loads(added_res.output)
    assert added["dry_run"] is False
    assert added["content_digest"] == preview["content_digest"]
    assert added["item"]["review"]["status"] == "unreviewed"

    # Before acceptance: inspectable via show, pending in review_queue, gap still open, caution in orient.
    shown = json.loads(runner.invoke(main, ["knowledge", "show", "architecture/ledger-purity", "--root", str(python_repo)]).output)
    assert shown["content_digest"] == added["content_digest"]
    assert shown["assessment"]["review_required"] is True
    assert shown["addressed_gaps"] == [
        {"id": "knowledge/architecture-rationale", "effective_status": "open", "effective_resolved_by": []}
    ]
    pre_status = status_report(python_repo)
    assert "architecture/ledger-purity" in pre_status["review_queue"]
    gap_pre = next(g for g in pre_status["gaps"] if g["id"] == "knowledge/architecture-rationale")
    assert gap_pre["status"] == "open"
    assert gap_pre["effective_status"] == "open"
    assert gap_pre["addressed_by"] == ["architecture/ledger-purity"]
    pre_orient = orientation(python_repo, OrientRequest(budget=30000, encoding="cl100k_base"))
    assert any(item["id"] == "architecture/ledger-purity" for item in pre_orient["cautions"])
    assert not any(item["id"] == "architecture/ledger-purity" for item in pre_orient["items"])
    assert any(g["id"] == "knowledge/architecture-rationale" for g in pre_orient["gaps"])

    # Explicit named version-bound acceptance.
    reviewed = runner.invoke(
        main,
        [
            "knowledge",
            "review",
            "architecture/ledger-purity",
            "--root",
            str(python_repo),
            "--expected-digest",
            shown["content_digest"],
            "--reviewer",
            "sam-reviewer",
            "--status",
            "accepted",
            "--note",
            "Verified against ledger.py lines 1-3.",
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output
    accepted = json.loads(reviewed.output)
    assert accepted["item"]["review"]["status"] == "accepted"
    assert accepted["item"]["review"]["reviewer"] == "sam-reviewer"
    assert accepted["assessment"]["freshness"] == "fresh"
    assert accepted["assessment"]["review_required"] is False
    assert accepted["addressed_gaps"][0]["effective_status"] == "resolved"

    # Effective gap closed while generated gap remains untouched; doctor and README reflect current state.
    post_doc = load_repository_map(python_repo)
    gen_gap = next(g for g in post_doc.generated.gaps if g.id == "knowledge/architecture-rationale")
    assert gen_gap.status == "open" and gen_gap.resolved_by == []
    post_status = status_report(python_repo)
    assert "architecture/ledger-purity" not in post_status["review_queue"]
    gap_post = next(g for g in post_status["gaps"] if g["id"] == "knowledge/architecture-rationale")
    assert gap_post["effective_status"] == "resolved"
    assert gap_post["effective_resolved_by"] == ["architecture/ledger-purity"]
    # Unrelated gaps remain open.
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "open" for g in post_status["gaps"])
    doctor = inspect_repository(python_repo)
    assert doctor["ready_for_reads"] is True
    assert doctor["reading_view_status"] == "current"
    assert "knowledge/architecture-rationale" not in doctor["open_gaps"]
    brief = (python_repo / ".project-knowledge/README.md").read_text(encoding="utf-8")
    assert "Addresses gaps: `knowledge/architecture-rationale`." in brief
    assert "- Gap `knowledge/architecture-rationale`:" not in brief

    # Retrievable through orient and task context.
    oriented = orientation(python_repo, OrientRequest(budget=30000, encoding="cl100k_base"))
    assert any(item["id"] == "architecture/ledger-purity" for item in oriented["items"])
    assert not any(g["id"] == "knowledge/architecture-rationale" for g in oriented["gaps"])
    assert "entrypoints" in oriented["coverage"]["included"]
    assert oriented["coverage"]["onboarding"]["responsibilities"]["status"] == "included"
    ctx = task_context(
        python_repo,
        ContextRequest(task="Refactor calculate_balance", paths=["src/billing/ledger.py"], budget=30000, encoding="cl100k_base"),
    )
    assert any(item["id"] == "architecture/ledger-purity" for item in ctx["items"])
    assert not any(g["id"] == "knowledge/architecture-rationale" for g in ctx["gaps"])

    # Export curated item with `knowledge template --from-item` and revise with version protection.
    exported_path = tmp_path / "exported.yaml"
    export_res = runner.invoke(
        main,
        [
            "knowledge",
            "template",
            "--root",
            str(python_repo),
            "--from-item",
            "architecture/ledger-purity",
            "--output",
            str(exported_path),
        ],
    )
    assert export_res.exit_code == 0, export_res.output
    exported_data = yaml.safe_load(exported_path.read_text(encoding="utf-8"))
    assert "digest" not in json.dumps(exported_data)
    exported_data["statement"] = (
        "calculate_balance stays pure and synchronous so ledger unit tests need no database or network."
    )
    _write_contrib(exported_path, exported_data)

    # Dry-run update leaves accepted claim unchanged.
    map_before_update = (python_repo / ".project-knowledge/map.yaml").read_bytes()
    dry_upd = runner.invoke(
        main,
        [
            "knowledge",
            "update",
            str(exported_path),
            "--root",
            str(python_repo),
            "--expected-digest",
            accepted["content_digest"],
            "--dry-run",
        ],
    )
    assert dry_upd.exit_code == 0, dry_upd.output
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == map_before_update

    # Real update invalidates previous acceptance and reopens gap.
    upd_res = runner.invoke(
        main,
        [
            "knowledge",
            "update",
            str(exported_path),
            "--root",
            str(python_repo),
            "--expected-digest",
            accepted["content_digest"],
        ],
    )
    assert upd_res.exit_code == 0, upd_res.output
    updated = json.loads(upd_res.output)
    assert updated["previous_content_digest"] == accepted["content_digest"]
    assert updated["previous_review_status"] == "accepted"
    assert updated["content_digest"] != accepted["content_digest"]
    assert updated["item"]["review"]["status"] == "unreviewed"
    assert updated["assessment"]["review_required"] is True
    assert updated["addressed_gaps"][0]["effective_status"] == "open"
    assert "knowledge/architecture-rationale" in inspect_repository(python_repo)["open_gaps"]

    # Re-accept the new version, then refresh after an unrelated file addition.
    record_review(
        python_repo,
        "architecture/ledger-purity",
        expected_digest=updated["content_digest"],
        reviewer="sam-reviewer",
        status="accepted",
    )
    (python_repo / "src/billing/notes.py").write_text("NOTE = 'unrelated'\n", encoding="utf-8")
    # Also add an unrelated curated component and curated gap to prove full curated preservation.
    doc_now = load_repository_map(python_repo)
    raw_map = doc_now.model_dump(mode="json")
    raw_map["curated"]["components"].append(
        Component(id="curated/finance", name="Finance rules", paths=["src/billing"], parent_id=src_component.id).model_dump(mode="json")
    )
    raw_map["curated"]["gaps"].append(
        KnowledgeGap(id="curated/audit-policy", topic="workflow", description="Audit signoff policy pending.").model_dump(mode="json")
    )
    (python_repo / ".project-knowledge/map.yaml").write_text(
        yaml.safe_dump(KnowledgeMap.model_validate(raw_map).model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    before_refresh = load_repository_map(python_repo)
    refreshed = refresh_repository(python_repo)
    assert refreshed["changes"]["curated_preserved"] is True
    after_refresh = load_repository_map(python_repo)
    assert after_refresh.curated == before_refresh.curated
    # Note: adding src/billing/notes.py changed directory `src` membership, so the item citing directory `src` becomes stale!
    assert show_item(python_repo, "architecture/ledger-purity")["assessment"]["freshness"] == "stale"
    assert any(
        g["id"] == "knowledge/architecture-rationale" and g["effective_status"] == "open"
        for g in refreshed["gaps"]
    )
    # Explicitly recapturing curated evidence after inspecting the change restores freshness and re-closes the gap.
    cur_digest = show_item(python_repo, "architecture/ledger-purity")["content_digest"]
    record_review(
        python_repo,
        "architecture/ledger-purity",
        expected_digest=cur_digest,
        reviewer="sam-reviewer",
        status="accepted",
        refresh_evidence=True,
    )
    assert any(
        g["id"] == "knowledge/architecture-rationale" and g["effective_status"] == "resolved"
        for g in status_report(python_repo)["gaps"]
    )


def test_source_dependency_rejection_and_retirement_reopen_addressed_gap(python_repo: Path, tmp_path: Path):
    doc = load_repository_map(python_repo)
    runtime_item = next(item for item in doc.generated.items if item.id.endswith("/runtime"))
    contrib = ContributionInput.model_validate(
        {
            "id": "workflow/release-owners",
            "topic": "workflow",
            "title": "Release ownership",
            "statement": "Billing releases require review against the supported Python version and ledger tests.",
            "author": "alex-maintainer",
            "basis": "maintainer",
            "evidence": ["tests/test_ledger.py"],
            "dependencies": [runtime_item.id],
            "addresses_gap_ids": ["knowledge/release-approvers"],
        }
    )
    added = add_contribution(python_repo, contrib)
    record_review(
        python_repo,
        contrib.id,
        expected_digest=added["content_digest"],
        reviewer="sam-reviewer",
        status="accepted",
    )
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "resolved" for g in status_report(python_repo)["gaps"])

    # 1. Editing cited source file makes item stale and reopens gap.
    (python_repo / "tests/test_ledger.py").write_text("def test_balance():\n    assert 1 == 1\n", encoding="utf-8")
    stale_status = status_report(python_repo)
    assert show_item(python_repo, contrib.id)["assessment"]["freshness"] == "stale"
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "open" for g in stale_status["gaps"])
    assert "knowledge/release-approvers" in inspect_repository(python_repo)["open_gaps"]
    assert CliRunner().invoke(main, ["knowledge", "status", "--root", str(python_repo), "--check"]).exit_code == 1

    # Explicit recapture restores fresh support and closes gap again.
    record_review(
        python_repo,
        contrib.id,
        expected_digest=show_item(python_repo, contrib.id)["content_digest"],
        reviewer="sam-reviewer",
        status="accepted",
        refresh_evidence=True,
    )
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "resolved" for g in status_report(python_repo)["gaps"])

    # 2. Upstream dependency change (pyproject.toml -> refresh) makes item stale and reopens gap.
    pyproject = python_repo / "pyproject.toml"
    pyproject.write_text(pyproject.read_text(encoding="utf-8").replace(">=3.11", ">=3.12"), encoding="utf-8")
    refreshed = refresh_repository(python_repo)
    assert refreshed["changes"]["curated_preserved"] is True
    assert contrib.id in refreshed["review_queue"]
    assert any(r.startswith("dependency_version_changed:") for r in show_item(python_repo, contrib.id)["assessment"]["reasons"])
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "open" for g in refreshed["gaps"])

    # Recapture dependencies after reviewing upstream change.
    record_review(
        python_repo,
        contrib.id,
        expected_digest=show_item(python_repo, contrib.id)["content_digest"],
        reviewer="sam-reviewer",
        status="accepted",
        refresh_evidence=True,
    )
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "resolved" for g in status_report(python_repo)["gaps"])

    # 3. Rejecting the contribution reopens the gap.
    cur_digest = show_item(python_repo, contrib.id)["content_digest"]
    record_review(python_repo, contrib.id, expected_digest=cur_digest, reviewer="sam-reviewer", status="rejected")
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "open" for g in status_report(python_repo)["gaps"])

    # Re-accept, then retire via CLI -> preserves historical acceptance, sets retired, reopens gap.
    record_review(python_repo, contrib.id, expected_digest=cur_digest, reviewer="sam-reviewer", status="accepted")
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "resolved" for g in status_report(python_repo)["gaps"])
    retired_res = CliRunner().invoke(
        main,
        ["knowledge", "retire", contrib.id, "--root", str(python_repo), "--expected-digest", cur_digest],
    )
    assert retired_res.exit_code == 0, retired_res.output
    retired = json.loads(retired_res.output)
    assert retired["item"]["lifecycle"] == "retired"
    assert retired["item"]["review"]["status"] == "accepted"
    assert retired["assessment"]["freshness"] == "stale"
    assert retired["addressed_gaps"][0]["effective_status"] == "open"
    assert any(g["id"] == "knowledge/release-approvers" and g["effective_status"] == "open" for g in status_report(python_repo)["gaps"])
    # Retired curated item cannot be updated or retired again.
    with pytest.raises(ValueError, match="retired"):
        update_contribution(python_repo, contrib, expected_digest=cur_digest)
    with pytest.raises(ValueError, match="already retired"):
        retire_contribution(python_repo, contrib.id, expected_digest=cur_digest)


def test_documented_basis_still_requires_acceptance_to_resolve_gap(python_repo: Path):
    contrib = ContributionInput.model_validate(
        {
            "id": "domain/billing-terms",
            "topic": "domain",
            "title": "Billing terminology",
            "statement": "README defines this repository as a ledger checks service.",
            "author": "alex-maintainer",
            "basis": "documented",
            "evidence": [{"path": "README.md", "start_line": 1, "end_line": 4}],
            "addresses_gap_ids": ["knowledge/domain-meaning"],
        }
    )
    added = add_contribution(python_repo, contrib)
    assert added["item"]["basis"] == "documented"
    assert added["item"]["review"]["status"] == "unreviewed"
    # Unreviewed contribution does not close the gap yet.
    assert any(g["id"] == "knowledge/domain-meaning" and g["effective_status"] == "open" for g in status_report(python_repo)["gaps"])
    # After explicit acceptance, the gap resolves.
    record_review(python_repo, contrib.id, expected_digest=added["content_digest"], reviewer="sam-reviewer", status="accepted")
    assert any(g["id"] == "knowledge/domain-meaning" and g["effective_status"] == "resolved" for g in status_report(python_repo)["gaps"])


def test_source_free_maintainer_assertion_remains_unknown_and_never_closes_gap(python_repo: Path, tmp_path: Path):
    contrib_file = _write_contrib(
        tmp_path / "assertion.json",
        {
            "id": "decision/team-cadence",
            "topic": "decision",
            "title": "Weekly triage decision",
            "statement": "The owning team reviews open ledger alerts every Thursday.",
            "author": "alex-maintainer",
            "basis": "maintainer",
            "addresses_gap_ids": ["knowledge/decisions"],
        },
        as_json=True,
    )
    runner = CliRunner()
    added_res = runner.invoke(main, ["knowledge", "add", str(contrib_file), "--root", str(python_repo)])
    assert added_res.exit_code == 0, added_res.output
    added = json.loads(added_res.output)
    assert added["item"]["basis"] == "maintainer"
    assert added["item"]["evidence"] == []
    assert added["assessment"]["freshness"] == "unknown"
    assert "no_verifiable_source" in added["assessment"]["reasons"]
    assert added["assessment"]["review_required"] is True

    # Explicit acceptance succeeds, keeps freshness unknown, and does not fabricate fresh gap closure.
    reviewed = runner.invoke(
        main,
        [
            "knowledge",
            "review",
            "decision/team-cadence",
            "--root",
            str(python_repo),
            "--expected-digest",
            added["content_digest"],
            "--reviewer",
            "sam-reviewer",
            "--status",
            "accepted",
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output
    accepted = json.loads(reviewed.output)
    assert accepted["item"]["review"]["status"] == "accepted"
    assert accepted["assessment"]["freshness"] == "unknown"
    assert "no_verifiable_source" in accepted["assessment"]["reasons"]
    assert accepted["assessment"]["review_required"] is True
    assert accepted["addressed_gaps"][0]["effective_status"] == "open"
    oriented = orientation(python_repo, OrientRequest(budget=30000, encoding="cl100k_base"))
    assert any(item["id"] == "decision/team-cadence" for item in oriented["cautions"])
    assert not any(item["id"] == "decision/team-cadence" for item in oriented["items"])
    assert any(g["id"] == "knowledge/decisions" for g in oriented["gaps"])


def test_invalid_inputs_references_duplicates_and_generated_overwrites_leave_files_unchanged(
    python_repo: Path, tmp_path: Path, monkeypatch
):
    runner = CliRunner()
    (python_repo / ".env").write_text("SECRET=never-read\n", encoding="utf-8")
    (python_repo / ".knowledgeignore").write_text("ignored_doc.md\n", encoding="utf-8")
    (python_repo / "ignored_doc.md").write_text("# Ignored\n", encoding="utf-8")

    # Seed one valid curated item first.
    first = add_contribution(
        python_repo,
        ContributionInput.model_validate(
            {
                "id": "conventions/naming",
                "topic": "conventions",
                "title": "Naming convention",
                "statement": "Use snake_case for billing functions.",
                "author": "alex",
                "basis": "maintainer",
                "evidence": ["src/billing/ledger.py"],
            }
        ),
    )
    second = add_contribution(
        python_repo,
        ContributionInput.model_validate(
            {
                "id": "conventions/ledger-style",
                "topic": "conventions",
                "title": "Ledger style",
                "statement": "Ledger helpers follow naming convention.",
                "author": "alex",
                "basis": "inferred",
                "dependencies": ["conventions/naming"],
            }
        ),
    )
    assert second["item"]["dependencies"][0]["item_id"] == "conventions/naming"

    map_before = (python_repo / ".project-knowledge/map.yaml").read_bytes()
    brief_before = (python_repo / ".project-knowledge/README.md").read_bytes()

    base = {
        "id": "architecture/rejected-case",
        "topic": "architecture",
        "title": "Rejected test case",
        "statement": "Should never persist.",
        "author": "alex",
        "basis": "maintainer",
        "evidence": ["README.md"],
    }

    bad_payloads = [
        # Duplicate curated ID
        {**base, "id": "conventions/naming"},
        # Generated item overwrite
        {**base, "id": "project/purpose"},
        # Reserved conflict-resolution namespace
        {**base, "id": "conflict-resolution/custom"},
        # Excluded / ignored / missing / nonportable / bad-range / wrong-kind evidence
        {**base, "evidence": [".env"]},
        {**base, "evidence": [".project-knowledge/map.yaml"]},
        {**base, "evidence": ["ignored_doc.md"]},
        {**base, "evidence": ["src/billing/missing.py"]},
        {**base, "evidence": ["../outside.py"]},
        {**base, "evidence": ["src\\billing\\ledger.py"]},
        {**base, "evidence": [{"path": "src/billing/ledger.py", "start_line": 1, "end_line": 99}]},
        {**base, "evidence": [{"path": "src/billing/ledger.py", "start_line": 2}]},
        {**base, "evidence": [{"path": "src/billing/ledger.py", "start_line": 3, "end_line": 1}]},
        {**base, "evidence": ["src/billing"]},
        {**base, "evidence": [{"kind": "directory", "path": "README.md"}]},
        # Unknown references
        {**base, "component_ids": ["directory/unknown"]},
        {**base, "dependencies": ["missing/item"]},
        {**base, "addresses_gap_ids": ["knowledge/nonexistent-gap"]},
        # Self dependency
        {**base, "dependencies": ["architecture/rejected-case"]},
        # Unsupported input fields / basis rules
        {**base, "review": {"status": "accepted", "reviewer": "fake"}},
        {**base, "origin": "generated"},
        {**base, "basis": "observed"},
        {**base, "basis": "documented", "evidence": []},
        {**base, "basis": "inferred", "evidence": [], "dependencies": []},
        {**base, "extra_field": "nope"},
        {**base, "author": "   "},
    ]

    for index, payload in enumerate(bad_payloads):
        path = _write_contrib(tmp_path / f"bad_{index}.yaml", payload)
        res = runner.invoke(main, ["knowledge", "add", str(path), "--root", str(python_repo)])
        assert res.exit_code != 0, f"payload {index} unexpectedly succeeded: {payload}"

    # Non-mapping YAML file is rejected.
    non_map = tmp_path / "list.yaml"
    non_map.write_text("- not-a-mapping\n", encoding="utf-8")
    assert runner.invoke(main, ["knowledge", "add", str(non_map), "--root", str(python_repo)]).exit_code != 0

    # Updating generated item, unknown ID, or introducing a dependency cycle are rejected.
    gen_update = _write_contrib(tmp_path / "gen_update.yaml", {**base, "id": "project/purpose"})
    assert (
        runner.invoke(
            main,
            ["knowledge", "update", str(gen_update), "--root", str(python_repo), "--expected-digest", "0" * 64],
        ).exit_code
        != 0
    )
    unknown_update = _write_contrib(tmp_path / "unknown_update.yaml", {**base, "id": "architecture/unknown"})
    assert (
        runner.invoke(
            main,
            ["knowledge", "update", str(unknown_update), "--root", str(python_repo), "--expected-digest", "0" * 64],
        ).exit_code
        != 0
    )
    cycle_update = _write_contrib(
        tmp_path / "cycle_update.yaml",
        {
            "id": "conventions/naming",
            "topic": "conventions",
            "title": "Naming convention",
            "statement": "Creates cycle.",
            "author": "alex",
            "basis": "inferred",
            "dependencies": ["conventions/ledger-style"],
        },
    )
    cycle_res = runner.invoke(
        main,
        [
            "knowledge",
            "update",
            str(cycle_update),
            "--root",
            str(python_repo),
            "--expected-digest",
            first["content_digest"],
        ],
    )
    assert cycle_res.exit_code != 0
    assert "cycles" in cycle_res.output

    # Retiring a generated item is rejected.
    gen_digest = show_item(python_repo, "project/purpose")["content_digest"]
    assert (
        runner.invoke(
            main, ["knowledge", "retire", "project/purpose", "--root", str(python_repo), "--expected-digest", gen_digest]
        ).exit_code
        != 0
    )

    # Reparse point / symlink evidence is rejected before reading.
    original_lstat = Path.lstat
    target = python_repo / "README.md"
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024, raising=False)

    def fake_lstat(path, *args, **kwargs):
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=1024)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    link_path = _write_contrib(tmp_path / "link.yaml", base)
    assert runner.invoke(main, ["knowledge", "add", str(link_path), "--root", str(python_repo)]).exit_code != 0
    monkeypatch.undo()

    # None of the invalid operations modified map.yaml or README.md.
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == map_before
    assert (python_repo / ".project-knowledge/README.md").read_bytes() == brief_before


def test_stale_version_and_concurrent_edits_are_rejected_without_overwriting(python_repo: Path, tmp_path: Path, monkeypatch):
    runner = CliRunner()
    v1_file = _write_contrib(
        tmp_path / "v1.yaml",
        {
            "id": "decision/storage",
            "topic": "decision",
            "title": "Storage decision",
            "statement": "Version 1 statement.",
            "author": "alex",
            "basis": "maintainer",
            "evidence": ["README.md"],
        },
    )
    v1 = json.loads(runner.invoke(main, ["knowledge", "add", str(v1_file), "--root", str(python_repo)]).output)

    v2_file = _write_contrib(
        tmp_path / "v2.yaml",
        {
            "id": "decision/storage",
            "topic": "decision",
            "title": "Storage decision",
            "statement": "Version 2 newer statement.",
            "author": "alex",
            "basis": "maintainer",
            "evidence": ["README.md"],
        },
    )
    v2 = json.loads(
        runner.invoke(
            main,
            ["knowledge", "update", str(v2_file), "--root", str(python_repo), "--expected-digest", v1["content_digest"]],
        ).output
    )
    map_v2 = (python_repo / ".project-knowledge/map.yaml").read_bytes()
    brief_v2 = (python_repo / ".project-knowledge/README.md").read_bytes()

    # Outdated digest is rejected and v2 is preserved.
    stale_file = _write_contrib(
        tmp_path / "stale.yaml",
        {
            "id": "decision/storage",
            "topic": "decision",
            "title": "Storage decision",
            "statement": "Outdated writer trying to clobber v2.",
            "author": "other-author",
            "basis": "maintainer",
            "evidence": ["README.md"],
        },
    )
    stale_res = runner.invoke(
        main,
        ["knowledge", "update", str(stale_file), "--root", str(python_repo), "--expected-digest", v1["content_digest"]],
    )
    assert stale_res.exit_code != 0
    assert "version changed" in stale_res.output
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == map_v2
    assert (python_repo / ".project-knowledge/README.md").read_bytes() == brief_v2

    # Outdated expected_map_digest on add and update is rejected.
    stale_map_add = _write_contrib(
        tmp_path / "stale_add.yaml",
        {
            "id": "decision/other",
            "topic": "decision",
            "title": "Other",
            "statement": "Other decision.",
            "author": "alex",
            "basis": "maintainer",
            "evidence": ["README.md"],
        },
    )
    assert (
        runner.invoke(
            main,
            ["knowledge", "add", str(stale_map_add), "--root", str(python_repo), "--expected-map-digest", v1["map_digest"]],
        ).exit_code
        != 0
    )
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == map_v2

    # Held build lock rejects edit cleanly.
    lock = python_repo / ".project-knowledge/.build.lock"
    lock.write_text("held", encoding="utf-8")
    try:
        with pytest.raises(BuildError, match="another build"):
            add_contribution(python_repo, load_contribution(stale_map_add))
    finally:
        lock.unlink()
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == map_v2

    # Concurrent file modification between load and atomic replace is never overwritten.
    from code_mem_graph.knowledge import workflow as workflow_mod

    original_render = workflow_mod.render_brief
    concurrent_bytes = None

    def concurrent_write(document, root):
        nonlocal concurrent_bytes
        path = root / ".project-knowledge/map.yaml"
        current = yaml.safe_load(path.read_text(encoding="utf-8"))
        current["refresh"] = {"status": "failed", "error": "concurrent writer"}
        concurrent_bytes = yaml.safe_dump(current, sort_keys=False).encode("utf-8")
        path.write_bytes(concurrent_bytes)
        return original_render(document, root)

    monkeypatch.setattr(workflow_mod, "render_brief", concurrent_write)
    with pytest.raises(BuildError, match="changed during"):
        update_contribution(
            python_repo,
            load_contribution(stale_file),
            expected_digest=v2["content_digest"],
        )
    assert (python_repo / ".project-knowledge/map.yaml").read_bytes() == concurrent_bytes


def test_typescript_repo_contributions_preserve_conflicts_and_p03e_warnings(typescript_repo: Path, tmp_path: Path):
    runner = CliRunner()
    initial = status_report(typescript_repo)
    # P03e missing script key (`yarn test:e2e` vs `test-e2e`) stays a warning, NOT a hard conflict.
    doc = load_repository_map(typescript_repo)
    cmd_warnings = [item for item in doc.generated.items if item.command_checks and item.command_checks[0].status == "script_not_declared"]
    assert len(cmd_warnings) == 1
    assert len(initial["conflicts"]) == 1  # Only the README Node 20 vs package.json >=22 conflict.
    conflict = initial["conflicts"][0]

    # Resolve the Node runtime conflict first (creates a curated conflict-resolution/... decision).
    resolved_res = runner.invoke(
        main,
        [
            "knowledge",
            "resolve-conflict",
            conflict["id"],
            "--root",
            str(typescript_repo),
            "--expected-map-digest",
            initial["map_digest"],
            "--reviewer",
            "dana-maintainer",
            "--explanation",
            "Follow package.json engines >=22; README Node 20 note is outdated.",
        ],
    )
    assert resolved_res.exit_code == 0, resolved_res.output
    resolution_id = next(
        item.id for item in load_repository_map(typescript_repo).curated.items if item.id.startswith("conflict-resolution/")
    )

    # Add a TypeScript domain/vocabulary contribution addressing knowledge/domain-meaning.
    src_component = next(c for c in doc.generated.components if c.paths == ["src"])
    ts_contrib = _write_contrib(
        tmp_path / "orders_vocab.yaml",
        {
            "id": "domain/orders-view",
            "topic": "domain",
            "title": "OrdersView terminology",
            "statement": "OrdersView renders the customer order list from server state; no client-side calculation.",
            "author": "dana-maintainer",
            "basis": "maintainer",
            "component_ids": [src_component.id],
            "evidence": [{"path": "src/orders/OrdersView.tsx", "start_line": 1, "end_line": 3}],
            "addresses_gap_ids": ["knowledge/domain-meaning"],
        },
    )
    added = json.loads(runner.invoke(main, ["knowledge", "add", str(ts_contrib), "--root", str(typescript_repo)]).output)
    reviewed = runner.invoke(
        main,
        [
            "knowledge",
            "review",
            "domain/orders-view",
            "--root",
            str(typescript_repo),
            "--expected-digest",
            added["content_digest"],
            "--reviewer",
            "dana-maintainer",
            "--status",
            "accepted",
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output

    # Conflict resolution item cannot be modified or retired through contribution commands.
    res_digest = show_item(typescript_repo, resolution_id)["content_digest"]
    assert (
        runner.invoke(
            main,
            ["knowledge", "retire", resolution_id, "--root", str(typescript_repo), "--expected-digest", res_digest],
        ).exit_code
        != 0
    )

    # Refresh preserves conflict resolution, curated contribution, gap link, and P03e command warning.
    refreshed = refresh_repository(typescript_repo)
    assert refreshed["changes"]["curated_preserved"] is True
    assert refreshed["conflicts"][0]["effective_status"] == "resolved"
    assert any(g["id"] == "knowledge/domain-meaning" and g["effective_status"] == "resolved" for g in refreshed["gaps"])

    # Context exposes the contribution, conflict resolution, and P03e warning without executing commands.
    ctx = task_context(
        typescript_repo,
        ContextRequest(task="Update OrdersView", paths=["src/orders/OrdersView.tsx"], budget=30000, encoding="cl100k_base"),
    )
    assert any(item["id"] == "domain/orders-view" for item in ctx["items"])
    assert any(item.get("command_checks") for item in ctx["items"])
    assert ctx["coverage"]["documented_commands"]["script_not_declared_ids"] == [cmd_warnings[0].id]


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ["detailed", "compact"])
@pytest.mark.parametrize("detail", ["full", "focused"])
async def test_contributions_preserve_compact_focused_accounting_and_readonly_mcp(
    python_repo: Path, representation: str, detail: str
):
    # Add two contributions: one accepted addressing a gap, one unreviewed addressing another gap.
    accepted_contrib = ContributionInput.model_validate(
        {
            "id": "architecture/ledger-boundary",
            "topic": "architecture",
            "title": "Ledger boundary",
            "statement": "Ledger calculations stay pure under src/billing/ledger.py.",
            "author": "alex-maintainer",
            "basis": "maintainer",
            "evidence": [{"path": "src/billing/ledger.py", "start_line": 1, "end_line": 3}],
            "addresses_gap_ids": ["knowledge/architecture-rationale"],
        }
    )
    added = add_contribution(python_repo, accepted_contrib)
    record_review(
        python_repo,
        accepted_contrib.id,
        expected_digest=added["content_digest"],
        reviewer="sam-reviewer",
        status="accepted",
    )
    pending_contrib = ContributionInput.model_validate(
        {
            "id": "domain/ledger-entry",
            "topic": "domain",
            "title": "Ledger entry meaning",
            "statement": "Each integer represents minor currency units.",
            "author": "alex-maintainer",
            "basis": "maintainer",
            "evidence": ["src/billing/ledger.py"],
            "addresses_gap_ids": ["knowledge/domain-meaning"],
        }
    )
    add_contribution(python_repo, pending_contrib)

    # MCP server exposes ONLY read-only tools.
    server = build_server(python_repo)
    assert server.name == "code-mem-graph"
    tools_result = await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    assert {tool.name for tool in tools_result.root.tools} == {"orient", "context", "knowledge_item"}

    # CLI orient + context accounting and lossless compact roundtrip.
    for kind in ("orient", "context"):
        common = dict(budget=30000, encoding="cl100k_base", representation=representation, detail=detail)
        payload = (
            orientation(python_repo, OrientRequest(**common), transport="cli")
            if kind == "orient"
            else task_context(
                python_repo,
                ContextRequest(task="Inspect ledger boundary", paths=["src/billing/ledger.py"], **common),
                transport="cli",
            )
        )
        assert PayloadCounter("cl100k_base")(payload) == payload["size"]["tokens"] <= 30000
        expanded = expand_bundle(payload)
        assert any(
            item["id"] == "architecture/ledger-boundary"
            and item.get("addresses_gap_ids") == ["knowledge/architecture-rationale"]
            for item in expanded["items"]
        )
        assert not any(g["id"] == "knowledge/architecture-rationale" for g in expanded["gaps"])
        open_domain = next(g for g in expanded["gaps"] if g["id"] == "knowledge/domain-meaning")
        assert open_domain["effective_status"] == "open"
        assert open_domain["addressed_by"] == ["domain/ledger-entry"]

    # MCP orient, context, and knowledge_item enforce complete CallToolResult envelope accounting.
    for tool_name, handler, args in (
        ("orient", handle_orient, {"budget": 30000, "encoding": "cl100k_base", "representation": representation, "detail": detail}),
        (
            "context",
            handle_context,
            {
                "task": "Inspect ledger boundary",
                "paths": ["src/billing/ledger.py"],
                "budget": 30000,
                "encoding": "cl100k_base",
                "representation": representation,
                "detail": detail,
            },
        ),
        (
            "knowledge_item",
            handle_item,
            {
                "item_id": "architecture/ledger-boundary",
                "budget": 30000,
                "encoding": "cl100k_base",
                "representation": representation,
                "detail": detail,
            },
        ),
    ):
        response = await handler(python_repo, args)
        assert response.isError is False, f"{tool_name} failed: {response}"
        wire = response.model_dump(mode="json", by_alias=True, exclude_none=True)
        parsed = json.loads(response.content[0].text)
        assert PayloadCounter("cl100k_base")(wire) == parsed["size"]["tokens"] <= 30000
        expanded = expand_bundle(parsed)
        assert any(item["id"] == "architecture/ledger-boundary" for item in expanded["items"])


def test_historical_maps_preserve_exact_digests_without_addresses_gap_ids():
    fixtures_root = Path(__file__).resolve().parents[1] / "fixtures/knowledge"
    for name in ("python_service", "typescript_app"):
        map_path = fixtures_root / name / "map.yaml"
        raw = yaml.safe_load(map_path.read_text(encoding="utf-8"))
        document = load_map(map_path)
        dumped = document.model_dump(mode="json")
        # Additive addresses_gap_ids field is omitted when empty so historical digests stay identical.
        for layer in ("generated", "curated"):
            for item in dumped[layer]["items"]:
                assert "addresses_gap_ids" not in item
        assert fingerprint(dumped) == fingerprint(raw)
        for item in document.items:
            if item.review.status != "unreviewed":
                assert item.review.content_digest == item.content_digest


def test_model_validates_addresses_gap_ids_and_digest_binding(python_repo: Path):
    from pydantic import ValidationError
    from code_mem_graph.knowledge.models import KnowledgeItem

    added = add_contribution(
        python_repo,
        ContributionInput.model_validate(
            {
                "id": "architecture/gap-digest-test",
                "topic": "architecture",
                "title": "Gap digest test",
                "statement": "Explains ledger module boundaries.",
                "author": "alex",
                "basis": "maintainer",
                "evidence": ["src/billing/ledger.py"],
                "addresses_gap_ids": ["knowledge/architecture-rationale"],
            }
        ),
    )
    record_review(
        python_repo,
        "architecture/gap-digest-test",
        expected_digest=added["content_digest"],
        reviewer="sam",
        status="accepted",
    )
    doc = load_repository_map(python_repo)
    raw = doc.model_dump(mode="json")

    # Mutating addresses_gap_ids without clearing review fails claim-version check.
    tampered = json.loads(json.dumps(raw))
    tampered["curated"]["items"][0]["addresses_gap_ids"] = ["knowledge/decisions"]
    with pytest.raises(ValidationError, match="claim version"):
        KnowledgeItem.model_validate(tampered["curated"]["items"][0])

    # Unknown gap ID fails KnowledgeMap graph validation.
    unknown_gap = json.loads(json.dumps(raw))
    unknown_gap["curated"]["items"][0]["review"] = {"status": "unreviewed", "reviewer": None, "content_digest": None, "note": None}
    unknown_gap["curated"]["items"][0]["addresses_gap_ids"] = ["knowledge/not-a-gap"]
    with pytest.raises(ValidationError, match="unknown gap reference"):
        KnowledgeMap.model_validate(unknown_gap)

    # Duplicate gap IDs fail KnowledgeItem validation.
    dup_gap = json.loads(json.dumps(raw))
    dup_gap["curated"]["items"][0]["review"] = {"status": "unreviewed", "reviewer": None, "content_digest": None, "note": None}
    dup_gap["curated"]["items"][0]["addresses_gap_ids"] = ["knowledge/decisions", "knowledge/decisions"]
    with pytest.raises(ValidationError, match="duplicate gap references"):
        KnowledgeItem.model_validate(dup_gap["curated"]["items"][0])


def _seed_structured_contribution(root: Path) -> KnowledgeItem:
    """Represent a valid pre-existing curated record authored via the map/API."""
    manifest = root / "package.json"
    manifest.write_text('{"scripts":{"lint":"never-run"}}', encoding="utf-8")
    readme = root / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nRun `npm run lint` for checks.\n", encoding="utf-8")
    document = build_repository(root)
    item = KnowledgeItem(
        id="architecture/structured-knowledge", topic="architecture",
        title="Existing curated metadata", statement="The documented ledger location and lint declaration.",
        origin="maintainer", basis="maintainer", author="fixture-maintainer",
        component_ids=["project"],
        evidence=[capture_file(root, path) for path in ("README.md", "src/billing/ledger.py", "package.json")],
        relationships=[LocationRelationship(kind="documented_location", source="README.md", target="src/billing/ledger.py")],
        command_checks=[CommandCheck(
            document="README.md", manager="npm", script="lint", directory=".", manifest="package.json",
            status="declared_script", scope_basis="document_directory",
        )],
    )
    data = document.model_dump(mode="json")
    data["curated"]["items"].append(item.model_dump(mode="json"))
    (root / ".project-knowledge/map.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    record_review(root, item.id, expected_digest=item.content_digest, reviewer="fixture-reviewer", status="accepted")
    return next(entry for entry in load_repository_map(root).curated.items if entry.id == item.id)


@pytest.mark.parametrize("output_format", ["yaml", "json"])
def test_export_update_preserves_existing_structured_metadata(python_repo: Path, tmp_path: Path, output_format):
    runner = CliRunner()
    item = _seed_structured_contribution(python_repo)
    exported = runner.invoke(main, ["knowledge", "template", "--root", str(python_repo),
                                   "--from-item", item.id, "--format", output_format])
    assert exported.exit_code == 0, exported.output
    edited = yaml.safe_load(exported.output)
    edited["statement"] += " Revised explanation."
    contribution_file = _write_contrib(tmp_path / ("edited." + output_format), edited, as_json=output_format == "json")
    arguments = ["knowledge", "update", str(contribution_file), "--root", str(python_repo),
                 "--expected-digest", item.content_digest]
    before = [(python_repo / ".project-knowledge" / name).read_bytes() for name in ("map.yaml", "README.md")]
    preview = runner.invoke(main, [*arguments, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert [(python_repo / ".project-knowledge" / name).read_bytes() for name in ("map.yaml", "README.md")] == before
    updated = runner.invoke(main, arguments)
    assert updated.exit_code == 0, updated.output
    for output in (preview.output, updated.output):
        report = json.loads(output)
        assert report["item"]["relationships"] == [edge.model_dump(mode="json") for edge in item.relationships]
        assert report["item"]["command_checks"] == [check.model_dump(mode="json") for check in item.command_checks]
        assert report["item"]["review"]["status"] == "unreviewed"
        assert report["content_digest"] != item.content_digest
    stored = next(entry for entry in load_repository_map(python_repo).curated.items if entry.id == item.id)
    assert stored.relationships == item.relationships and stored.command_checks == item.command_checks


@pytest.mark.parametrize("missing_path", ["README.md", "src/billing/ledger.py", "package.json"])
def test_update_rejects_removed_structured_metadata_evidence_without_writes(python_repo: Path, tmp_path: Path, missing_path):
    runner = CliRunner()
    item = _seed_structured_contribution(python_repo)
    exported = runner.invoke(main, ["knowledge", "template", "--root", str(python_repo), "--from-item", item.id])
    assert exported.exit_code == 0, exported.output
    edited = yaml.safe_load(exported.output)
    edited["evidence"] = [reference for reference in edited["evidence"] if reference["path"] != missing_path]
    contribution_file = _write_contrib(tmp_path / "unsupported-edit.yaml", edited)
    arguments = ["knowledge", "update", str(contribution_file), "--root", str(python_repo),
                 "--expected-digest", item.content_digest]
    before = [(python_repo / ".project-knowledge" / name).read_bytes() for name in ("map.yaml", "README.md")]
    for flags in (["--dry-run"], []):
        result = runner.invoke(main, [*arguments, *flags])
        assert result.exit_code != 0
        assert "retain evidence for" in result.output and missing_path in result.output
        assert [(python_repo / ".project-knowledge" / name).read_bytes() for name in ("map.yaml", "README.md")] == before

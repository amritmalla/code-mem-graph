import json
from pathlib import Path
import shutil
import sys

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import BuildError, load_repository_map
from code_mem_graph.knowledge.models import KnowledgeItem, review_item
from code_mem_graph.onboarding import (
    agent_configuration,
    initialize_repository,
    inspect_repository,
)


@pytest.fixture(params=["python_service", "typescript_app"])
def repository(tmp_path, request):
    root = tmp_path / "unrelated project with spaces"
    shutil.copytree(
        Path(__file__).parents[1] / "fixtures" / "knowledge" / request.param, root
    )
    return root


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_doctor_missing_map_is_actionable_and_read_only(repository):
    before = files(repository)
    report = inspect_repository(repository)
    assert report["status"] == "not_initialized" and report["issues"] == [
        "knowledge_map_missing"
    ]
    assert report["client_connection_verified"] is False
    assert files(repository) == before
    result = CliRunner().invoke(main, ["doctor", "--root", str(repository), "--json"])
    assert (
        result.exit_code == 1
        and json.loads(result.output)["status"] == "not_initialized"
    )


def test_init_and_rerun_preserve_sources_client_config_and_curated_knowledge(
    repository,
):
    (repository / ".mcp.json").write_text(
        '{"mcpServers":{"other":{"command":"keep-me"}}}', encoding="utf-8"
    )
    before = files(repository)
    report = initialize_repository(repository, repository_id="external-project")
    assert report["reading_view_status"] == "current"
    if report["unresolved_conflicts"]:
        assert (
            not report["ready_for_reads"] and "unresolved_conflicts" in report["issues"]
        )
    else:
        assert report["ready_for_reads"]
    assert report["open_gaps"] and report["model_calls"] == 0
    assert all(
        (repository / path).read_bytes() == value for path, value in before.items()
    )
    document = load_repository_map(repository)
    data = document.model_dump(mode="json")
    authored = review_item(
        KnowledgeItem(
            id="decision/local",
            topic="decision",
            title="Local decision",
            statement="Maintainer-authored explanation.",
            origin="maintainer",
            basis="maintainer",
            author="maintainer",
        ),
        reviewer="maintainer",
        status="accepted",
    )
    data["curated"]["items"].append(authored.model_dump(mode="json"))
    (repository / ".project-knowledge/map.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    initialize_repository(repository)
    assert load_repository_map(repository).curated.items == [authored]
    assert all(
        (repository / path).read_bytes() == value for path, value in before.items()
    )
    saved = files(repository)
    with pytest.raises(BuildError, match="differs"):
        initialize_repository(repository, repository_id="wrong-project")
    assert files(repository) == saved


def test_doctor_reports_source_edits_and_outdated_reading_view_without_repair(
    repository,
):
    initialize_repository(repository, repository_id="external-project")
    (repository / "README.md").write_text(
        "Changed purpose after knowledge was built.\n", encoding="utf-8"
    )
    (repository / ".project-knowledge/README.md").write_text(
        "Old or edited reading view.\n", encoding="utf-8"
    )
    before = files(repository)
    report = inspect_repository(repository)
    assert report["status"] == "needs_attention" and not report["ready_for_reads"]
    assert "source_evidence_needs_refresh" in report["issues"]
    assert report["reading_view_status"] == "outdated"
    assert files(repository) == before
    refreshed = initialize_repository(repository)
    assert refreshed["reading_view_status"] == "current"
    assert "source_evidence_needs_refresh" not in refreshed["issues"]
    # The TypeScript fixture has an intentional runtime conflict; rebuilding
    # does not silently accept or erase its pending review.
    assert refreshed["ready_for_reads"] == (not refreshed["issues"])


def test_invalid_existing_map_is_preserved_and_reported(repository):
    directory = repository / ".project-knowledge"
    directory.mkdir()
    (directory / "map.yaml").write_text("not: [valid", encoding="utf-8")
    before = files(repository)
    report = inspect_repository(repository)
    assert report["issues"] == ["knowledge_map_invalid_or_unreadable"]
    result = CliRunner().invoke(
        main, ["init", "--root", str(repository), "--repository-id", "external-project"]
    )
    assert result.exit_code != 0 and files(repository) == before


def test_cli_initialization_and_root_pinned_config_from_other_working_directory(
    repository, tmp_path, monkeypatch
):
    runner = CliRunner()
    missing_id = runner.invoke(main, ["init", "--root", str(repository)])
    assert missing_id.exit_code != 0 and "explicit repository ID" in missing_id.output
    result = runner.invoke(
        main,
        [
            "init",
            "--root",
            str(repository),
            "--repository-id",
            "external-project",
            "--json",
        ],
    )
    assert result.exit_code == 0
    setup = json.loads(result.output)
    assert (
        setup["reading_view_status"] == "current"
        and setup["repository_id"] == "external-project"
    )
    monkeypatch.chdir(tmp_path)
    config = runner.invoke(main, ["mcp-config", "--root", str(repository)])
    assert config.exit_code == 0
    assert (
        json.loads(config.output)
        == setup["mcp_config"]
        == agent_configuration(repository)
    )
    server = setup["mcp_config"]["mcpServers"]["code-mem-graph"]
    assert server["command"] == sys.executable
    assert server["args"] == [
        "-m",
        "code_mem_graph.cli",
        "serve",
        "--root",
        str(repository.resolve()),
    ]
    plain = runner.invoke(main, ["doctor", "--root", str(repository)])
    assert (
        plain.exit_code == (0 if setup["ready_for_reads"] else 1)
        and "Missing knowledge:" in plain.output
    )
    assert "Client connection is not verified" in plain.output

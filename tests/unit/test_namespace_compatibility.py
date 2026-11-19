"""All implementation and entry points use the canonical namespace."""
import ast
import importlib
from pathlib import Path
import tomllib

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main


@pytest.mark.parametrize("module", [
    "config", "tokens", "units.model", "db.repository", "server",
    "embeddings.factory", "llm.factory",
])
def test_engine_modules_use_the_canonical_namespace(module):
    canonical = importlib.import_module("code_mem_graph.engine." + module)
    assert canonical.__name__ == "code_mem_graph.engine." + module
    assert "code_mem_graph" in Path(canonical.__file__).parts


def test_engine_factory_has_a_single_module_instance(monkeypatch):
    canonical = importlib.import_module("code_mem_graph.engine.embeddings.factory")
    marker = object()
    monkeypatch.setattr(canonical, "make_embedder", marker)
    assert importlib.import_module("code_mem_graph.engine.embeddings.factory").make_embedder is marker


def test_removed_namespace_is_not_shipped_or_imported():
    root = Path(__file__).resolve().parents[2] / "src"
    assert not (root / "claude_repo_mem").exists()
    metadata = tomllib.loads((root.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "code-mem-graph"
    assert metadata["project"]["scripts"] == {"code-mem-graph": "code_mem_graph.cli:main"}
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/code_mem_graph"]
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("claude_repo_mem"), path
            elif isinstance(node, ast.Import):
                assert not any(name.name.startswith("claude_repo_mem") for name in node.names), path


def test_memory_cli_uses_existing_state_without_touching_knowledge(tmp_path):
    from code_mem_graph.engine.cli import main as engine_main
    from code_mem_graph.engine.config import Settings
    from code_mem_graph.engine.db.connection import connect

    (tmp_path / "README.md").write_text("# Example\n\nA repository for health checks.\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def health(): return 'ok'\n", encoding="utf-8")
    state = tmp_path / ".claude-repo-mem"
    memory = state / "memory/decision.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# Existing decision\n\nKeep the health endpoint small.\n", encoding="utf-8")
    retained = memory.read_bytes()
    runner = CliRunner()
    first = runner.invoke(engine_main, ["index", "--root", str(tmp_path), "--no-embed"])
    assert first.exit_code == 0, first.output
    settings = Settings.for_repo(tmp_path)
    with connect(settings.db_path) as connection:
        before = [tuple(row) for row in connection.execute("SELECT id, content_hash FROM unit ORDER BY id")]
    second = runner.invoke(main, ["memory", "index", "--root", str(tmp_path), "--no-embed"])
    assert second.exit_code == 0, second.output
    with connect(settings.db_path) as connection:
        after = [tuple(row) for row in connection.execute("SELECT id, content_hash FROM unit ORDER BY id")]
    assert before == after and before
    assert memory.read_bytes() == retained
    assert not (tmp_path / ".code-mem-graph").exists()
    setup = runner.invoke(main, ["init", "--root", str(tmp_path), "--repository-id", "compatibility-test"])
    assert setup.exit_code == 0, setup.output
    assert (tmp_path / ".project-knowledge/map.yaml").exists()
    assert settings.db_path.exists() and memory.read_bytes() == retained


def test_memory_cli_delegates_help_and_errors():
    runner = CliRunner()
    help_result = runner.invoke(main, ["memory", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "install-hooks" in help_result.output and "index" in help_result.output
    bad = runner.invoke(main, ["memory", "unknown-command"])
    assert bad.exit_code != 0 and "No such command" in bad.output

from pathlib import Path
from click.testing import CliRunner
from code_mem_graph.engine.cli import main
from code_mem_graph.engine.config import Settings
from code_mem_graph.engine.db.connection import init_db
from code_mem_graph.engine.indexer.orchestrator import full_reindex


def test_doctor_reports_layers_and_counters(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f(): pass\n")
    s = Settings.for_repo(tmp_path); init_db(s.db_path)
    full_reindex(s, embedder=None)
    runner = CliRunner()
    res = runner.invoke(main, ["doctor", "--root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "by_layer" in res.output
    assert "counters" in res.output.lower()


def test_doctor_reports_t2_coverage(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f(): pass\n")
    s = Settings.for_repo(tmp_path); init_db(s.db_path)
    full_reindex(s, embedder=None)
    runner = CliRunner()
    res = runner.invoke(main, ["doctor", "--root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "t2_coverage" in res.output

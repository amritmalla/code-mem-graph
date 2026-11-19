from pathlib import Path
import yaml
from click.testing import CliRunner
from code_mem_graph.engine.cli import main
from code_mem_graph.engine.config import Settings
from code_mem_graph.engine.db.connection import init_db
from code_mem_graph.engine.indexer.orchestrator import full_reindex


def test_bench_prints_summary(tmp_path: Path):
    (tmp_path / "auth.py").write_text("def login_user(): pass\n")
    s = Settings.for_repo(tmp_path); init_db(s.db_path)
    full_reindex(s, embedder=None)
    fixture = tmp_path / "q.yaml"
    fixture.write_text(yaml.safe_dump({
        "queries": [{"q": "login", "expect_header_substring": "login"}],
    }))
    runner = CliRunner()
    res = runner.invoke(main, [
        "bench", "--root", str(tmp_path),
        "--fixture", str(fixture), "--no-embed",
    ])
    assert res.exit_code == 0, res.output
    assert "recall@5" in res.output
    assert "1/1" in res.output

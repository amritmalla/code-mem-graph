from pathlib import Path

import pytest

# Source snapshots contain upstream tests as evidence. Never import or execute
# them while collecting this project's regression suite.
collect_ignore = ["fixtures"]


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A temp directory standing in for a working repo."""
    (tmp_path / ".claude-repo-mem").mkdir()
    return tmp_path


@pytest.fixture
def db_path(tmp_repo: Path) -> Path:
    return tmp_repo / ".claude-repo-mem" / "db.sqlite"

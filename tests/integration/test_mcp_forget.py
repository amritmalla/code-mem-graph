import json
from pathlib import Path
import pytest
from code_mem_graph.engine.config import Settings
from code_mem_graph.engine.db.connection import init_db
from code_mem_graph.engine.memory.writer import remember
from code_mem_graph.engine.tools.forget import handle, tool_schema


@pytest.fixture
def settings(tmp_repo: Path):
    s = Settings.for_repo(tmp_repo)
    init_db(s.db_path)
    return s


def test_schema():
    s = tool_schema()
    assert s.name == "forget"
    assert "handle" in s.inputSchema["properties"]


@pytest.mark.asyncio
async def test_handle_tombstones(settings):
    r = remember(settings, fact="x", scope="x")
    out = await handle(settings, {"handle": r.handle})
    payload = json.loads(out[0].text)
    assert payload.get("ok") is True


@pytest.mark.asyncio
async def test_handle_unknown_returns_error(settings):
    out = await handle(settings, {"handle": "memory://decision/zzzzzzzzzzzz"})
    payload = json.loads(out[0].text)
    assert "error" in payload

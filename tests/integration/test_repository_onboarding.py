"""A real local MCP client, without a model, exercises generated setup configuration."""

import json
import os
from pathlib import Path
import shutil
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest

from code_mem_graph.onboarding import initialize_repository


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", ["python_service", "typescript_app"])
async def test_generated_configuration_serves_external_root_over_stdio(
    tmp_path, fixture
):
    root = tmp_path / "external repository with spaces"
    shutil.copytree(
        Path(__file__).parents[1] / "fixtures" / "knowledge" / fixture, root
    )
    setup = initialize_repository(root, repository_id="external-stdio")
    config = setup["mcp_config"]["mcpServers"]["code-mem-graph"]
    env = os.environ.copy()
    # Preserve this test interpreter's dependencies for the subprocess. Clean
    # installation is a separate P06 gate, not asserted by this source test.
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sys.path if path)
    parameters = StdioServerParameters(
        command=config["command"], args=config["args"], env=env, cwd=str(tmp_path)
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "orient",
                "context",
                "knowledge_item",
            }
            for tool, extra in [
                ("orient", {}),
                ("context", {"task": "Find documented implementation and checks"}),
                ("knowledge_item", {"item_id": "project/purpose"}),
            ]:
                result = await session.call_tool(
                    tool, {"budget": 20000, "encoding": "cl100k_base", **extra}
                )
                assert not result.isError
                payload = json.loads(result.content[0].text)
                assert payload["repository_id"] == "external-stdio"
                assert payload["knowledge_ref"]["path"] == ".project-knowledge/map.yaml"

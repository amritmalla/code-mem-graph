"""Exercise a non-editable installation from outside the source tree, without a model.

Run with the fresh environment's Python: python -I scripts/verify_install.py.
The explicit tool cap below is a test allowance, not a production default.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PATH"] = sysconfig.get_path("scripts") + os.pathsep + env.get("PATH", "")
    return env


def cli(cwd: Path, *arguments: str, expected: int = 0) -> str:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "code_mem_graph.cli", *arguments],
        cwd=cwd, env=clean_environment(), text=True, capture_output=True, timeout=120,
    )
    if result.returncode != expected:
        raise RuntimeError(f"CLI {arguments}: expected {expected}, got {result.returncode}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


async def read_tools(config: dict, cwd: Path, repository_id: str) -> list[str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=config["command"], args=config["args"], env=clean_environment(), cwd=str(cwd),
    )
    async with asyncio.timeout(120):
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert names == {"orient", "context", "knowledge_item"}, names
                for name, extra in (
                    ("orient", {}),
                    ("context", {"task": "Find the health implementation and documented checks"}),
                    ("knowledge_item", {"item_id": "project/purpose"}),
                ):
                    result = await session.call_tool(
                        name, {"budget": 20000, "encoding": "cl100k_base", **extra},
                    )
                    assert not result.isError, result
                    payload = json.loads(result.content[0].text)
                    assert payload["repository_id"] == repository_id, payload
                    assert payload["knowledge_ref"]["path"] == ".project-knowledge/map.yaml"
    return sorted(names)


def exercise_project(work: Path, language: str) -> dict:
    print(f"Checking installed workflow for {language}...", file=sys.stderr, flush=True)
    root = work / f"{language} project with spaces"
    root.mkdir()
    readme = root / "README.md"
    readme.write_text("# Parcel service\n\nThis project supplies parcel health checks.\n\n"
                      "Run the documented checks before changing the health endpoint.\n", encoding="utf-8")
    if language == "python":
        (root / "pyproject.toml").write_text(
            '[project]\nname = "parcel-service"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n',
            encoding="utf-8",
        )
        (root / "app.py").write_text('def health():\n    return {"status": "ok"}\n', encoding="utf-8")
    else:
        (root / "package.json").write_text(json.dumps({
            "name": "parcel-dashboard", "private": True,
            "scripts": {"check": "tsc --noEmit"}, "engines": {"node": ">=20"},
        }), encoding="utf-8")
        (root / "app.ts").write_text('export const health = () => ({status: "ok"});\n', encoding="utf-8")
    instructions = root / "AGENTS.md"
    instructions.write_text("# Practices\n\nKeep health checks deterministic.\n", encoding="utf-8")
    client = root / ".mcp.json"
    client.write_text('{"mcpServers":{"unrelated":{"command":"example"}}}\n', encoding="utf-8")
    originals = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
    repository_id = f"release-{language}"
    setup = json.loads(cli(work, "init", "--root", str(root), "--repository-id", repository_id, "--json"))
    assert setup["ready_for_reads"], setup
    cli(work, "doctor", "--root", str(root), "--json")
    assert all(path.read_bytes() == content for path, content in originals.items())
    config = json.loads(cli(work, "mcp-config", "--root", str(root)))["mcpServers"]["code-mem-graph"]
    assert Path(config["command"]).resolve() == Path(sys.executable).resolve()
    assert config["args"] == ["-m", "code_mem_graph.cli", "serve", "--root", str(root)]
    tools = asyncio.run(read_tools(config, work, repository_id))
    # Changed evidence must be caught, then refreshed without editing source/client files.
    readme.write_text(readme.read_text(encoding="utf-8") + "\nThe health endpoint is read-only.\n", encoding="utf-8")
    stale = json.loads(cli(work, "doctor", "--root", str(root), "--json", expected=1))
    assert "source_evidence_needs_refresh" in stale["issues"], stale
    cli(work, "knowledge", "refresh", "--root", str(root))
    refreshed = json.loads(cli(work, "doctor", "--root", str(root), "--json"))
    assert refreshed["ready_for_reads"], refreshed
    # Memory indexing needs no embedding/model when --no-embed is requested.
    memory = root / ".claude-repo-mem/memory/decision.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# Existing decision\n\nKeep the health endpoint small.\n", encoding="utf-8")
    saved = memory.read_bytes()
    knowledge = (root / ".project-knowledge/map.yaml").read_bytes()
    cli(work, "memory", "init", "--root", str(root), "--no-watch")
    entry = json.loads(client.read_text(encoding="utf-8"))["mcpServers"]
    assert entry["unrelated"] == {"command": "example"}
    assert entry["code-mem-graph-memory"]["command"] == "code-mem-graph"
    assert entry["code-mem-graph-memory"]["args"][:2] == ["memory", "serve"]
    cli(work, "memory", "index", "--root", str(root), "--no-embed")
    cli(work, "memory", "doctor", "--root", str(root))
    assert memory.read_bytes() == saved
    assert (root / ".project-knowledge/map.yaml").read_bytes() == knowledge
    assert instructions.read_bytes() == originals[instructions]
    return {"language": language, "tools": tools, "source_and_client_preserved": True,
            "staleness_and_refresh": True, "memory_without_embeddings": True,
            "saved_memory_and_knowledge_preserved": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save a JSON verification receipt")
    arguments = parser.parse_args()
    spec = importlib.util.find_spec("code_mem_graph")
    assert spec is not None and spec.origin
    package = Path(spec.origin).resolve()
    assert package.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()), (
        "Use a non-editable installation in the selected Python environment", package,
    )
    assert importlib.util.find_spec("claude_repo_mem") is None
    distribution = importlib.metadata.distribution("code-mem-graph")
    scripts = {entry.name: entry.value for entry in distribution.entry_points if entry.group == "console_scripts"}
    assert scripts == {"code-mem-graph": "code_mem_graph.cli:main"}, scripts
    executable = Path(sysconfig.get_path("scripts")) / ("code-mem-graph.exe" if os.name == "nt" else "code-mem-graph")
    with tempfile.TemporaryDirectory(prefix="code-mem-graph installed check ") as directory:
        work = Path(directory).resolve()
        result = subprocess.run([str(executable), "--help"], cwd=work, env=clean_environment(),
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0 and "mcp-config" in result.stdout, result
        cli(work, "memory", "--help")
        projects = [exercise_project(work, language) for language in ("python", "typescript")]
    receipt = {"distribution": "code-mem-graph", "version": distribution.version,
               "python": platform.python_version(), "platform": platform.platform(),
               "package_path": str(package), "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
               "non_editable_installation": True, "removed_namespace_absent": True,
               "console_entry_verified": True, "projects": projects, "model_calls": 0,
               "dependencies": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
               "limits": ["Does not download or run an embedding model or exercise paid providers.",
                          "The explicit MCP response cap is a test allowance, not a product default."]}
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

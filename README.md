# code-mem-graph

Build source-backed repository knowledge that engineers can read and AI agents
can query. Give a new agent a cited introduction, relevant project practices and
task context, with freshness, review status and missing information visible.

This is a source release candidate. The standalone GitHub release and package
registry publication have not been completed.

## What it provides

- **Project knowledge:** observable facts from supported manifests, source
  locations and linked documentation, plus explanations supplied by maintainers.
- **Agent access:** read-only MCP tools `orient`, `context` and `knowledge_item`.
- **Task navigation:** relevant knowledge, supported local relationships,
  application boundaries, applicable directory guidance and source/test candidates.
- **Upkeep:** detect changed evidence, refresh generated facts, preserve curated
  contributions and review exact claim versions.
- **Readable output:** `.project-knowledge/map.yaml` and a cited Markdown view
  inside the target repository.

Python runs the tool; the target project need not be Python. Current knowledge
extraction supports selected Python/Node manifests, documentation and static
source relationships. Undocumented architecture, decisions and team processes
need maintainer input. Coverage is partial for other stacks.

## Install from source

Use Python 3.11–3.13. From a prepared checkout, create a fresh environment:

```bash
python -m venv .venv
# Activate it: .venv\Scripts\Activate.ps1 on Windows, or source .venv/bin/activate.
python -m pip install .
code-mem-graph --help
```

Installation currently includes the indexing and local-embedding dependencies.
Knowledge commands do not download an embedding model or call a model API.
Tokenizer data may download on the first budgeted read.

## Use with your project

Run from any directory, substituting your project's absolute path and stable ID:

```bash
code-mem-graph init --root /absolute/project --repository-id my-project
code-mem-graph doctor --root /absolute/project
code-mem-graph mcp-config --root /absolute/project
```

`init` writes the knowledge map and reading view in the target project. It
preserves source, agent instructions and client settings. Open
`.project-knowledge/README.md` for the engineer view.

Merge the configuration printed by `mcp-config` into your agent client's
`mcpServers` settings, retaining other entries. It pins the installed Python
executable and target root. Reconnect the client, then ask the agent to orient
itself, request task context and inspect cited sources before changing code.

Each MCP read requires a positive token cap and an explicit supported tokenizer
encoding. Choose the cap from the agent's available context allocation. There
are no arbitrary 1k/3k defaults; automatic optimal sizing is not calibrated.

After changing source or documentation:

```bash
code-mem-graph doctor --root /absolute/project
code-mem-graph knowledge refresh --root /absolute/project
code-mem-graph knowledge status --root /absolute/project
```

Fresh evidence does not establish complete architecture coverage or prove an
agent's change is correct. Inspect omissions, gaps, conflicts and pending reviews.
Repository statements remain evidence under the agent's existing instructions.

See the [complete setup guide](docs/getting-started.md)
and [maintainer contribution workflow](docs/contributions.md).

## Indexing and durable memory

The indexing/retrieval engine is available under the same executable:

```bash
code-mem-graph memory init --root /absolute/project --no-watch
code-mem-graph memory index --root /absolute/project --no-embed
code-mem-graph memory doctor --root /absolute/project
```

Omit `--no-embed` to generate embeddings using the configured backend; the default
local model downloads on first use. Memory, task and handoff data retain their
`.claude-repo-mem` paths and existing environment settings. The Python package
`claude_repo_mem` and old executable aliases have been removed; imports now use
`code_mem_graph`, with the memory engine in `code_mem_graph.engine`.

Read the [migration guide](docs/migration.md),
[memory usage guide](docs/usage.md), [MCP tool reference](docs/tools.md) and
[engine architecture](docs/architecture.md).

## Documentation

- [Project documentation](docs/README.md)
- [Development and contribution guide](CONTRIBUTING.md)
- [Package verification](docs/verification.md)
- [Changelog](CHANGELOG.md)

## License and origin

MIT. See [LICENSE](LICENSE). This project builds on
[claude-repo-mem](https://github.com/amritmalla/claude-repo-mem); its original
license and authorship are retained.

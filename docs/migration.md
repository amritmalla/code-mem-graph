# Migrating from claude-repo-mem

The distribution and executable are `code-mem-graph`. All Python implementation
lives under `code_mem_graph`, with indexing/retrieval/memory under
`code_mem_graph.engine`. The old Python package and executable aliases are absent.

Use a fresh environment when installing the renamed package. An independently
installed old distribution can leave its own modules/scripts behind; installing
this project does not uninstall that distribution.

## Commands and client settings

Use `code-mem-graph memory <command>` for memory/indexing operations. Repository
knowledge commands such as `init`, `doctor`, `context` and `serve` are top-level.
The two servers have different roles:

| Server | Command | Client key |
|---|---|---|
| Read-only project knowledge | `code-mem-graph serve --root /absolute/project` | `code-mem-graph` |
| Full memory/indexing tools | `code-mem-graph memory serve --root /absolute/project --no-watch` | `code-mem-graph-memory` |

Regenerate knowledge client settings with `code-mem-graph mcp-config --root ...`.
Memory initialization generates the new memory key/command while preserving
unrelated entries. Inspect and remove obsolete entries deliberately; existing
client files are not rewritten merely by installing this package.

Existing indexing hooks need `code-mem-graph memory index --no-embed`. Review any
custom hook contents before reinstalling a hook explicitly.

## Stored data

Saved memories, indexes, tasks and handoffs retain `.claude-repo-mem` paths and
existing `CLAUDE_REPO_MEM_*` settings. Knowledge maps retain `.project-knowledge`.
These are data/configuration names, not a Python compatibility namespace.
Do not delete saved data as part of changing imports or client commands.

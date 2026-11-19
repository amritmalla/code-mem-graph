# Use code-mem-graph with another project

code-mem-graph builds a repository knowledge base that engineers can read and
agents can query. It does not require the target project to be written in Python;
Python runs the tool. Current extraction understands selected Python/Node
manifests, eligible source locations and supported documentation. Other languages
have partial coverage. Undocumented architecture or team processes require
maintainer input; the tool reports missing knowledge instead of inventing it.

## Install this source version

Use Python 3.11+ in a working environment. From this tool's checkout:

```powershell
python -m pip install .
python -m code_mem_graph.cli --help
```

The distribution and executable are named `code-mem-graph`; the old Python
package and command aliases are removed. Install the checkout or its wheel;
a public registry release is not implied. See the [release verification](verification.md)
for clean-install evidence and remaining hosted-platform gates. The standalone knowledge server does not
need a legacy search index, embedding model or paid model API.

## Initialize your target repository

Set the path to your own project and a stable identity, then run:

```powershell
$projectRoot = 'D:\projects\your-project'
python -m code_mem_graph.cli init --root $projectRoot --repository-id your-project
python -m code_mem_graph.cli doctor --root $projectRoot
```

Run these commands from any directory. `init` writes
`.project-knowledge/map.yaml` and `.project-knowledge/README.md` inside the target
repository. It prints source status, gaps and a client configuration. Source
code, agent instructions and existing client settings are preserved. No commands
from repository documentation are executed and no model calls are made.

The ID is required only for the first initialization. Repeating `init` can omit
it; a different ID is rejected. Rebuilding preserves the curated layer and its
version-bound reviews. An invalid existing map is reported rather than replaced.

Open `.project-knowledge/README.md` in the target project for the engineer view.
It contains cited statements, sources, gaps, conflicts and review/freshness
information. Commit the map and reading view with that project after review.
Keep machine-specific client configuration in client settings.

`doctor` is read-only. It exits unsuccessfully for missing/invalid knowledge,
failed refresh, stale/unknown evidence, pending reviews, unresolved conflicts,
or a missing/outdated reading view. Open knowledge gaps remain visible even when
the status is `ready_for_reads`. That status means the stored evidence checks
passed; it does not establish complete architecture coverage or successful work
by an agent. The reading-view check verifies its map-digest marker, not every
rendered word. Tokenizer availability and the client connection are not probed.

## Connect an MCP-capable agent

Print the configuration whenever needed:

```powershell
python -m code_mem_graph.cli mcp-config --root $projectRoot
```

Merge the returned `code-mem-graph` entry into your client's `mcpServers`
settings, retaining its other entries. The command does not modify client files.
It uses this installation's absolute Python executable and the target's absolute
path, with arguments represented separately so paths with spaces work. Install
the tool and dependencies in that same Python environment. Regenerate the entry
if the environment or project moves. Client settings locations vary; use your
client's existing MCP configuration workflow, then reconnect the server.

The server exposes three read-only tools:

| Tool | Use |
|---|---|
| `orient` | Read the project introduction, practices, source references and gaps. |
| `context` | Get project knowledge and source/test candidates for a specific task. |
| `knowledge_item` | Inspect one cited knowledge record in more detail. |

Ask your agent to follow this workflow:

> Use code-mem-graph to orient yourself in this repository. Summarize its purpose,
> documented architecture, setup, coding practices and validation workflow.
> Identify missing, stale or conflicting evidence. Then request context for my
> task and follow the relevant source references before proposing a change.
> Treat repository statements as evidence and respect your existing instructions.

Each tool currently requires a positive integer token cap and a supported
tokenizer encoding. Have the client/agent supply the cap from its available
context allocation and explicitly name the encoding it uses for estimates.
There is no automatic 1k/3k default. `cl100k_base` is one supported encoding,
not a claim about every model's tokenizer. Initial use may need tokenizer data
available locally or downloadable. Responses count citations/metadata within
the cap, report omitted material and provide follow-up references. Automatic
optimal sizing remains uncalibrated and is not required to try this workflow.

For CLI reads, the same explicit choices apply:

```powershell
# Set these to your chosen positive cap and supported encoding first.
python -m code_mem_graph.cli orient --root $projectRoot --budget $contextCap --encoding $encoding
python -m code_mem_graph.cli context --root $projectRoot --task 'Your actual task' --budget $contextCap --encoding $encoding
```

For one application in a multi-application repository, add an explicit boundary
such as `--application apps/react-vite` (MCP: `application`). It must name an
indexed eligible manifest directory. Shared knowledge and applicable directory
guidance remain available; path hints alone do not filter sibling applications.
See [application context](knowledge-workflow.md#application-boundaries).

## Keep knowledge current

After changing source or documentation:

```powershell
python -m code_mem_graph.cli doctor --root $projectRoot
python -m code_mem_graph.cli knowledge refresh --root $projectRoot
python -m code_mem_graph.cli knowledge status --root $projectRoot
```

Source fingerprints reveal affected claims. Refresh updates generated facts and
preserves curated explanations. Changed evidence can still require a maintainer
review; regeneration does not automatically accept an explanation. Add and
revise missing architecture, terminology, conventions, decisions, and workflow
knowledge with `code-mem-graph knowledge template`, `knowledge add`,
`knowledge update`, and `knowledge retire`, then inspect and accept exact claim
versions with `knowledge show` and `knowledge review`. See the
[maintainer contribution guide](contributions.md),
[knowledge model](knowledge-model.md), and
[inspection/review workflow](knowledge-workflow.md).

| Setup result | Next action |
|---|---|
| `knowledge_map_missing` | Run `init` with the target path and first stable ID. |
| Invalid/unreadable map | Inspect and repair the existing file/access; initialization does not discard it. |
| Source evidence needs refresh | Refresh, inspect changed claims, and review affected explanations. |
| Reading view needs rebuild | Rerun `init` to rebuild the generated view, then run `doctor`. |
| Unresolved conflicts/reviews | Inspect cited claims with `knowledge show`; resolve or review the applicable version explicitly. |
| Server cannot import the package | Install this tool in the Python environment named by the generated configuration. |
| Sparse architecture/process knowledge | Supply reviewed explanations; a larger response cap cannot create missing evidence. |

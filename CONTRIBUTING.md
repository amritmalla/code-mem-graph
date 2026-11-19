# Contributing to code-mem-graph

## Development setup

Use Python 3.11–3.13 and a fresh virtual environment (Python 3.14+ is not currently supported due to dependency wheel availability):

```bash
# Windows (PowerShell with py launcher):
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# Linux / macOS:
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Alternatively, `uv` provides significantly faster installation on Windows:

```powershell
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

### Windows requirements and notes

- Supported Python versions: 3.11, 3.12, and 3.13 (64-bit). Ensure your default `python` does not resolve to an unsupported prerelease (e.g., Python 3.14).
- Binary wheels: Pre-built binary wheels on PyPI have been verified locally on Windows 11 x86_64 for CPython 3.11.15, 3.12.9, and 3.13.5 without requiring Visual Studio or C++ build tools. Other Windows architectures (e.g., ARM64) or non-standard Python distributions have not been verified and may require a C++ build toolchain or lack compatible binary wheels.
- PowerShell script execution: If `.venv\Scripts\Activate.ps1` is restricted by execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` or invoke `.venv\Scripts\python.exe` directly.

Optional provider integrations use the `openai`, `voyage` and `anthropic` extras.
The knowledge-only workflow does not require an embedding model or provider key.
The base installation currently includes local-embedding dependencies.

## Repository layout

| Location | Responsibility |
|---|---|
| `src/code_mem_graph/cli.py` | Public command groups and shared workflow entry points. |
| `src/code_mem_graph/onboarding.py`, `onboarding_cli.py` | Target initialization, readiness and client settings. |
| `src/code_mem_graph/knowledge/` | Model, source selection/extraction, maps, task context, freshness and contributions. |
| `src/code_mem_graph/mcp.py` | The three read-only knowledge tools. |
| `src/code_mem_graph/engine/` | Indexing, SQLite retrieval, durable memory, tasks, handoffs and provider adapters. |
| `src/code_mem_graph/evaluation/` | Recorded-run evaluation, accounting and controlled-trial tooling. |
| `tests/` | Unit/integration tests and fictional source fixtures. |
| `tests/fixtures/evaluation/repository-onboarding.yaml` | Active regression/discovery corpus; freeze exact source before any new trial. |
| `scripts/` | Distribution, installed-package and source-bundle verification utilities. |
| `docs/` | Current setup, architecture, contracts, usage and maintenance information. |

Knowledge maps and memory indexes are separate stores. Read the
[architecture](docs/architecture.md) and [knowledge model](docs/knowledge-model.md)
before changing their behavior.

## Coding practices

- Keep modules focused and tool handlers thin; put subsystem logic in the owning module.
- Use `from __future__ import annotations` and type annotations for public functions.
- Keep CLI help and MCP descriptions accurate: agents use them to choose actions.
- Preserve source evidence, review bindings, existing memory and client settings.
- Keep knowledge reads read-only; make contribution/review/refresh writes explicit.
- Report incomplete evidence and unsupported behavior rather than inventing facts.
- Add regression coverage for meaningful behavior changes; avoid tests that only repeat implementation details.

Knowledge MCP tools are registered in `mcp.py`. Memory tool handlers live in
`engine/tools/` and are registered in `engine/server.py`.

## Tests and checks

```bash
python -m pytest -q
python -m ruff check src/code_mem_graph scripts
```

Pytest uses `src` on its import path, automatic asyncio handling and skips tests
marked `slow` by default. Those tests can download models or be expensive;
run them deliberately with `python -m pytest -m slow`.
Fixture source under `tests/fixtures/` is evidence and is excluded from test collection.
Preserve the canonical LF bytes specified in `.gitattributes` for fingerprint fixtures.

See [verification](docs/verification.md) for package-content and clean-install checks.

## What belongs in Git

Commit source, tests/fixtures, package metadata, license, current reference docs,
verification scripts and the active evaluation corpus. Documentation should
explain current repository behavior, contracts and operating practices.

Keep roadmaps, task lists, implementation plans, progress reports, old experiment
write-ups and study drafts under ignored `.local/`. Do not add them to public
`docs/`. Local client settings, secrets, environments, caches, trial outputs and
this tool checkout's generated knowledge/memory state are also ignored.
The top-level `evaluation/` directory is local output/history; the active corpus
is versioned with the test fixtures. Keep `scripts/` in Git for reproducible
build, installation and source-bundle checks.
Ignore rules here do not prevent consumers from reviewing and versioning a
knowledge map in their own target repository.

## Distribution verification

Update the changelog/version when preparing a release. Build and validate locally:

```bash
python -m build
python -I scripts/check_distributions.py --dist dist
python -m twine check --strict dist/*
```

Use fresh environments for installed-wheel and source-distribution checks.
Automated CI (`.github/workflows/ci.yml`) runs on push and pull requests, covering
Python 3.11, 3.12, and 3.13 on Ubuntu and Windows with lint, default tests, package
building, distribution checks, and installed-package verification without source-tree imports.
Model downloads and slow watcher tests run separately via `.github/workflows/model-download.yml`.
Neither workflow exercises live paid providers.
Publishing and credentials are separate from building a source bundle; the scripts and CI do not publish.

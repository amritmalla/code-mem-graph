# Verification

## Source checks

From a development environment installed with `.[dev]`:

```bash
python -m pytest -q
python -m ruff check src/code_mem_graph scripts
```

The default suite excludes slow/model-download tests. Use
`python -m pytest -m slow` deliberately when checking those backends.
The active corpus at `tests/fixtures/evaluation/repository-onboarding.yaml` and
fictional fixtures stay in Git because tests use them. The top-level
`evaluation/` directory holds local records and is entirely ignored.

## Distribution contents

```bash
python -m build
python -I scripts/check_distributions.py --dist dist --output build/receipts/packages.json
python -m twine check --strict dist/*
```

Use an output directory with exactly one wheel and one source distribution.
The content checker verifies canonical package bytes, required release metadata,
license and console entry. Use a current Twine; old validators can reject modern
metadata emitted by the build backend.

## Installed-package workflow

Create a fresh virtual environment, install the built wheel with its declared
dependencies, and use that environment's Python:

```bash
python -m pip check
python -I scripts/verify_install.py --output build/receipts/wheel.json
```

Repeat in another fresh environment installed from the source distribution.
The script rejects editable/source-tree imports and the removed namespace. It
checks console commands, target paths containing spaces, initialization, real
stdio MCP reads on Python/TypeScript sample projects, staleness/refresh, memory
indexing without embeddings, and preservation of source/client/saved data.

These checks do not execute an embedding model or a paid-provider call. Tokenizer
data may download. The explicit response cap is a test allowance, not a default
sizing policy. A passing sample workflow does not establish general agent quality,
architecture completeness or token savings.

## Source bundles

```bash
python -I scripts/prepare_release.py --output .local/source-review
```

The output directory must be new. The allowlist includes source/tests, current
public documentation, release scripts, package/license files and the active
evaluation corpus. Git ignore rules are respected. Local state, client settings,
planning records and experiment artifacts are excluded.

The ZIP and external hash manifest identify exact working-tree bytes; they do not
contain Git history or publish anything. A repository without commits or a remote
is supported: those metadata fields are recorded as absent. Original foundation
authorship/license remain attributed separately from the current Git repository.

## Automated CI workflows

Automated verification is defined in `.github/workflows/ci.yml` and `.github/workflows/model-download.yml`:

- **Default CI (`.github/workflows/ci.yml`)**:
  - Triggers on push and pull requests targeting `master` and `main`, as well as manual `workflow_dispatch`.
  - Minimal permissions (`contents: read`) and explicit job timeouts.
  - Requires zero secrets or paid provider services.
  - **Jobs**:
    1. `lint`: Dedicated job on `ubuntu-latest` (Python 3.12) running `ruff check src/code_mem_graph scripts`.
    2. `test`: Matrix job covering Ubuntu (`ubuntu-latest`) and Windows (`windows-latest`) across Python 3.11, 3.12, and 3.13, running `pip check` and default `pytest -q` (slow tests deselected by default).
    3. `distribution`: Dedicated job on `ubuntu-latest` building wheel and source distributions, running `scripts/check_distributions.py` and `twine check --strict`, and archiving package artifacts.
    4. `installed-package`: Matrix job covering Ubuntu and Windows across Python 3.11, 3.12, and 3.13, downloading the built distribution artifacts, creating separate fresh virtual environments (`.venv-wheel`, `.venv-sdist`), installing the wheel and sdist with non-editable installs, and running `pip check` and `scripts/verify_install.py` without source-tree imports.
- **Model download and slow-test workflow (`.github/workflows/model-download.yml`)**:
  - Separate manual `workflow_dispatch` workflow for tests marked `slow`: local embedding-model execution and watcher integration tests. Kept isolated from default required checks; it does not exercise live paid providers.

### Local reproduction of CI checks

To run the exact checks locally:

```bash
# 1. Lint
python -m ruff check src/code_mem_graph scripts

# 2. Default tests (slow tests excluded by default configuration)
python -m pytest -q

# 3. Distribution build and content validation
python -m build
python -I scripts/check_distributions.py --dist dist --output build/receipts/packages.json
python -m twine check --strict dist/*

```

Use new environment directories for the following checks. On Linux/macOS (Bash):

```bash
set -euo pipefail
# Fresh-environment installed wheel verification
uv venv --seed --python 3.12 .local/venv-wheel-check
uv pip install --python .local/venv-wheel-check/bin/python dist/code_mem_graph-*.whl
.local/venv-wheel-check/bin/python -m pip check
.local/venv-wheel-check/bin/python -I scripts/verify_install.py --output build/receipts/wheel.json

# Fresh-environment installed sdist verification
uv venv --seed --python 3.12 .local/venv-sdist-check
uv pip install --python .local/venv-sdist-check/bin/python dist/code_mem_graph-*.tar.gz
.local/venv-sdist-check/bin/python -m pip check
.local/venv-sdist-check/bin/python -I scripts/verify_install.py --output build/receipts/sdist.json
```

On Windows (PowerShell), explicitly resolve archive paths; native commands do not
receive expanded file wildcards:

```powershell
$wheelArchive = @(Get-ChildItem -Path dist/code_mem_graph-*.whl -File)
$sdistArchive = @(Get-ChildItem -Path dist/code_mem_graph-*.tar.gz -File)
if ($wheelArchive.Count -ne 1 -or $sdistArchive.Count -ne 1) {
    throw 'Expected exactly one wheel and one source distribution in dist.'
}
uv venv --seed --python 3.12 .local/venv-wheel-check
if ($LASTEXITCODE -ne 0) { throw 'Wheel environment creation failed.' }
uv pip install --python .local/venv-wheel-check/Scripts/python.exe $wheelArchive[0].FullName
if ($LASTEXITCODE -ne 0) { throw 'Wheel installation failed.' }
.local/venv-wheel-check/Scripts/python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Wheel dependencies are inconsistent.' }
.local/venv-wheel-check/Scripts/python.exe -I scripts/verify_install.py --output build/receipts/wheel.json
if ($LASTEXITCODE -ne 0) { throw 'Wheel verification failed.' }

uv venv --seed --python 3.12 .local/venv-sdist-check
if ($LASTEXITCODE -ne 0) { throw 'Source environment creation failed.' }
uv pip install --python .local/venv-sdist-check/Scripts/python.exe $sdistArchive[0].FullName
if ($LASTEXITCODE -ne 0) { throw 'Source installation failed.' }
.local/venv-sdist-check/Scripts/python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Source dependencies are inconsistent.' }
.local/venv-sdist-check/Scripts/python.exe -I scripts/verify_install.py --output build/receipts/sdist.json
if ($LASTEXITCODE -ne 0) { throw 'Source verification failed.' }
```

If the local checkout has no configured git remote (`git remote -v` is empty), hosted execution remains pending until pushed to an authorized remote repository. Local verification on the host platform confirms script and configuration correctness but does not substitute for hosted multi-platform matrix runs.

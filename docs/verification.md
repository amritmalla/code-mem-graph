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

There is currently no hosted CI workflow in this checkout. Run relevant local
checks before sharing changes and verify other platforms explicitly when needed.

# Maintainer contributions and upkeep

Maintainers can add, revise, review, and retire architecture explanations,
decisions, domain vocabulary, conventions, and workflow knowledge using small
YAML or JSON contribution files instead of editing `.project-knowledge/map.yaml`
directly. Evidence fingerprints and upstream dependency digests are captured
internally through the existing source-policy and version-binding rules.

## Commands

All write operations remain explicit CLI/API actions. MCP tools (`orient`,
`context`, `knowledge_item`) stay read-only.

| Command | Purpose |
|---|---|
| `code-mem-graph knowledge template` | Print or write a starter YAML/JSON contribution file, or export an existing curated item with `--from-item <id>` without hashes. |
| `code-mem-graph knowledge add <file>` | Create a new unreviewed curated contribution. Supports `--dry-run` and optional `--expected-map-digest`. Rejects duplicate IDs and generated-item overwrites. |
| `code-mem-graph knowledge update <file>` | Revise an existing active curated contribution. Requires `--expected-digest`, supports `--dry-run` and `--expected-map-digest`, and invalidates previous acceptance. |
| `code-mem-graph knowledge retire <item_id>` | Retire an active curated contribution while preserving its historical record and review. Requires `--expected-digest`. |
| `code-mem-graph knowledge show <item_id>` | Inspect the stored record, `content_digest`, `map_digest`, freshness/review assessment, current evidence, dependents, and addressed gap status. |
| `code-mem-graph knowledge status` | Inspect source freshness, pending `review_queue`, effective gap status, and conflict status. |
| `code-mem-graph knowledge review <item_id>` | Record an explicit named, version-bound `accepted` or `rejected` review (and optionally `--refresh-evidence` for curated items). |
| `code-mem-graph knowledge refresh` | Rebuild generated facts while preserving all curated contributions, reviews, and durable gap associations. |

## Contribution file format

A contribution file is a strict YAML or JSON mapping (`extra="forbid"`). Do not
include `review`, `origin`, `generator`, `lifecycle`, or manual hashes.

```yaml
id: architecture/ledger-purity
topic: architecture # overview | architecture | domain | conventions | workflow | decision
title: Ledger calculation stays pure
statement: >-
  calculate_balance performs deterministic summation without external I/O so
  unit tests stay isolated.
basis: inferred # documented | inferred | maintainer (default: maintainer)
author: priya-maintainer
component_ids:
  - project
facets: [] # purpose | stack | entrypoints | components | constraints | workflow
onboarding: [] # purpose | connection | prerequisites | responsibilities | conventions | validation
critical: false
evidence:
  - path: src/billing/ledger.py # shorthand string "src/billing/ledger.py" also works
    start_line: 1
    end_line: 3
  - kind: directory
    path: src
    recursive: true
dependencies:
  - project/purpose # existing active item IDs; current versions are captured internally
addresses_gap_ids:
  - knowledge/architecture-rationale # existing gap IDs from `knowledge status`
```

Epistemic distinctions are preserved:

- `documented`: explicit statement grounded in cited sources; requires direct `evidence`.
- `inferred`: interpretation derived from sources or existing claims; requires `evidence` or `dependencies`.
- `maintainer`: explicitly attributed maintainer assertion; may cite `evidence`, `dependencies`, or neither. A source-free maintainer assertion retains `freshness: unknown` (`no_verifiable_source`) even after acceptance and never closes a gap as verified support.

## End-to-end workflow: create → inspect → review → update → refresh

Initialize a target repository (or use an already initialized one):

```powershell
$projectRoot = 'D:\projects\your-project'
code-mem-graph init --root $projectRoot --repository-id your-project
code-mem-graph knowledge status --root $projectRoot
```

### 1. Create a contribution from a starter template

Generate a starter file, edit it to cite real paths and existing IDs, preview
with `--dry-run`, then add it:

```powershell
code-mem-graph knowledge template `
  --id architecture/service-boundary `
  --topic architecture `
  --author $env:KNOWLEDGE_AUTHOR `
  --basis inferred `
  --gap knowledge/architecture-rationale `
  --output contrib.yaml

# Preview captured fingerprints, dependency digests, and resulting claim digest without writing.
code-mem-graph knowledge add contrib.yaml --root $projectRoot --dry-run

# Persist atomically under the build lock and rebuild .project-knowledge/README.md.
code-mem-graph knowledge add contrib.yaml --root $projectRoot
```

New contributions always start `unreviewed`. Generated items and `conflict-resolution/`
IDs cannot be overwritten; duplicate IDs, excluded/missing/non-portable paths,
invalid line ranges, unknown component/dependency/gap IDs, and cycles are
rejected without modifying `.project-knowledge/map.yaml` or `README.md`.

### 2. Inspect before acceptance

Inspect the saved version, captured evidence, and gap status:

```powershell
$inspected = code-mem-graph knowledge show architecture/service-boundary --root $projectRoot | ConvertFrom-Json
code-mem-graph knowledge status --root $projectRoot
```

Until explicitly accepted, an `inferred` or `maintainer` contribution appears in
`review_queue` and `cautions` (not usable `items`), and any gap listed in
`addresses_gap_ids` remains effectively `open`.

### 3. Explicitly review the inspected version

Bind a named maintainer review to that exact `content_digest`:

```powershell
code-mem-graph knowledge review architecture/service-boundary `
  --root $projectRoot `
  --expected-digest $inspected.content_digest `
  --reviewer $env:KNOWLEDGE_REVIEWER `
  --status accepted `
  --note "Checked against cited source lines."
```

Once accepted with fresh evidence and unblocked dependencies, the contribution
enters usable `items` in `orient` and `context` (including `compact` and
`focused` views) and updates effective gap status in `knowledge status`,
`doctor`, `.project-knowledge/README.md`, and bundle `gaps`.

### 4. Update with version protection

Export the current curated item without hashes, edit the file, and update
against the inspected `content_digest`:

```powershell
$current = code-mem-graph knowledge show architecture/service-boundary --root $projectRoot | ConvertFrom-Json
code-mem-graph knowledge template `
  --root $projectRoot `
  --from-item architecture/service-boundary `
  --output contrib-v2.yaml

# Edit contrib-v2.yaml, then preview or apply with the exact expected digest.
code-mem-graph knowledge update contrib-v2.yaml `
  --root $projectRoot `
  --expected-digest $current.content_digest `
  --dry-run

code-mem-graph knowledge update contrib-v2.yaml `
  --root $projectRoot `
  --expected-digest $current.content_digest
```

An outdated `--expected-digest` or `--expected-map-digest`, or a concurrent edit
to `.project-knowledge/map.yaml`, is rejected without overwriting newer
knowledge. Successful edits reset `review.status` to `unreviewed`, invalidate
the previous acceptance, rebuild `README.md`, and reopen addressed gaps until
the new version is explicitly reviewed again.

Updates retain an existing item's `relationships` and `command_checks`, even
though those fields are not editable through the compact contribution format.
Both previews and writes reject removal of evidence required by that metadata,
leaving the map and reading view unchanged. Inspect the full retained metadata
with `knowledge show`; an ordinary prose edit still invalidates acceptance.

### 5. Refresh and verify staleness / gap reopening

After repository files change:

```powershell
code-mem-graph knowledge refresh --root $projectRoot
code-mem-graph knowledge status --root $projectRoot
```

`refresh` updates generated facts and preserves all curated contributions,
reviews, and `addresses_gap_ids` links. If a cited source file or upstream
dependency changed, `status` and `show` report the contribution as `stale` and
in `review_queue`, and any gap it addressed visibly reopens (`effective_status:
open`) until the maintainer inspects the change and either updates the
contribution or explicitly recaptures evidence with `knowledge review
--refresh-evidence`. Rejecting or retiring (`knowledge retire`) the contribution
likewise leaves the gap visibly open.

## Durable gap addressing and compatibility

- Gap links are stored in the optional `addresses_gap_ids` list on `KnowledgeItem`.
  Generated extraction outputs in `generated.gaps` are never overwritten, and
  unrelated gaps stay open.
- Effective gap resolution combines stored `resolved_by` support with curated
  items that name the gap in `addresses_gap_ids` and are currently `active`,
  `accepted`, `fresh`, and not `review_required`.
- Empty `addresses_gap_ids` lists are omitted from serialization, preserving
  historical claim, review, and map digests. Older maps remain readable without
  migration.
- Existing conflict handling (`resolve-conflict`), source-free assertion
  semantics, documented-command warnings, and complete-payload CLI/MCP
  size accounting are unchanged.

## Limitations

- Contributions are authored and reviewed explicitly via CLI/API; there is no
  GUI, automated approval, or semantic verification of prose accuracy.
- Source evidence hashes whole files conservatively under source policy v2;
  non-allowlisted or ignored files cannot be cited.
- `knowledge retire` keeps retired curated items as immutable history; to
  publish revised guidance afterward, add a new contribution ID.

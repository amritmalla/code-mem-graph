# Knowledge model

The knowledge contract defines evidence, review, freshness and safe refresh.
See [setup](getting-started.md), [knowledge workflows](knowledge-workflow.md)
and [maintainer contributions](contributions.md) for operations.

## Document and graph

A `KnowledgeMap` is portable YAML or JSON with `schema_version: 1` and an explicit
repository identity. It has two independently owned layers: `generated` and
`curated`. Each contains components, knowledge items, gaps, and conflicts. Local
absolute roots and Git HEAD are not the authority for source freshness.

IDs are explicit, stable repository-scoped identities such as `indexing/pipeline`.
Curated identities need not follow file locations; moving a cited file should
update their evidence rather than rename the knowledge item. Automatic directory
and manifest facts have deterministic location identities; a move retires the
old record rather than guessing semantic equivalence. Item IDs, component IDs, gap IDs, and conflict IDs
are each unique across both layers. An ID cannot move between layers through an
automatic refresh. Components contain names, paths, a parent, and a lifecycle;
responsibility explanations belong in cited knowledge items. Component parent
edges and item evidence-dependency edges must form acyclic graphs.

An item contains its topic, optional orientation facets, statement, component references, origin, evidence
basis, provenance, sources, versioned dependencies, criticality, review, and
lifecycle. Topics cover overview, architecture, domain, conventions, workflow,
and decisions. Inferred dependencies reference both an item ID and its content
digest, so replacing an upstream claim cannot silently validate old explanations.
Do not use evidence-dependency edges for runtime call graphs, which may be cyclic.

The optional `relationships` field carries typed source/target location edges:
`documented_location` associates a cited chapter with a named existing location;
`static_import` records supported static local module resolution. Both endpoints
require direct evidence. These edges may be cyclic and are separate from
versioned knowledge dependencies. Empty lists are omitted to preserve historical
digests; adding/changing an edge changes its claim version. See
[relationships](knowledge-workflow.md#relationships-and-command-checks) for resolution, selection and limits.

The optional `command_checks` field records a documented package command's
manager/name, directory/manifest, scope basis and `declared_script`,
`script_not_declared` or `scope_unknown` status. Known comparisons require both
document and manifest evidence. Empty lists preserve historical digests;
populated comparisons change claim versions. Absence of a script key is not a
runtime failure or an automatic authority conflict. See
[command comparisons](knowledge-workflow.md#relationships-and-command-checks).

The optional `onboarding` annotation identifies practical evidence for `purpose`,
`connection`, `prerequisites`, `responsibilities`, `conventions`, or `validation`. Generated annotations use
documented excerpts; maintainer annotations follow the ordinary review rules.
They describe evidence coverage rather than proof that a task has been answered.
Empty annotations are omitted from serialization to preserve historical
claim/review/map digests. Adding an annotation changes the claim digest and
requires a new review where review is required. Older maps remain readable;
refresh is needed to extract the new annotations.

The optional `addresses_gap_ids` field on a `KnowledgeItem` references existing
gap IDs (generated or curated) that the claim addresses without overwriting
`generated.gaps`. Empty lists are omitted from serialization to preserve
historical claim/review/map digests; adding or changing gap references changes
the claim digest and invalidates prior review. Unknown or duplicate gap IDs fail
validation. See [maintainer contributions](contributions.md).

## Evidence, review, and freshness are independent

| Dimension | Values | Meaning |
|---|---|---|
| Origin | generated / maintainer | Who owns this item and may replace it. Generated items name a generator/version; maintainer items name their author. |
| Basis | observed / documented / inferred / maintainer | Directly observable source fact, explicit source statement, interpretation, or explicitly attributed maintainer assertion. Documentation is not proof of implementation. |
| Review | unreviewed / accepted / rejected | A named reviewer assessed this exact claim version. Accepted does not mean current. |
| Freshness | fresh / stale / unknown | Current allowed evidence matches, a known dependency changed/disappeared, or validation could not establish freshness. Computed, never trusted from a saved flag. |
| Lifecycle | active / retired | The claim is still produced, or retained as historical evidence for existing links. Retired claims are stale. |

Observed/documented claims require direct evidence. Inferences require direct
evidence or versioned knowledge dependencies and require review before being
treated as accepted guidance. Source-free maintainer assertions are permitted,
explicitly attributed, and have unknown verifiable freshness. Review never turns
an unsupported inference into an observed fact. Rejected items and dependencies
requiring review remain blockers even if their source fingerprints match.

The content digest includes the statement, topic, title, evidence, dependencies,
component references, provenance, and criticality. Review and lifecycle are
excluded so retirement preserves the historical review. Any other change creates
a new claim version. A scored review must name a reviewer and match that digest;
copying an old review onto revised content fails validation. Review recording is
an explicit API operation, not cryptographic proof of reviewer identity.

Freshness checks hash current working-tree bytes rather than relying on commit
IDs or modification times. Known changes dominate unknown signals. Staleness,
unknown validation, and review blockers propagate through evidence dependencies.
An item in an unresolved conflict requires review while its fingerprint-based
freshness remains a separate result.

## Source policy v2

Evidence paths must be canonical repository-relative POSIX paths. Absolute,
drive, UNC, traversal, backslash, ambiguous Windows device, and nonportable path
forms are rejected. `.` is allowed only for a directory. Optional file line
ranges are one-based, inclusive, paired, and checked against the captured file.
The digest covers the **whole file**, conservatively invalidating the claim when
any byte changes, even outside the cited range.

This version supports evidence hashing for:

- Source files: Python, JavaScript/JSX, TypeScript/TSX, Java, Go, and Rust.
- Documentation: Markdown and `.markdown` files.
- Explicit manifest names: `pyproject.toml`, `package.json`, `tsconfig.json`,
  `Cargo.toml`, `go.mod`, and `pom.xml`, including nested packages.
- Workflow configuration: immediate `.github/workflows/*.yml` and `*.yaml` files.

Hash support does not imply that semantic extractors for these formats exist.
Other JSON/YAML/TOML files are excluded. Dot-environment files, other hidden paths,
Git metadata, credentials directories, environments, dependencies, build outputs,
and legacy/new cache directories are excluded. A future versioned knowledge
document directory, `.project-knowledge`, is also excluded to avoid self-evidence.
Nested `.gitignore` and root `.knowledgeignore` patterns additionally exclude
files and directories. They cannot override the fixed allowlist or exclusions.
The policy revision intentionally invalidates v1 source evidence; it must be
recaptured before being considered fresh. Existing claim reviews are not
automatically rebound to revised evidence.

Descendant symlinks and Windows reparse points/junctions are rejected, including
links pointing inside the root. The caller-selected repository root is resolved
once. This is a best-effort local filesystem check; there is no concurrent-write
snapshot or operating-system race isolation. It does not read linked content
outside the root, shell out, or extract credential values into the map.

Directory evidence fingerprints sorted eligible file paths, either immediately
inside the selected directory or recursively. It detects eligible additions,
deletions, and renames. It does not hash file contents or empty directories;
claims about contents require separate file evidence. Excluded files do not
invalidate membership. An unreadable or linked relevant subtree prevents a
complete scan and produces unknown freshness, never a partial successful digest.
Every source records the policy digest; policy changes invalidate old evidence.

## Gaps, conflicts, and safe refresh

Gaps are first-class missing knowledge, not unsupported statements. They name a
topic and optional components. A resolved gap references supporting item IDs;
those items' current evidence/review assessments still need to be inspected.
Curated contributions can also durably associate themselves with an existing gap
through `addresses_gap_ids` without modifying `generated.gaps`. Effective gap
reporting treats a gap as resolved when either its stored `resolved_by` items
have usable support or an active, explicitly `accepted`, `fresh`, unblocked
contribution addresses it. Unreviewed, stale, rejected, retired, conflicting, or
source-free (`unknown` freshness) contributions leave the gap visibly `open`.
Conflicts reference distinct competing claims. Resolving one requires an
explanation, resolver, and exact claim-version digests. A resolution becomes
ineffective if those versions or their evidence change. Both sides remain in
the map; the assessment returns effective unresolved conflict IDs.

`refresh_generated` is a pure, validated merge:

1. Preserve the curated layer in full, including its components, gaps, conflicts,
   explanations, attribution, dependencies, and reviews.
2. Accept a complete new generated layer. Discard generator-supplied review
   authority. Retain a prior review only for an unchanged complete claim version.
3. Retain removed generated components/items as retired so curated references
   remain resolvable. Keep old gaps/conflicts until explicitly resolved. Retired
   records are historical, not recommended guidance.
   Preserve named conflict resolutions for the same competing claim identities,
   retaining their original version bindings; changed versions remain unresolved
   in current assessments. Generator input cannot create review/resolution authority.
4. Validate the combined graph and reject ownership collisions or dangling/cyclic
   references. A rejected merge leaves the original map unchanged.
5. On failed refresh, keep the previous layers and fingerprints, record failure,
   and mark otherwise unchanged generated knowledge unknown. Known stale evidence
   remains stale. A successful refresh clears the failure flag but does not
   refresh old fingerprints unless new evidence was actually captured.

The build workflow now publishes `.project-knowledge/map.yaml` atomically and
writes a derived Markdown view. The merge itself remains a pure in-memory API.
Contract fixtures use `map.yaml` at their fixture root and remain manually
assembled examples rather than extractor output.

## Inspect and validate

From a development checkout with the dependencies installed:

```powershell
python -m code_mem_graph.cli knowledge schema
python -m code_mem_graph.cli knowledge validate tests/fixtures/knowledge/python_service/map.yaml --root tests/fixtures/knowledge/python_service
python -m code_mem_graph.cli knowledge validate tests/fixtures/knowledge/typescript_app/map.yaml --root tests/fixtures/knowledge/typescript_app
```

The installed `code-mem-graph` command accepts the same arguments. Validation
without `--root` checks structure, references, and review bindings only; it does
not claim source freshness. The schema command exports the structural JSON
Schema; graph, review-digest, and filesystem invariants additionally require the
runtime validator. A valid map may intentionally contain stale, unreviewed,
missing, or conflicting knowledge; validation is not product task acceptance.

The fixtures are fictional and manually assembled from the public model API.
The Python service includes observed facts, a derived explanation, a synthetic
maintainer decision, and missing process knowledge. The TypeScript app includes
an intentional README/manifest runtime conflict. They are contract examples,
not generated output or evidence of agent-performance improvement.

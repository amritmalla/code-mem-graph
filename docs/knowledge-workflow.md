# Task context and knowledge upkeep

Agent knowledge reads are local and read-only. Response caps and tokenizer encodings are explicit; automatic sizing is uncalibrated.

## Get task-specific context

After [building knowledge](getting-started.md), choose the caller's chosen
cap and tokenizer, then request context:

```powershell
code-mem-graph context --root . --task "Add a freshness check" --path src/code_mem_graph/knowledge/freshness.py --budget $env:PROJECT_CONTEXT_LIMIT --encoding $env:PROJECT_CONTEXT_ENCODING
```

The cap and encoding are required; there is no fixed default or `auto` policy.
Repeat `--path`, `--component`, or `--item` for multiple hints. Paths are portable
repository-relative files/directories and can name a proposed new file. Component
and item hints must refer to known IDs; unknown IDs are errors. An excluded or
unknown file path is reported unavailable rather than read outside the source policy.

## Application boundaries

For a manifest-backed application, add `--application apps/react-vite` (or the
MCP `application` field). This filters candidates and independent peer-package
claims; shared knowledge and required dependency/conflict proof remain available.
Paths remain hints when no boundary is supplied. Applicable ancestor/local
AGENTS/CLAUDE guidance is mandatory and has explicit inclusion/freshness status.
New applicable instructions require refresh before the read can continue.
See [application boundaries and directory guidance](knowledge-workflow.md#application-boundaries)
for eligibility, broad directory hints and cross-boundary proof behavior.

## Relationships and command checks

Saved document-location and static-import relationships can add source candidates
that filename matching misses. Candidates reference supporting knowledge IDs;
stale/review-blocked relationships cannot introduce new candidates. Selection
takes one incident-record step from the original locations/selected guidance,
with another step available by record inspection. Test locations remain
candidates, not proof of behavioral coverage. See
[source-backed relationships](knowledge-workflow.md#relationships-and-command-checks) for supported resolution and
the distinction from knowledge-evidence dependencies or runtime graphs.

Documented package-command comparisons expose declared/absent manifest keys and
unknown application scope. Applicable missing-key warnings are mandatory;
`coverage.documented_commands` retains omitted/unavailable IDs. Comparisons do
not prove execution or authorize spelling-based replacements. See
[scoped command comparisons](knowledge-workflow.md#relationships-and-command-checks).

The selector uses explicit hints, recorded text, and eligible filenames.
It splits camel/snake/path names, drops common English function/action words,
and orders matches by explicit item, path evidence, component scope, and term
overlap. This is a deterministic, explainable heuristic, not semantic inference
or a calibrated relevance probability. A no-match task stays explicitly partial.

Matched components bring relevant declared practices and workflows; repository
constraints and critical root guidance remain applicable. Evidence dependencies,
derived explanations/decisions, competing claims, and resolution explanations
remain connected. These are **knowledge-evidence dependencies**, not runtime
import/call/data-flow edges. Component boundaries are as precise as the saved
knowledge, which may initially only identify source directories.

The response separates usable `items` from `cautions`, reports selection reasons,
coverage/omissions, and adds `source_candidates`. Candidates are current allowed
file paths with fingerprints. A `test_candidate` is based on filename/location
conventions; it is not proof that a test validates the requested change. Source
bodies are not copied into context. Inspect cited files or use existing source
retrieval tools for the details. Incomplete source discovery is reported unknown;
available saved knowledge can still be returned with partial status.

Task text is identified by a digest rather than echoed in every response.
Candidate paths and optional facts are packed under the cap. No arbitrary result
count cuts off the selection. Complete dependency/conflict groups travel together;
if mandatory groups cannot fit, only status and references are returned. Size
includes citations, status metadata, the size field, and the complete MCP result
envelope. Counts remain explicitly named-tokenizer estimates.

Task-context coverage requires applicable constraints/workflow plus matching
usable knowledge. Candidate filenames alone do not establish task understanding.
`complete` describes packing and recorded coverage, not task success or semantic
completeness. Global gaps, unknown rationale, and stale historical claims can
still be visible. Nothing in this operation executes repository commands.

## MCP tools

The knowledge server exposes three read-only tools; the memory server
registers these alongside its existing tools:

| Tool | Required arguments | Optional hints |
|---|---|---|
| `orient` | `budget`, `encoding` | None |
| `context` | `task`, `budget`, `encoding` | `application`, `paths`, `component_ids`, `item_ids` |
| `knowledge_item` | `item_id`, `budget`, `encoding` | `application` |

`knowledge_item` inspects a cited record and its dependency/conflict group without
turning its inspection text into another broad search. It includes critical
repository guidance when applicable. Use it for follow-up on returned IDs.
No embeddings, LLM synthesis, or implicit paid evaluations run on these paths.
The root remains pinned at server startup. Review, contribution, and refresh
writes are CLI/API operations rather than agent-callable MCP mutation tools.

## Add, update, and retire maintainer contributions

Use small YAML or JSON contribution files instead of hand-editing
`.project-knowledge/map.yaml`:

```powershell
code-mem-graph knowledge template --output contrib.yaml
code-mem-graph knowledge add contrib.yaml --root . --dry-run
code-mem-graph knowledge add contrib.yaml --root .
code-mem-graph knowledge template --root . --from-item $itemId --output contrib-v2.yaml
code-mem-graph knowledge update contrib-v2.yaml --root . --expected-digest $inspected.content_digest
code-mem-graph knowledge retire $itemId --root . --expected-digest $inspected.content_digest
```

Fingerprints and dependency digests are captured internally through source
policy v2. Contributions start `unreviewed`; editing a curated item requires its
exact `content_digest` and invalidates prior acceptance. Use `addresses_gap_ids`
to address existing gaps durably across refreshes without overwriting generated
extraction results. See [maintainer contributions](contributions.md).

## Inspect and refresh

```powershell
code-mem-graph knowledge status --root .
code-mem-graph knowledge status --root . --check
code-mem-graph knowledge refresh --root .
code-mem-graph knowledge show project/purpose --root .
```

Status recomputes source fingerprints and transitive applicability, lists review
requirements, and shows effective conflict/gap status. `--check` exits with code
1 for stale/unknown active evidence, a pending review queue, or failed refresh.
Open knowledge gaps alone are not a failure of this source-health check; it is
not a completeness or task-acceptance gate.

Refresh requires an existing map, reruns deterministic extraction, and reports
added, changed, retired, and reactivated generated claims. Curated text, evidence,
and reviews are preserved exactly. Changed source facts lose old generated
approvals; authored explanations with changed dependencies remain stale rather
than receiving new fingerprints silently. Additions, deletions, renames, ignore
changes, and failed refreshes use the source/dependency checks already defined by
the [knowledge model](knowledge-model.md). There is no watcher or incremental
extractor yet: refresh is explicit and extraction is a complete snapshot.

Refresh uses the inspected map version under the build lock. A newer map version
causes a retry error, not an overwrite. Map replacement is atomic, with the same
optimistic editor protection and separately written Markdown view as the build
workflow. Extraction failure retains prior material and records failure when
storage permits. Failed source access is unknown, never a successful partial scan.

Show returns the full record, content digest, assessment, stored/current source
fingerprints, and IDs of dependent records. It does not display source bodies or
assert that changed evidence still supports the explanation.

## Review an exact claim version

Inspect the item and cited source changes first. Then use the inspected digest
and your reviewer identity explicitly:

```powershell
$inspected = code-mem-graph knowledge show $itemId --root . | ConvertFrom-Json
code-mem-graph knowledge review $itemId --root . --expected-digest $inspected.content_digest --reviewer $env:KNOWLEDGE_REVIEWER --status accepted
```

Set `$itemId` to the record you reviewed and the reviewer variable to your name.
`--status rejected` records a rejection, including for an outdated claim. The
operation refuses an item version different from the one inspected. A review
is a named maintainer attestation, not cryptographic authentication.

Acceptance requires current evidence and reviewed evidence dependencies. Rebuild
generated facts when sources change; the review command cannot rebind their
fingerprints. For a **curated** explanation that you have checked still holds,
`--refresh-evidence` explicitly recaptures its existing file/directory references
and current dependency versions before binding the new review. It preserves the
statement. It cannot repair moved citations, invented statements, missing files,
or unsupported source formats. Edit those references/explanations deliberately
and reset/re-record the review through the model API. No automatic recapture or
review happens during context reads or refresh.

Source-free maintainer assertions can receive explicit review but retain unknown
verifiable freshness. Approval does not turn them into observed facts or silently
make them usable guidance. Claims based on rejected or blocked dependencies must
have those dependencies addressed first.

## Explain a conflict

Inspect `knowledge status`, both competing claims, and their current sources.
Resolve only against that exact map version, with a named explanation:

```powershell
code-mem-graph knowledge resolve-conflict $conflictId --root . --expected-map-digest $inspectedMapDigest --reviewer $env:KNOWLEDGE_REVIEWER --explanation $resolutionText
```

Resolution requires fresh evidence for every competing claim. Its explanation
and claim-version bindings are preserved on unchanged rebuilds, and the explicit
human explanation is also saved as curated decision knowledge. Context includes
the explanation with the competing claims even after resolution, avoiding two
contradictory statements presented without their interpretation. New claim
versions or changed evidence make the old resolution ineffective; it does not
approve the new versions. Generated input cannot confer review/resolution authority.

Historical conflicts are retained conservatively. If a claim is retired or the
conflict no longer applies, inspect and deliberately edit/remove that historical
conflict entry in the versioned map rather than pretending its old evidence is
fresh. Authored resolution text remains available in the curated layer. The
`conflict-resolution/` item namespace is reserved for this workflow; repeated
explicit resolutions retain prior authored records as retired history.

See [verification](verification.md) for the source and installed-package checks.

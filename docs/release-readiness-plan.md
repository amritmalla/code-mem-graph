# Release readiness implementation plan

Created 2026-11-19 following a sampled technical assessment of this checkout.
This is planned work, not a statement that the checks or improvements are complete.
The user explicitly requested this plan in `docs/`; it is an exception to the
general contribution policy that keeps planning records under `.local/`.

## Objective and workflow

Make code-mem-graph reproducibly verifiable, easier to install, and supported by
evidence of practical value. Antigravity implements one step at a time; Codex
independently verifies that step before dependent work proceeds.

Keep changes isolated and preserve existing user work. Inspect current source and
repository instructions before implementation: the assessment is a starting point,
not proof of a defect. Do not publish packages, deploy, or start paid trials as
part of this plan without separate authorization.

## Assessment baseline

- The architecture separates deterministic knowledge maps from the memory engine.
- Source evidence, freshness propagation, version-bound reviews, and atomic file
  replacement are existing strengths to preserve.
- The sampled checkout contained 120 test modules and package verification scripts.
- Ruff passed during assessment. A passing test suite was not established: the
  existing environment referenced a missing Python installation, an alternate
  invocation encountered dependency-loading errors, and a subsequent attempt was
  stopped after making no visible progress. These are not confirmed product bugs.
- Hosted CI was absent. The base dependency list included local embeddings.
- Read-time filesystem work is a performance hypothesis requiring measurement.
- The reviewed active evaluation corpus targeted this repository; broader benefits
  were not established by the evidence reviewed.

## Step 1: Restore a reproducible development environment

### Implementation

Inspect setup and create a fresh development environment from declared dependencies.
Run the documented default tests and lint. Diagnose failures before changing
application behavior. Document supported setup and required Windows dependencies.
Do not reuse or commit the existing virtual environment, disable failing tests,
or change assertions merely to obtain a passing result.

### Acceptance criteria

- A fresh installation succeeds using documented commands.
- Default tests and lint finish successfully, or each remaining failure has an
  explanation and reproduction steps; unresolved failures remain verification blockers.
- Slow/model-download tests remain explicitly separate.
- Record Python version, installation command, test counts, skips, and results.

### Verification handoff

Environment instructions, dependency changes, and complete check summaries.

## Step 2: Automate tests and package verification

### Implementation

Add GitHub Actions using the existing verification scripts. Cover Python 3.11–3.13
on Windows and Linux. Run lint, default tests, distribution-content checks, and
strict package metadata validation. Verify wheel and source-distribution installation
in separate clean environments without source-tree imports.
Keep paid-provider and model-download checks separate from required default checks.

### Acceptance criteria

- Required jobs pass on the supported matrix.
- Installed-package checks exercise the actual built artifacts.
- Default CI requires no secrets or paid services.
- Documentation explains required checks and local reproduction.

### Verification handoff

Workflow files, successful run links when available, and package verification
receipts. Mark remote execution pending if unavailable; local checks do not prove
that the hosted matrix passes.

## Step 3: Establish an independent effectiveness benchmark

### Implementation

Extend existing evaluation tooling rather than creating a second framework.
Freeze a protocol comparing ordinary repository exploration/search against the
same agent with code-mem-graph knowledge tools. Use several independently selected
repositories across supported Python, TypeScript, and mixed-project layouts,
pinning exact revisions. Include onboarding, locating code/tests, and tracing
change impact. Keep model, prompts, environment, and resource limits equivalent.
Separate development examples from held-out evaluation tasks.

### Acceptance criteria

- Record scoring criteria and stopping rules before trials.
- Hide review criteria from the agent performing the task.
- Measure correctness, important omissions, citation accuracy, elapsed time,
  tool calls, and total model token usage.
- Account for setup/build costs separately from repeated-use costs.
- Preserve failures and interrupted runs.
- Distinguish observed results from unsupported generalizations.
- Obtain explicit budget approval before paid trials; elapsed time is not approval.

### Verification handoff

Protocol, pinned corpus, scoring rubric, and a small pilot demonstrating recording
and scoring. Use an unpaid recording/scoring smoke test if paid execution is not
authorized; identify effectiveness trials as pending.

## Step 4: Make knowledge-only installation lightweight

### Implementation

Audit imports and dependencies. Make the three-tool knowledge workflow usable
without embedding dependencies or memory-only parsers. Introduce clearly named
optional extras for memory and local embeddings. Preserve memory capabilities
when the appropriate extras are installed. Missing optional dependencies must
produce actionable errors. Update installation and migration guidance.

### Acceptance criteria

- Clean base installation supports initialization, doctor, refresh, and all three
  knowledge MCP tools.
- Base installation excludes `sentence-transformers` and its heavyweight
  transitive dependencies.
- Memory and local-embedding installation variants each have smoke coverage;
  actual model execution remains a separately identified check.
- CLI help and knowledge commands do not eagerly import optional engine dependencies.
- Package checks and CI cover supported installation variants.

### Verification handoff

Dependency/import changes, installed dependency lists for each variant, and
clean-install results.

## Step 5: Measure read performance before optimizing

### Implementation

Add reproducible benchmarks for build, refresh, `orient`, `context`, and
`knowledge_item` on small, medium, and large repositories. Include citations to
different ranges of the same file. Measure latency, file reads, directory scans,
and peak memory where practical. Separate cold and warm runs. Record repository
size and environment details.

### Acceptance criteria

- Benchmarks require no paid services and baselines are reproducible.
- Results identify measured bottlenecks.
- Proposed targets are justified by measurements.
- Ordinary correctness tests avoid fragile wall-clock thresholds.

### Verification handoff

Benchmark commands, baseline results, and a prioritized optimization proposal.

## Step 6: Optimize confirmed bottlenecks without weakening freshness

### Implementation

Implement only optimizations supported by step 5. Evaluate sharing source bytes,
hashes, and ignore-rule parsing within one request. Preserve range validation,
repository boundaries, exclusions, and freshness semantics. Avoid persistent
caching unless its invalidation behavior is explicitly designed and tested.

### Acceptance criteria

- Equivalent repository state yields equivalent knowledge and status results.
- Source edits and ignore-rule changes are visible on subsequent requests.
- Different ranges of a file remain independently validated.
- Relevant correctness tests pass.
- Comparable before/after benchmarks show improvement and disclose tradeoffs.
- If measurements justify no optimization, document the evidence rather than
  introducing speculative complexity.

### Verification handoff

Focused implementation diff, regression coverage, and comparable benchmark results.

## Step 7: Simplify onboarding and prepare the release candidate

### Implementation

Make knowledge the primary documented workflow. Present memory, embeddings,
task planning, and handoffs as optional extensions. Explain extraction coverage,
maintainer contributions, freshness, and explicit response budgets.
Run step 3's independent evaluation against the final candidate, subject to
approved costs. Complete package and installation checks for the final revision.

### Acceptance criteria

- One guide takes a new user through installation, initialization, MCP connection,
  knowledge queries, and refresh.
- Optional features and installation requirements are easy to find.
- Quality, speed, and token-saving claims match measured evidence.
- Remaining limitations and release blockers are explicit.
- Final package checks and required CI pass.
- Pending effectiveness trials are disclosed and do not become marketing claims.

### Verification handoff

Final documentation, evaluation report, artifact receipts, and exact candidate
revision. Publishing remains a separate authorized action after verification.

## Required handoff for every step

Provide:

1. What changed and why.
2. Commit identifier, if committed, or precise changed-file list.
3. Commands run and results, including counts and skips where relevant.
4. Acceptance criteria satisfied, with supporting evidence.
5. Known failures, limitations, and pending checks.

Use existing ignored local-output locations for verbose logs and receipts; do not
commit environments, secrets, model caches, or generated trial output. Do not mark
a step verified based solely on the implementer's report.

Ask Codex: "Verify step N against docs/release-readiness-plan.md. Inspect the
implementation, rerun relevant checks, and report blockers before I proceed."

Resolve verification blockers before starting dependent work. Work through steps
1–7 in order; step 6 specifically depends on measured results from step 5.

"""Offline broader-study commitments, planning and immutable draft registration.

This module has no model/execution adapter. Freezing a protocol authorizes no
calls. Unknown scientific inputs stay unknown rather than becoming defaults.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, variance
from typing import Literal

from pydantic import Field, model_validator

from code_mem_graph.knowledge.models import Digest, Identity
from code_mem_graph.serialization import fingerprint

from .acceptance import AcceptanceRegistration
from .controls import codex_capabilities
from .experiment import Candidate, Experiment, load_experiment
from .models import Contract

SCENARIOS = (
    "orientation",
    "architecture",
    "local-change",
    "cross-component-change",
    "conventions",
    "workflow",
    "missing-evidence",
    "stale-evidence",
    "conflicting-evidence",
)
SPLITS = ("calibration", "validation", "heldout")


class CandidateRecipe(Contract):
    id: Identity
    representation: Literal["detailed", "compact"]
    detail: Literal["full", "focused"]
    boundary_rule: Literal["max_preserved_evidence_then_min_wire_tokens"] = (
        "max_preserved_evidence_then_min_wire_tokens"
    )


def select_candidate(experiment: Experiment, recipe: CandidateRecipe) -> Candidate:
    """No criteria, expected locations, answer reviews or outcomes enter selection."""
    candidates = [
        entry
        for entry in experiment.candidates
        if entry.payload["policy"].get("representation", "detailed")
        == recipe.representation
        and entry.payload["policy"].get("detail", "full") == recipe.detail
        and not entry.payload["omitted"]["mandatory_claims"]
        and not entry.payload["omitted"]["conflicts"]
    ]
    if not candidates:
        raise ValueError("no registered bundle boundary matches the recipe")
    return min(
        candidates,
        key=lambda entry: (
            -(len(entry.payload["items"]) + len(entry.payload["cautions"])),
            sum(entry.payload["omitted"].values()),
            entry.tokens,
            entry.id,
        ),
    )


class CaseCommitment(Contract):
    schema_version: Literal[1] = 1
    id: Identity
    split: Literal["calibration", "validation", "heldout"]
    domain: Literal["service", "frontend"]
    repository_family: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    source_digest: Digest
    artifact_sha256: Digest
    artifact_digest: Digest
    artifact_kind: Literal["discovery", "change"]
    task_id: str
    task_prompt_digest: Digest
    scenarios: list[str] = Field(min_length=1)
    exposure: Literal["inspected", "curator_only"]
    curator_reference: str = Field(min_length=1)
    synthetic: bool
    partial_snapshot: bool
    affected_components: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_scope(self):
        if set(self.scenarios) - set(SCENARIOS) or len(self.scenarios) != len(
            set(self.scenarios)
        ):
            raise ValueError("unsupported or duplicate scenario labels")
        change = bool(set(self.scenarios) & {"local-change", "cross-component-change"})
        if change != (self.artifact_kind == "change"):
            raise ValueError(
                "actual change scenarios require an acceptance registration, not a read-only context task"
            )
        if (
            "cross-component-change" in self.scenarios
            and len(set(self.affected_components)) < 2
        ):
            raise ValueError(
                "cross-component work requires at least two explicitly reviewed component identities"
            )
        if self.split != "calibration" and self.exposure != "curator_only":
            raise ValueError("inspected material belongs only to calibration")
        return self


def commit_case(
    path: Path,
    *,
    case_id: str,
    split: str,
    domain: str,
    repository_family: str,
    scenarios: list[str],
    curator_reference: str,
    exposure: str,
    synthetic: bool,
    partial_snapshot: bool,
    kind: str = "discovery",
    affected_components: list[str] | None = None,
) -> CaseCommitment:
    raw = path.read_bytes()
    if kind == "discovery":
        artifact = load_experiment(path)
        snapshot, task = artifact.snapshot, artifact.suite.task(artifact.task_id)
        digest = artifact.digest
        if task.category == "change":
            raise ValueError("read-only experiments cannot register actual change work")
    elif kind == "change":
        artifact = AcceptanceRegistration.model_validate_json(raw)
        snapshot, task = artifact.snapshot, artifact.suite.task(artifact.task_id)
        digest = artifact.digest
    else:
        raise ValueError("unsupported study artifact kind")
    if not any(criterion.critical for criterion in task.criteria):
        raise ValueError("a study task must declare critical quality criteria")
    return CaseCommitment(
        id=case_id,
        split=split,
        domain=domain,
        repository_family=repository_family,
        repository_id=task.repository_id,
        source_digest=snapshot.source_digest,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_digest=digest,
        artifact_kind=kind,
        task_id=task.id,
        task_prompt_digest=fingerprint(task.prompt),
        scenarios=scenarios,
        curator_reference=curator_reference,
        exposure=exposure,
        synthetic=synthetic,
        partial_snapshot=partial_snapshot,
        affected_components=affected_components or [],
    )


class ReviewerPolicy(Contract):
    primary_model: str = "gpt-6-luna"
    primary_profile_digest: Digest | None = None
    selection_basis: str = "Maintainer-selected GPT 6 Luna; use a fresh independently recorded review session."
    coverage: Literal["every_answer_all_units_and_criteria"] = (
        "every_answer_all_units_and_criteria"
    )
    blinding: Literal["withhold_approach_cost_and_previous_opinions"] = (
        "withhold_approach_cost_and_previous_opinions"
    )
    disagreements: Literal["retain_and_block_no_automatic_majority"] = (
        "retain_and_block_no_automatic_majority"
    )
    max_review_model_calls_per_answer: int = Field(default=1, gt=0)
    human_sample: Literal["not_registered_no_human_validation_claim"] = (
        "not_registered_no_human_validation_claim"
    )


class StudyResources(Contract):
    future_preparation_model_call_limit: int | None = Field(default=None, ge=0)
    future_model_call_limit: int | None = Field(default=None, gt=0)
    future_provider_token_limit: int | None = Field(default=None, gt=0)
    provider_tokens_per_call_ceiling: int | None = Field(default=None, gt=0)
    max_task_model_turns: int | None = Field(default=None, gt=0)
    max_task_tool_calls: int | None = Field(default=None, gt=0)
    max_task_elapsed_ms: float | None = Field(default=None, gt=0)
    max_request_payload_tokens: int | None = Field(default=None, gt=0)
    max_response_payload_tokens: int | None = Field(default=None, gt=0)
    max_total_visible_payload_tokens: int | None = Field(default=None, gt=0)
    monetary_ceiling: str | None = None
    currency: str | None = None
    pricing_reference: str | None = None

    @model_validator(mode="after")
    def monetary_basis(self):
        if self.monetary_ceiling is not None:
            from decimal import Decimal, InvalidOperation

            try:
                amount = Decimal(self.monetary_ceiling)
            except InvalidOperation as exc:
                raise ValueError("invalid monetary ceiling") from exc
            if (
                not amount.is_finite()
                or amount <= 0
                or not self.currency
                or not self.pricing_reference
            ):
                raise ValueError(
                    "money ceilings require a positive finite amount, currency and explicit pricing basis"
                )
        return self


class AnalysisPlan(Contract):
    family_alpha: float = Field(gt=0, lt=0.5)
    planning_power: float = Field(gt=0.5, lt=1)
    monte_carlo_tail_relative_se: float = Field(gt=0, lt=1)
    uncertainty: Literal["paired_repository_cluster_percentile_bootstrap"] = (
        "paired_repository_cluster_percentile_bootstrap"
    )
    multiplicity: Literal[
        "bonferroni_confirmatory_split_domain_cache_endpoint_family"
    ] = "bonferroni_confirmatory_split_domain_cache_endpoint_family"
    sample_size_rule: Literal[
        "paired_log_ratio_delta_method_plus_zero_regression_bound"
    ] = "paired_log_ratio_delta_method_plus_zero_regression_bound"
    independent_unit: Literal["repository_family_not_tasks_or_repeats"] = (
        "repository_family_not_tasks_or_repeats"
    )
    zero_acceptance: Literal["undefined_no_imputation_no_savings_claim"] = (
        "undefined_no_imputation_no_savings_claim"
    )
    allocation: Literal[
        "balanced_case_counts_equal_weight_repositories_within_domain"
    ] = "balanced_case_counts_equal_weight_repositories_within_domain"


class StudyDesign(Contract):
    schema_version: Literal[1] = 1
    id: Identity
    quality_rule: Literal["reject_any_observed_acceptance_regression"] = (
        "reject_any_observed_acceptance_regression"
    )
    primary_metric: Literal[
        "total_model_tokens_per_accepted_result_including_failures"
    ] = "total_model_tokens_per_accepted_result_including_failures"
    meaningful_token_reduction: float | None = Field(default=None, gt=0, lt=1)
    max_unseen_repository_regression_probability: float | None = Field(
        default=None, gt=0, lt=1
    )
    domains: list[Literal["service", "frontend"]]
    cache_states: list[Literal["cold", "warm"]]
    cache_control_references: dict[str, str] = Field(default_factory=dict)
    randomization_seed: int
    repeats: int | None = Field(default=None, gt=0)
    repeat_decision_reference: str | None = None
    required_scenarios: list[str]
    recipes: list[CandidateRecipe]
    cases: list[CaseCommitment] = Field(default_factory=list)
    analysis: AnalysisPlan
    reviewer_policy: ReviewerPolicy
    resources: StudyResources
    agent_profile_digest: Digest | None = None
    source_preparation_rule: Literal[
        "same_frozen_corpus_tools_instructions_limits_and_acceptance"
    ] = "same_frozen_corpus_tools_instructions_limits_and_acceptance"
    selection_rule: Literal[
        "calibration_only_then_immutable_policy_lock_before_validation_and_heldout"
    ] = "calibration_only_then_immutable_policy_lock_before_validation_and_heldout"
    stopping_rule: Literal[
        "no_efficacy_peeking_stop_on_quality_regression_budget_or_protocol_failure"
    ] = "no_efficacy_peeking_stop_on_quality_regression_budget_or_protocol_failure"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_inputs(self):
        for entries in (self.domains, self.cache_states, self.required_scenarios):
            if not entries or len(entries) != len(set(entries)):
                raise ValueError("declare nonempty unique study dimensions")
        if set(self.required_scenarios) != set(SCENARIOS):
            raise ValueError("broader study must cover every required scenario")
        if set(self.domains) != {"service", "frontend"} or set(self.cache_states) != {
            "cold",
            "warm",
        }:
            raise ValueError(
                "broader study requires both domains and both declared cache conditions"
            )
        for entries in (self.recipes, self.cases):
            if len({entry.id for entry in entries}) != len(entries):
                raise ValueError("study recipe/case IDs must be unique")
        if not self.recipes:
            raise ValueError("register at least one candidate recipe")
        return self

    @property
    def digest(self):
        return fingerprint(self.model_dump(mode="json"))


class PilotCluster(Contract):
    repository_family: str
    domain: Literal["service", "frontend"]
    recipe_id: Identity
    cache_state: Literal["cold", "warm"]
    scheduled_tasks: int = Field(gt=0)
    native_accepted: int = Field(ge=0)
    candidate_accepted: int = Field(ge=0)
    native_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(gt=0)
    observed_acceptance_regressions: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_counts(self):
        if (
            max(
                self.native_accepted,
                self.candidate_accepted,
                self.observed_acceptance_regressions,
            )
            > self.scheduled_tasks
        ):
            raise ValueError("pilot counts exceed scheduled tasks")
        if self.observed_acceptance_regressions < max(
            0, self.native_accepted - self.candidate_accepted
        ):
            raise ValueError(
                "paired regressions cannot be smaller than the acceptance count deficit"
            )
        if self.observed_acceptance_regressions > min(
            self.native_accepted, self.scheduled_tasks - self.candidate_accepted
        ):
            raise ValueError(
                "paired regression count is incompatible with acceptance counts"
            )
        return self


class PilotEvidence(Contract):
    schema_version: Literal[1] = 1
    split: Literal["calibration"] = "calibration"
    evidence_class: Literal["provider_validated", "diagnostic"]
    report_digests: list[Digest] = Field(min_length=1)
    assumptions_and_provenance: str = Field(min_length=1)
    provider_calls_complete: bool
    answer_reviews_complete: bool
    all_failures_and_costs_included: bool
    fixed_build_generation_review_tokens_included: bool
    independent_families_assessed: bool
    rows: list[PilotCluster] = Field(min_length=1)

    @property
    def digest(self):
        return fingerprint(self.model_dump(mode="json"))


def split_issues(design: StudyDesign) -> list[str]:
    issues = []
    for index, first in enumerate(design.cases):
        for second in design.cases[index + 1 :]:
            if first.split == second.split:
                if (
                    first.repository_family != second.repository_family
                    and first.source_digest == second.source_digest
                ):
                    issues.append("cloned_sources_declared_as_independent_families")
                continue
            if (
                first.repository_family == second.repository_family
                or first.repository_id == second.repository_id
                or first.source_digest == second.source_digest
            ):
                issues.append("repository_or_source_family_crosses_splits")
            if first.task_prompt_digest == second.task_prompt_digest:
                issues.append("identical_task_prompt_crosses_splits")
    return sorted(set(issues))


def sample_plan(design: StudyDesign, pilot: PilotEvidence | None) -> dict:
    family_tests = (
        2 * len(design.domains) * len(design.cache_states) * 2
    )  # validation/heldout x domains x cache x cost/quality
    alpha = design.analysis.family_alpha / family_tests
    tail = alpha / 2
    draws = math.ceil(
        (1 - tail) / (tail * design.analysis.monte_carlo_tail_relative_se**2)
    )
    blockers = []
    if design.meaningful_token_reduction is None:
        blockers.append("meaningful_token_reduction_undecided")
    if design.max_unseen_repository_regression_probability is None:
        blockers.append("unseen_regression_probability_target_unset")
    if pilot is None:
        blockers.append("independent_repository_pilot_variance_missing")
    elif pilot.evidence_class != "provider_validated" or not all(
        (
            pilot.provider_calls_complete,
            pilot.answer_reviews_complete,
            pilot.all_failures_and_costs_included,
            pilot.fixed_build_generation_review_tokens_included,
            pilot.independent_families_assessed,
        )
    ):
        blockers.append("pilot_evidence_incomplete_or_diagnostic")
    groups = defaultdict(list)
    if pilot:
        for row in pilot.rows:
            groups[(row.domain, row.recipe_id, row.cache_state)].append(row)
        expected = {
            (domain, recipe.id, cache)
            for domain in design.domains
            for recipe in design.recipes
            for cache in design.cache_states
        }
        if set(groups) - expected:
            blockers.append("pilot_contains_unregistered_dimensions")
        for row in pilot.rows:
            matching = [
                case
                for case in design.cases
                if case.split == "calibration"
                and case.domain == row.domain
                and case.repository_family == row.repository_family
                and not case.synthetic
                and not case.partial_snapshot
            ]
            if not matching:
                blockers.append(
                    "pilot_family_missing_from_eligible_calibration_commitments"
                )
    estimates = []
    for domain in design.domains:
        for recipe in design.recipes:
            for cache in design.cache_states:
                rows = groups[(domain, recipe.id, cache)]
                reasons = []
                if len({row.repository_family for row in rows}) != len(rows):
                    reasons.append(
                        "tasks_or_repeats_are_not_independent_repository_samples"
                    )
                if len(rows) < 2:
                    reasons.append(
                        "at_least_two_independent_families_needed_to_estimate_variance"
                    )
                if len({row.scheduled_tasks for row in rows}) > 1:
                    reasons.append("unbalanced_repository_case_counts")
                means = {
                    key: mean(getattr(row, key) for row in rows) if rows else 0.0
                    for key in (
                        "native_accepted",
                        "candidate_accepted",
                        "native_total_tokens",
                        "candidate_total_tokens",
                    )
                }
                if not means["native_accepted"] or not means["candidate_accepted"]:
                    reasons.append("zero_acceptance_denominator")
                estimated = None
                sd = None
                if not reasons:
                    influence = [
                        (
                            row.candidate_total_tokens / means["candidate_total_tokens"]
                            - 1
                        )
                        - (row.candidate_accepted / means["candidate_accepted"] - 1)
                        - (row.native_total_tokens / means["native_total_tokens"] - 1)
                        + (row.native_accepted / means["native_accepted"] - 1)
                        for row in rows
                    ]
                    sd = math.sqrt(variance(influence))
                    if sd == 0:
                        reasons.append(
                            "zero_observed_variance_is_not_proof_of_zero_population_variance"
                        )
                    elif (
                        design.meaningful_token_reduction is not None
                        and design.max_unseen_repository_regression_probability
                        is not None
                    ):
                        z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(
                            design.analysis.planning_power
                        )
                        effect = abs(math.log1p(-design.meaningful_token_reduction))
                        cost_n = math.ceil((z * sd / effect) ** 2)
                        quality_n = math.ceil(
                            math.log(alpha)
                            / math.log1p(
                                -design.max_unseen_repository_regression_probability
                            )
                        )
                        estimated = max(2, cost_n, quality_n)
                if any(row.observed_acceptance_regressions for row in rows):
                    reasons.append("observed_quality_regression_disqualifies_recipe")
                estimates.append(
                    {
                        "domain": domain,
                        "recipe": recipe.id,
                        "cache": cache,
                        "independent_families": len(rows),
                        "paired_log_ratio_influence_sd": sd,
                        "planning_repository_count_per_split": estimated
                        if not reasons and not blockers
                        else None,
                        "issues": reasons,
                        "method": "first-order normal planning approximation plus exact zero-event bound; not a power guarantee",
                    }
                )
    return {
        "status": "pending"
        if blockers or any(entry["issues"] for entry in estimates)
        else "approximate_planning_only",
        "blockers": blockers,
        "groups": estimates,
        "confirmatory_family_tests": family_tests,
        "per_comparison_alpha": alpha,
        "bootstrap_draws_for_declared_tail_mc_precision": draws,
        "pilot_digest": pilot.digest if pilot else None,
        "independent_unit": "repository family; repeats do not increase this count",
        "quality_interpretation": "Reject any observed acceptance regression. Finite zero-event data cannot prove true zero population regression risk.",
    }


def study_plan(design: StudyDesign, pilot: PilotEvidence | None = None) -> dict:
    sizing = sample_plan(design, pilot)
    issues = split_issues(design)
    issues.extend(sizing["blockers"])
    if sizing["status"] != "approximate_planning_only":
        issues.append("sample_size_not_plannable")
    if design.repeats is None or not design.repeat_decision_reference:
        issues.append("repeat_count_and_variability_based_rationale_missing")
    if design.agent_profile_digest is None:
        issues.append("task_agent_profile_missing")
    if design.reviewer_policy.primary_profile_digest is None:
        issues.append("reviewer_profile_missing")
    if set(design.cache_states) - {
        key for key, value in design.cache_control_references.items() if value.strip()
    }:
        issues.append("provider_cache_control_evidence_missing")
    resources = design.resources
    required_limits = (
        "future_preparation_model_call_limit",
        "future_model_call_limit",
        "future_provider_token_limit",
        "provider_tokens_per_call_ceiling",
        "max_task_model_turns",
        "max_task_tool_calls",
        "max_task_elapsed_ms",
        "max_request_payload_tokens",
        "max_response_payload_tokens",
        "max_total_visible_payload_tokens",
    )
    if any(getattr(resources, key) is None for key in required_limits):
        issues.append("explicit_future_resource_ceiling_missing")
    matrix = []
    for split in SPLITS:
        for domain in design.domains:
            cases = [
                entry
                for entry in design.cases
                if entry.split == split and entry.domain == domain
            ]
            eligible = [
                entry
                for entry in cases
                if not entry.synthetic and not entry.partial_snapshot
            ]
            covered = set(
                scenario for entry in eligible for scenario in entry.scenarios
            )
            missing = sorted(set(design.required_scenarios) - covered)
            if missing:
                issues.append(f"{split}/{domain}:scenario_coverage_incomplete")
            families = {entry.repository_family for entry in eligible}
            counts = Counter(entry.repository_family for entry in eligible)
            if len(set(counts.values())) > 1:
                issues.append(f"{split}/{domain}:unbalanced_repository_case_counts")
            relevant = [
                entry
                for entry in sizing["groups"]
                if entry["domain"] == domain
                and entry["planning_repository_count_per_split"] is not None
            ]
            required = max(
                (entry["planning_repository_count_per_split"] for entry in relevant),
                default=None,
            )
            if (
                required is not None
                and split != "calibration"
                and len(families) < required
            ):
                issues.append(f"{split}/{domain}:independent_repository_pool_too_small")
            matrix.append(
                {
                    "split": split,
                    "domain": domain,
                    "committed_cases": len(cases),
                    "eligible_families": len(families),
                    "required_independent_families": required,
                    "missing_scenarios": missing,
                }
            )
    # A conservative ceiling includes each native/treatment pair, its reviews,
    # and the separately declared preparation allowance. It grants no spend.
    task_runs = None
    max_calls = None
    max_tokens = None
    if design.repeats is not None:
        calibration = sum(case.split == "calibration" for case in design.cases)
        confirmatory = len(design.cases) - calibration
        task_runs = (
            2
            * len(design.cache_states)
            * design.repeats
            * (calibration * len(design.recipes) + confirmatory)
        )
        if (
            resources.max_task_model_turns is not None
            and resources.future_preparation_model_call_limit is not None
        ):
            max_calls = (
                task_runs
                * (
                    resources.max_task_model_turns
                    + design.reviewer_policy.max_review_model_calls_per_answer
                )
                + resources.future_preparation_model_call_limit
            )
            if (
                resources.future_model_call_limit is not None
                and max_calls > resources.future_model_call_limit
            ):
                issues.append("scheduled_worst_case_calls_exceed_declared_ceiling")
            if resources.provider_tokens_per_call_ceiling is not None:
                max_tokens = max_calls * resources.provider_tokens_per_call_ceiling
                if (
                    resources.future_provider_token_limit is not None
                    and max_tokens > resources.future_provider_token_limit
                ):
                    issues.append(
                        "scheduled_worst_case_provider_tokens_exceed_declared_ceiling"
                    )
    protocol_ready = not issues
    # The actual T09 runner/analysis and strict provider guarantees are not
    # implemented/validated by freezing this offline protocol.
    execution_blockers = [
        "study_execution_not_implemented_in_t08",
        "pilot_provenance_requires_independent_t09_verification",
        "full_provider_accounting_pending",
        "complete_host_isolation_pending",
        "hard_provider_budget_pending",
    ]
    if not protocol_ready:
        execution_blockers.append("protocol_inputs_incomplete")
    return {
        "schema_version": 1,
        "design_digest": design.digest,
        "registration_status": "protocol_ready" if protocol_ready else "draft",
        "protocol_blockers": sorted(set(issues)),
        "execution_ready": False,
        "execution_blockers": execution_blockers,
        "workload_matrix": matrix,
        "sample_plan": sizing,
        "conditional_schedule": {
            "calibration": "each registered case/cache/repeat x every candidate recipe x a separate paired native baseline",
            "validation": "selected repository pool x cases/cache/repeats x native and immutable calibration-selected recipe",
            "heldout": "reveal only after policy lock; run the same frozen selected recipe and native; no tuning",
            "repeats": design.repeats,
            "materialized_runs": 0,
            "missing_inputs_do_not_become_one_repeat_or_one_repository": True,
        },
        "resource_envelope": {
            "task_runs": task_runs,
            "worst_case_model_calls": max_calls,
            "worst_case_provider_tokens": max_tokens,
            "interpretation": "sufficient conservative planning envelope; requires actual provider enforcement before execution",
        },
        "current_codex_strict_capabilities": codex_capabilities(),
        "model_calls": 0,
        "default_selected": None,
    }


def register_study(
    design: StudyDesign,
    output: Path,
    pilot: PilotEvidence | None = None,
    *,
    allow_draft=False,
) -> dict:
    report = study_plan(design, pilot)
    if report["protocol_blockers"] and not allow_draft:
        raise ValueError(
            "study protocol is incomplete; inspect plan or explicitly freeze a draft"
        )
    output.mkdir(parents=True, exist_ok=False)
    artifact = {
        "schema_version": 1,
        "design": design.model_dump(mode="json"),
        "pilot": pilot.model_dump(mode="json") if pilot else None,
        "report": report,
        "authorization": "offline registration only; no model or acceptance commands are executed",
    }
    payload = json.dumps(artifact, ensure_ascii=False, indent=2)
    (output / "protocol.json").write_bytes(payload.encode("utf-8"))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "design_digest": design.digest,
                "registration_status": report["registration_status"],
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def load_study(path: Path) -> tuple[StudyDesign, PilotEvidence | None, dict]:
    raw = (path / "protocol.json").read_bytes()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["implementation_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError(
            "study implementation changed; register a new version rather than silently reusing a protocol"
        )
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("registered study artifact bytes changed")
    artifact = json.loads(raw)
    design = StudyDesign.model_validate(artifact["design"])
    pilot = (
        PilotEvidence.model_validate(artifact["pilot"]) if artifact["pilot"] else None
    )
    report = study_plan(design, pilot)
    if design.digest != manifest["design_digest"] or report != artifact["report"]:
        raise ValueError(
            "registered study protocol/report no longer matches this implementation"
        )
    return design, pilot, report


class PolicyLock(Contract):
    schema_version: Literal[1] = 1
    design_digest: Digest
    recipe_id: Identity
    calibration_report_digests: list[Digest] = Field(min_length=1)
    selection_review_reference: str = Field(min_length=1)
    rule: Literal["calibration_only_no_validation_or_heldout_tuning"] = (
        "calibration_only_no_validation_or_heldout_tuning"
    )


def lock_policy(
    registration: Path,
    *,
    recipe_id: str,
    calibration_report_digests: list[str],
    selection_review_reference: str,
) -> PolicyLock:
    """Record a curator's selection; does not authenticate the reports or run models."""
    design, _, report = load_study(registration)
    if report["protocol_blockers"]:
        raise ValueError("cannot lock a policy from an incomplete draft")
    if recipe_id not in {recipe.id for recipe in design.recipes}:
        raise ValueError("policy recipe was not preregistered")
    lock = PolicyLock(
        design_digest=design.digest,
        recipe_id=recipe_id,
        calibration_report_digests=calibration_report_digests,
        selection_review_reference=selection_review_reference,
    )
    with (registration / "policy-lock.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(lock.model_dump_json(indent=2))
    return lock


def load_policy_lock(registration: Path, design: StudyDesign) -> PolicyLock:
    lock = PolicyLock.model_validate_json(
        (registration / "policy-lock.json").read_bytes()
    )
    if lock.design_digest != design.digest or lock.recipe_id not in {
        recipe.id for recipe in design.recipes
    }:
        raise ValueError("policy lock does not match registered design")
    return lock


def schedule(registration: Path, *, split: str) -> list[dict]:
    """Stable paired/randomized identifiers only; never read task/rubric artifacts."""
    design, _, report = load_study(registration)
    if split not in SPLITS:
        raise ValueError("unknown study split")
    if report["protocol_blockers"]:
        raise ValueError("cannot materialize schedule from an incomplete draft")
    recipes = (
        [recipe.id for recipe in design.recipes]
        if split == "calibration"
        else [load_policy_lock(registration, design).recipe_id]
    )
    pairs = []
    for case in design.cases:
        if case.split != split:
            continue
        for cache in design.cache_states:
            for repeat in range(design.repeats):
                for recipe in recipes:
                    identity = {
                        "design": design.digest,
                        "case": case.id,
                        "recipe": recipe,
                        "cache": cache,
                        "repeat": repeat,
                    }
                    pair_id = fingerprint(identity)
                    arms = ["native", "bundle"]
                    if (
                        int(
                            fingerprint(
                                [design.randomization_seed, pair_id, "arm-order"]
                            ),
                            16,
                        )
                        % 2
                    ):
                        arms.reverse()
                    pairs.append(
                        (
                            fingerprint(
                                [design.randomization_seed, pair_id, "pair-order"]
                            ),
                            [
                                {
                                    "pair_id": pair_id,
                                    "run_id": fingerprint([pair_id, arm]),
                                    "split": split,
                                    "case_id": case.id,
                                    "artifact_sha256": case.artifact_sha256,
                                    "recipe_id": recipe,
                                    "cache_state": cache,
                                    "repeat": repeat,
                                    "arm": arm,
                                    "fresh_session_required": True,
                                }
                                for arm in arms
                            ],
                        )
                    )
    return [
        entry for _, pair in sorted(pairs, key=lambda value: value[0]) for entry in pair
    ]


def verify_case(registration: Path, case_id: str, artifact_path: Path) -> dict:
    """Check the seal before parsing or returning any committed source/rubric bytes.

    Curator separation and filesystem access are organizational controls, not a
    security boundary. A policy lock attests selection; it does not prove blindness.
    """
    design, _, report = load_study(registration)
    case = next((entry for entry in design.cases if entry.id == case_id), None)
    if case is None:
        raise ValueError("case is not committed in this study")
    if case.split != "calibration":
        if report["protocol_blockers"]:
            raise ValueError(
                "cannot reveal confirmatory cases from an incomplete draft"
            )
        load_policy_lock(registration, design)
    raw = artifact_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != case.artifact_sha256:
        raise ValueError("case artifact bytes do not match the commitment")
    artifact = (
        load_experiment(artifact_path)
        if case.artifact_kind == "discovery"
        else AcceptanceRegistration.model_validate_json(raw)
    )
    task = artifact.suite.task(artifact.task_id)
    if (
        artifact.digest != case.artifact_digest
        or artifact.snapshot.source_digest != case.source_digest
        or task.id != case.task_id
        or task.repository_id != case.repository_id
        or fingerprint(task.prompt) != case.task_prompt_digest
    ):
        raise ValueError("case identity or source changed")
    # The command exposes only verification metadata. Agent packets must still
    # be built with the existing rubric-free packet constructor in T09.
    return {
        "case_id": case.id,
        "artifact_digest": artifact.digest,
        "verified": True,
        "model_calls": 0,
        "agent_packet_emitted": False,
    }

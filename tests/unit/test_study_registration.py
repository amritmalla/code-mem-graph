"""Offline fixtures test protocol mechanics, never empirical power or efficacy."""

import hashlib
import json

from click.testing import CliRunner
from pydantic import ValidationError
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.study import (
    SCENARIOS,
    CandidateRecipe,
    CaseCommitment,
    PilotCluster,
    PilotEvidence,
    StudyDesign,
    commit_case,
    load_study,
    lock_policy,
    register_study,
    sample_plan,
    schedule,
    select_candidate,
    split_issues,
    study_plan,
    verify_case,
)
from code_mem_graph.evaluation.experiment import prepare_experiment, save_experiment
from code_mem_graph.evaluation.models import Suite
from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.serialization import fingerprint


def seal(split, domain, index, *, kind="discovery", **changes):
    family = f"{split}-{domain}-{index}"
    scenarios = [
        scenario
        for scenario in SCENARIOS
        if ("change" in scenario) == (kind == "change")
    ]
    data = dict(
        id=f"{family}-{kind}",
        split=split,
        domain=domain,
        repository_family=family,
        repository_id=family,
        source_digest=fingerprint(family),
        artifact_sha256=fingerprint([family, kind]),
        artifact_digest=fingerprint([family, kind, "artifact"]),
        artifact_kind=kind,
        task_id=kind,
        task_prompt_digest=fingerprint([family, kind, "prompt"]),
        scenarios=scenarios,
        exposure="inspected" if split == "calibration" else "curator_only",
        curator_reference="unit fixture",
        synthetic=False,
        partial_snapshot=False,
        affected_components=["a", "b"] if kind == "change" else [],
    )
    data.update(changes)
    return CaseCommitment(**data)


def design(**changes):
    data = dict(
        id="unit-study",
        domains=["service", "frontend"],
        cache_states=["cold", "warm"],
        randomization_seed=17,
        required_scenarios=list(SCENARIOS),
        recipes=[dict(id="detailed-full", representation="detailed", detail="full")],
        analysis=dict(
            family_alpha=0.05, planning_power=0.8, monte_carlo_tail_relative_se=0.1
        ),
        reviewer_policy={},
        resources={},
    )
    data.update(changes)
    return StudyDesign(**data)


def pilot(value, *, diagnostic=False, spread=1):
    rows = []
    for domain in value.domains:
        for recipe in value.recipes:
            for cache in value.cache_states:
                for index, tokens in enumerate((100 - spread, 100 + spread)):
                    rows.append(
                        PilotCluster(
                            repository_family=f"calibration-{domain}-{index}",
                            domain=domain,
                            recipe_id=recipe.id,
                            cache_state=cache,
                            scheduled_tasks=2,
                            native_accepted=2,
                            candidate_accepted=2,
                            native_total_tokens=200,
                            candidate_total_tokens=tokens,
                            observed_acceptance_regressions=0,
                        )
                    )
    return PilotEvidence(
        evidence_class="diagnostic" if diagnostic else "provider_validated",
        report_digests=[fingerprint("fictional report")],
        assumptions_and_provenance="FICTIONAL unit data, no provider validation",
        provider_calls_complete=True,
        answer_reviews_complete=True,
        all_failures_and_costs_included=True,
        fixed_build_generation_review_tokens_included=True,
        independent_families_assessed=True,
        rows=rows,
    )


def ready_design():
    # Extreme thresholds keep this mechanical fixture small, not scientifically useful.
    resources = dict(
        future_preparation_model_call_limit=0,
        future_model_call_limit=100000,
        future_provider_token_limit=100000000,
        provider_tokens_per_call_ceiling=100,
        max_task_model_turns=3,
        max_task_tool_calls=10,
        max_task_elapsed_ms=1000,
        max_request_payload_tokens=100,
        max_response_payload_tokens=100,
        max_total_visible_payload_tokens=1000,
    )
    cases = [
        seal(split, domain, index, kind=kind)
        for split in ("calibration", "validation", "heldout")
        for domain in ("service", "frontend")
        for index in range(2)
        for kind in ("discovery", "change")
    ]
    return design(
        cases=cases,
        meaningful_token_reduction=0.99,
        max_unseen_repository_regression_probability=0.99,
        repeats=1,
        repeat_decision_reference="fictional variability rationale",
        resources=resources,
        agent_profile_digest=fingerprint("agent"),
        reviewer_policy={"primary_profile_digest": fingerprint("reviewer")},
        cache_control_references={
            "cold": "fictional cold control",
            "warm": "fictional warm control",
        },
    )


def test_undecided_effect_never_becomes_a_convenient_sample_count():
    value = design()
    report = study_plan(value)
    assert report["registration_status"] == "draft"
    assert "meaningful_token_reduction_undecided" in report["protocol_blockers"]
    assert all(
        row["planning_repository_count_per_split"] is None
        for row in report["sample_plan"]["groups"]
    )
    assert report["conditional_schedule"]["repeats"] is None
    assert report["model_calls"] == 0 and not report["execution_ready"]


def test_sample_size_changes_with_variability_and_meaningful_effect():
    value = ready_design().model_copy(
        update={
            "meaningful_token_reduction": 0.05,
            "max_unseen_repository_regression_probability": 0.95,
        }
    )
    low = sample_plan(value, pilot(value, spread=1))["groups"][0][
        "planning_repository_count_per_split"
    ]
    high = sample_plan(value, pilot(value, spread=40))["groups"][0][
        "planning_repository_count_per_split"
    ]
    smaller = sample_plan(
        value.model_copy(update={"meaningful_token_reduction": 0.01}),
        pilot(value, spread=40),
    )["groups"][0]["planning_repository_count_per_split"]
    assert low < high < smaller
    repeated = sample_plan(
        value.model_copy(update={"repeats": 50}), pilot(value, spread=40)
    )
    assert repeated["groups"][0]["planning_repository_count_per_split"] == high


@pytest.mark.parametrize(
    "fault",
    [
        "diagnostic",
        "duplicate",
        "zero-variance",
        "zero-success",
        "regression",
        "unknown-family",
        "unregistered",
    ],
)
def test_ineligible_pilot_cannot_size_a_study(fault):
    value = ready_design()
    evidence = pilot(
        value,
        diagnostic=fault == "diagnostic",
        spread=0 if fault == "zero-variance" else 1,
    )
    if fault == "duplicate":
        evidence.rows.append(evidence.rows[0])
    if fault == "zero-success":
        for row in evidence.rows:
            row.native_accepted = 0
            row.candidate_accepted = 0
    if fault == "regression":
        evidence.rows[0].candidate_accepted = 1
        evidence.rows[0].observed_acceptance_regressions = 1
    if fault == "unknown-family":
        evidence.rows[0].repository_family = "invented-family"
    if fault == "unregistered":
        evidence.rows[0].recipe_id = "unregistered"
    report = sample_plan(value, evidence)
    assert report["status"] == "pending"
    assert any(
        row["planning_repository_count_per_split"] is None for row in report["groups"]
    )


def test_paired_acceptance_counts_cannot_hide_a_regression():
    row = pilot(ready_design()).rows[0].model_dump()
    row["candidate_accepted"] = 1
    with pytest.raises(ValidationError, match="deficit"):
        PilotCluster(**row)


def test_partial_synthetic_exposed_and_cloned_repositories_cannot_be_heldout():
    value = ready_design()
    case = value.cases[0]
    with pytest.raises(ValidationError, match="only to calibration"):
        CaseCommitment(**(case.model_dump() | {"split": "heldout"}))
    clone = seal("heldout", "service", 100, source_digest=case.source_digest)
    value.cases.append(clone)
    assert "repository_or_source_family_crosses_splits" in split_issues(value)
    value = ready_design()
    for case in value.cases:
        case.partial_snapshot = True
    assert all(
        row["eligible_families"] == 0
        for row in study_plan(value, pilot(value))["workload_matrix"]
    )


def test_readonly_tasks_cannot_be_mislabeled_actual_cross_component_work():
    with pytest.raises(ValidationError, match="actual change scenarios"):
        seal("calibration", "service", 1, scenarios=["cross-component-change"])
    with pytest.raises(ValidationError, match="two explicitly"):
        seal("calibration", "service", 1, kind="change", affected_components=["a"])


def test_draft_registration_is_exact_byte_bound_and_not_overwritten(tmp_path):
    target = tmp_path / "draft"
    with pytest.raises(ValueError, match="incomplete"):
        register_study(design(), target)
    assert not target.exists()
    register_study(design(), target, allow_draft=True)
    assert load_study(target)[2]["registration_status"] == "draft"
    manifest = json.loads((target / "manifest.json").read_bytes())
    assert (
        manifest["sha256"]
        == hashlib.sha256((target / "protocol.json").read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        register_study(design(), target, allow_draft=True)
    with pytest.raises(ValueError, match="incomplete"):
        schedule(target, split="calibration")
    with pytest.raises(ValueError, match="incomplete"):
        lock_policy(
            target,
            recipe_id="detailed-full",
            calibration_report_digests=[fingerprint("report")],
            selection_review_reference="fixture",
        )
    with (target / "protocol.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        load_study(target)


def test_ready_protocol_still_cannot_run_models_and_requires_policy_lock(tmp_path):
    value = ready_design()
    target = tmp_path / "registered"
    report = register_study(value, target, pilot(value))
    assert report["registration_status"] == "protocol_ready"
    assert not report["execution_ready"] and report["model_calls"] == 0
    first = schedule(target, split="calibration")
    assert first == schedule(target, split="calibration")
    assert len({row["run_id"] for row in first}) == len(first)
    assert all(
        {row["arm"] for row in first if row["pair_id"] == pair} == {"native", "bundle"}
        for pair in {row["pair_id"] for row in first}
    )
    assert {row["arm"] for row in first[::2]} == {"native", "bundle"}
    assert all("prompt" not in row and "rubric" not in row for row in first)
    with pytest.raises(FileNotFoundError):
        schedule(target, split="heldout")
    lock_policy(
        target,
        recipe_id="detailed-full",
        calibration_report_digests=[fingerprint("report")],
        selection_review_reference="fixture selection",
    )
    heldout = schedule(target, split="heldout")
    assert {row["recipe_id"] for row in heldout} == {"detailed-full"}
    with pytest.raises(FileExistsError):
        lock_policy(
            target,
            recipe_id="detailed-full",
            calibration_report_digests=[fingerprint("report")],
            selection_review_reference="overwrite",
        )


def test_resource_plan_includes_task_and_review_costs_and_blocks_insufficient_caps():
    value = ready_design()
    value.resources.future_model_call_limit = 1
    value.resources.future_provider_token_limit = 1
    report = study_plan(value, pilot(value))
    assert (
        report["resource_envelope"]["worst_case_model_calls"]
        > report["resource_envelope"]["task_runs"]
    )
    assert (
        "scheduled_worst_case_calls_exceed_declared_ceiling"
        in report["protocol_blockers"]
    )
    assert (
        "scheduled_worst_case_provider_tokens_exceed_declared_ceiling"
        in report["protocol_blockers"]
    )


def test_cli_draft_plan_register_and_inspect(tmp_path):
    runner = CliRunner()
    source = tmp_path / "design.json"
    source.write_text(design().model_dump_json(), encoding="utf-8")
    assert runner.invoke(main, ["study", "schema", "design"]).exit_code == 0
    result = runner.invoke(main, ["study", "plan", str(source)])
    assert result.exit_code == 0 and json.loads(result.output)["model_calls"] == 0
    target = tmp_path / "draft"
    args = ["study", "register", str(source), "--output", str(target)]
    assert runner.invoke(main, args).exit_code != 0
    assert runner.invoke(main, args + ["--allow-draft"]).exit_code == 0
    assert runner.invoke(main, ["study", "inspect", str(target)]).exit_code == 0


def test_real_artifact_commitment_selection_and_seal_verification(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "README.md").write_text("FICTIONAL source repository.\n")
    (root / "app.py").write_text('def health(): return "ok"\n')
    build_repository(root, repository_id="unit")
    suite = Suite.model_validate(
        {
            "schema_version": 1,
            "id": "unit",
            "protocol": {
                "id": "unit",
                "max_tool_calls": 2,
                "max_model_turns": 2,
                "max_elapsed_ms": 1000,
                "stopping_rule": "fixture",
            },
            "repositories": [{"id": "unit", "revision": "fixture"}],
            "tasks": [
                {
                    "id": "overview",
                    "repository_id": "unit",
                    "category": "orientation",
                    "prompt": "Describe the project.",
                    "criteria": [
                        {
                            "id": "purpose",
                            "description": "PRIVATE RUBRIC",
                            "critical": True,
                            "source_refs": ["README.md"],
                        }
                    ],
                    "acceptance_checks": {"answer": "PRIVATE CHECK"},
                }
            ],
        }
    )
    experiment = prepare_experiment(
        root,
        suite,
        task_id="overview",
        mode="orient",
        encoding="cl100k_base",
        ceiling=10000,
        split="calibration",
        replicates=1,
        cache_state="cold",
    )
    path = tmp_path / "experiment.json"
    save_experiment(experiment, path)
    case = commit_case(
        path,
        case_id="real-artifact",
        split="calibration",
        domain="service",
        repository_family="unit",
        scenarios=["orientation"],
        curator_reference="unit",
        exposure="inspected",
        synthetic=True,
        partial_snapshot=True,
    )
    assert "PRIVATE" not in case.model_dump_json()
    recipe = CandidateRecipe(
        id="detailed-full", representation="detailed", detail="full"
    )
    selected = select_candidate(experiment, recipe)
    experiment.suite.tasks[0].criteria[0].description = "CHANGED PRIVATE RUBRIC"
    assert select_candidate(experiment, recipe).id == selected.id
    value = design(cases=[case])
    target = tmp_path / "registered"
    register_study(value, target, allow_draft=True)
    assert verify_case(target, case.id, path)["verified"]
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="commitment"):
        verify_case(target, case.id, path)

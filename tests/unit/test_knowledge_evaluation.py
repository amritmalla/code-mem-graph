from copy import deepcopy
import json
from pathlib import Path

from click.testing import CliRunner
from pydantic import ValidationError
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.evaluation.models import (
    AgentProfile, CaptureInfo, ModelEvent, PolicyProfile, Review, Suite, ToolEvent, Trace, Usage, UsageCapture, UsageObservation,
    serialize,
)
from code_mem_graph.evaluation.report import PayloadCounter, build_report, score_run
from code_mem_graph.evaluation.trace import TraceRecorder, write_trace
from tests.review_fixtures import source_snapshot, supported_audit


class CharacterCounter:
    """Deterministic proxy for accounting tests, never presented as real tokens."""

    description = {"basis": "test_character_count"}

    def __call__(self, value):
        return len(serialize(value))


@pytest.fixture
def suite():
    return Suite.model_validate({
        "schema_version": 1, "id": "test-suite",
        "protocol": {"id": "test-protocol", "max_tool_calls": 2, "max_model_turns": 2,
                     "max_elapsed_ms": 1000.0, "stopping_rule": "Stop on answer or ceiling."},
        "repositories": [{"id": "repo", "revision": "snapshot:"+source_snapshot().digest}],
        "tasks": [{"id": "overview", "repository_id": "repo", "category": "orientation",
                   "prompt": "Explain the project.", "criteria": [
                       {"id": "purpose", "description": "Cite purpose.", "critical": True,
                        "source_refs": ["README.md"]},
                       {"id": "checks", "description": "Cite checks.", "critical": False,
                        "source_refs": ["CONTRIBUTING.md"]},
                   ], "acceptance_checks": {"validation": "Independent acceptance check."}}],
    })


def reviewed():
    return Review(verdict="accepted", reviewer="independent-test-fixture",
                  criteria={"purpose": "met", "checks": "met"},
                  acceptance_checks={"validation": "passed"})


def run(suite, *, name="native", run_id="baseline", usage=True):
    tokens = Usage(input_tokens=100, output_tokens=20, cached_input_tokens=60) if usage else None
    trace = Trace(
        schema_version=1, run_id=run_id, suite_digest=suite.digest,
        task_id="overview", repository_revision=suite.repositories[0].revision,
        replicate="01", cache_state="cold",
        agent=AgentProfile(kind="agent", provider="fixture", model="fixture",
                           model_version="1", instructions_digest="fixture-hash", settings={}),
        policy=PolicyProfile(name=name, version="1", parameters={}),
        events=[
            ToolEvent(sequence=0, name="read", request={"path": "README.md"},
                      response={"text": "project", "citations": ["README.md:1"],
                                "metadata": {"complete": True}}, elapsed_ms=2.0),
            ModelEvent(sequence=1, request={"messages": ["task", "project"]},
                       response={"tool_calls":[],"final":"A fixture project [README.md:1]; documented validation is pytest tests [CONTRIBUTING.md:1]."},
                       usage=tokens, usage_capture=UsageCapture(scope='provider_calls', source='fictional test provider',
                           expected_calls=1, internal_calls_complete=True, observations=[UsageObservation(id=run_id,
                               reference='fixture response', outcome='completed', usage=tokens)])),
        ], elapsed_ms=10.0, outcome="completed", review=reviewed(),
        coordination={'registration_digest':'fictional-shared-registration','deviations':[]},
        capture=CaptureInfo(request_capture='complete',provider_usage_available=True,model_version_pinned=True,
                            host_tools_enforced=True,provider_token_budget_enforced=True),
    )
    trace.review.validation=supported_audit(suite,trace,source_snapshot()).model_dump(mode='json')
    return trace


def test_complete_payload_and_no_double_counting(suite):
    trace = run(suite)
    result = score_run(suite, trace, CharacterCounter())
    tool, turn = trace.events
    assert result["tool_response_tokens_estimate"] == len(serialize(tool.response))
    assert result["tool_response_tokens_estimate"] > len(serialize(tool.response["text"]))
    assert result["model_total_tokens_estimate"] == len(serialize(turn.request)) + len(serialize(turn.response))
    assert result["reported_model_total_tokens"] == 120  # cached 60 is already part of input 100
    assert result["observed_cached_input_tokens"] == 60
    assert result["accepted"] is True


def test_repeated_context_and_partial_usage(suite):
    trace = run(suite)
    turn = trace.events[-1].model_copy(deep=True)
    turn.sequence, turn.usage = 2, None
    turn.usage_capture = None
    trace.events.append(turn)
    result = score_run(suite, trace, CharacterCounter())
    assert result["model_input_tokens_estimate"] == 2 * len(serialize(turn.request))
    assert result["reported_model_total_tokens"] is None
    assert result["usage_complete"] is False
    assert result["usage_turns"] == 1
    assert result["observed_input_tokens"] == 100


def test_missing_model_usage_is_unknown(suite):
    trace = run(suite)
    trace.events.pop()
    result = score_run(suite, trace, CharacterCounter())
    assert result["model_total_tokens_estimate"] is None
    assert result["reported_model_total_tokens"] is None
    assert result["usage_complete"] is False


@pytest.mark.parametrize("change,expected", [
    (lambda trace: setattr(trace, "review", Review()), "unreviewed"),
    (lambda trace: setattr(trace, "outcome", "failed"), "failed"),
    (lambda trace: setattr(trace, "outcome", "aborted"), "aborted"),
    (lambda trace: setattr(trace, "elapsed_ms", 1001.0), "limit_exceeded"),
    (lambda trace: trace.review.criteria.pop("purpose"), "rejected"),
    (lambda trace: trace.review.criteria.update(checks="unknown"), "rejected"),
    (lambda trace: trace.review.acceptance_checks.update(validation="not_run"), "rejected"),
    (lambda trace: trace.review.unsupported_high_impact_claims.append("Invented approver"), "rejected"),
])
def test_quality_and_failures_remain_visible(suite, change, expected):
    trace = run(suite)
    change(trace)
    report = build_report(suite, [trace], CharacterCounter(), baseline="native")
    assert report["status_counts"] == {expected: 1}
    assert report["runs"][0]["accepted"] is False
    assert report["calibration_status"] == "not_calibrated"


def test_omitted_critical_guidance_identified(suite):
    trace = run(suite)
    trace.review.criteria.pop("purpose")
    result = score_run(suite, trace, CharacterCounter())
    assert result["missing_critical_criteria"] == ["purpose"]
    assert result["coverage"] == 0.5


@pytest.mark.parametrize("field,value,error", [
    ("suite_digest", "different", "fingerprint mismatch"),
    ("repository_revision", "dirty-or-different", "revision mismatch"),
    ("task_id", "absent", "unknown task"),
])
def test_mismatched_experiment_rejected(suite, field, value, error):
    trace = run(suite)
    setattr(trace, field, value)
    with pytest.raises(ValueError, match=error):
        score_run(suite, trace, CharacterCounter())


@pytest.mark.parametrize("field", ["criteria", "acceptance_checks"])
def test_unknown_review_keys_rejected(suite, field):
    trace = run(suite)
    getattr(trace.review, field)["typo"] = "met" if field == "criteria" else "passed"
    with pytest.raises(ValueError, match="unknown"):
        score_run(suite, trace, CharacterCounter())


def test_paired_usage_savings(suite):
    baseline = run(suite)
    candidate = run(suite, name="knowledge", run_id="candidate")
    candidate.events[-1].usage = Usage(input_tokens=40, output_tokens=20, cached_input_tokens=0)
    candidate.events[-1].usage_capture.observations[0].usage = candidate.events[-1].usage
    report = build_report(suite, [baseline, candidate], CharacterCounter(), baseline="native")
    pair = report["comparisons"][0]
    assert pair["eligible"] is True
    assert pair["token_basis"] == "reported_model_total_tokens"
    assert pair["token_saving_fraction"] == 0.5


@pytest.mark.parametrize('coordination,blocker', [
    (None,'coordination_undeclared'),
    ({'registration_digest':None,'deviations':[]},'cohort_registration_missing'),
    ({'registration_digest':'different','deviations':[]},'cohort_registrations_differ'),
    ({'registration_digest':'fictional-shared-registration','deviations':['dispatch_delivery_uncertain']},'dispatch_delivery_uncertain'),
])
def test_coordination_gates_optimistic_complete_capture(suite,coordination,blocker):
    candidate = run(suite,name='knowledge',run_id='candidate')
    candidate.coordination = coordination
    pair = build_report(suite,[run(suite),candidate],CharacterCounter(),baseline='native')['comparisons'][0]
    assert pair['eligible'] is False and pair['token_saving_fraction'] is None
    assert blocker in pair['capture_blockers']


def test_estimates_remain_diagnostic_when_provider_usage_is_missing(suite):
    candidate = run(suite, name="knowledge", run_id="candidate", usage=False)
    report = build_report(suite, [run(suite), candidate], CharacterCounter(), baseline="native")
    pair = report["comparisons"][0]
    assert pair["eligible"] is False
    assert pair["token_basis"] is None and pair['token_saving_fraction'] is None
    assert report['runs'][1]['model_total_tokens_estimate'] is not None


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('field,value,blocker', [
    ('request_capture', 'visible_only', 'only_visible_requests_captured'),
    ('provider_usage_available', False, 'provider_usage_unavailable'),
    ('model_version_pinned', False, 'provider_model_version_unpinned'),
    ('host_tools_enforced', False, 'host_tool_isolation_not_enforced'),
    ('provider_token_budget_enforced', False, 'provider_token_budget_not_enforced'),
])
def test_evaluate_blocks_known_partial_capture_on_either_arm(suite, side, field, value, blocker):
    traces = [run(suite), run(suite, name='knowledge', run_id='candidate')]
    declaration = dict(request_capture='complete', provider_usage_available=True,
                       model_version_pinned=True, host_tools_enforced=True, provider_token_budget_enforced=True)
    declaration[field] = value
    traces[side].capture = CaptureInfo.model_validate(declaration)
    pair = build_report(suite, traces, CharacterCounter(), baseline='native')['comparisons'][0]
    assert pair['eligible'] is False and pair['reason'] == 'incomplete_capture'
    assert blocker in pair['capture_blockers']
    assert pair['token_saving_fraction'] is None and pair['token_basis'] is None


@pytest.mark.parametrize("change", [
    lambda trace: setattr(trace, "cache_state", "warm"),
    lambda trace: setattr(trace, "replicate", "02"),
    lambda trace: setattr(trace.agent, "model_version", "2"),
    lambda trace: trace.agent.settings.update(tools=["different-tool"]),
    lambda trace: setattr(trace.agent, "instructions_digest", "different"),
])
def test_different_agent_or_trial_conditions_not_paired(suite, change):
    candidate = run(suite, name="knowledge", run_id="candidate")
    change(candidate)
    report = build_report(suite, [run(suite), candidate], CharacterCounter(), baseline="native")
    assert report["comparisons"][0]["reason"] == "no_matching_baseline"


@pytest.mark.parametrize("scenario,reason", [
    ("scripted", "scripted_runs_do_not_measure_agent_performance"),
    ("failed", "both_runs_must_be_accepted"),
    ("no-model", "both_runs_must_be_accepted"),
])
def test_no_misleading_efficiency_claim(suite, scenario, reason):
    baseline = run(suite)
    candidate = run(suite, name="knowledge", run_id="candidate")
    if scenario == "scripted":
        baseline.agent.kind = candidate.agent.kind = "scripted"
    elif scenario == "failed":
        candidate.outcome = "failed"
    else:
        candidate.events.pop()
    pair = build_report(suite, [baseline, candidate], CharacterCounter(), baseline="native")["comparisons"][0]
    assert pair["eligible"] is False
    assert pair["reason"] == reason
    assert pair["token_saving_fraction"] is None


@pytest.mark.parametrize("scenario,error", [
    ("run", "run IDs"), ("policy", "duplicate policy"), ("baseline", "ambiguous"),
])
def test_ambiguous_and_duplicate_pairs_rejected(suite, scenario, error):
    baseline = run(suite)
    other = baseline.model_copy(deep=True)
    if scenario != "run":
        other.run_id = "other"
        other.events[-1].usage_capture.observations[0].id = 'other-fixture-call'
    if scenario == "baseline":
        other.policy.version = "2"
    with pytest.raises(ValueError, match=error):
        build_report(suite, [baseline, other], CharacterCounter(), baseline="native")


@pytest.mark.parametrize('scope', ['legacy', 'cli', 'partial', 'mismatch'])
def test_complete_capture_flags_cannot_override_incomplete_usage_evidence(suite, scope):
    baseline, candidate = run(suite), run(suite,name='knowledge',run_id='candidate')
    event = candidate.events[-1]
    if scope == 'legacy':
        event.usage_capture = None
    elif scope == 'cli':
        event.usage_capture.scope = 'cli_aggregate'
        event.usage_capture.internal_calls_complete = False
    elif scope == 'partial':
        event.usage_capture.expected_calls = 2
    else:
        event.usage = Usage(input_tokens=999,output_tokens=20,cached_input_tokens=60)
    report = build_report(suite,[baseline,candidate],CharacterCounter(),baseline='native')
    assert not report['comparisons'][0]['eligible']
    assert report['comparisons'][0]['token_saving_fraction'] is None
    assert not report['runs'][1]['provider_usage_complete']


@pytest.mark.parametrize("change", [
    lambda data: data.update(extra=True),
    lambda data: data["tasks"][0].update(repository_id="missing"),
    lambda data: data["tasks"].append(deepcopy(data["tasks"][0])),
    lambda data: data["tasks"][0]["criteria"].append(deepcopy(data["tasks"][0]["criteria"][0])),
])
def test_invalid_suite_contracts(suite, change):
    data = suite.model_dump()
    change(data)
    with pytest.raises(ValidationError):
        Suite.model_validate(data)


def test_trace_contract_and_round_trip(suite):
    trace = run(suite)
    restored = Trace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    data = trace.model_dump()
    data["events"][0]["sequence"] = 5
    with pytest.raises(ValidationError, match="ordered"):
        Trace.model_validate(data)
    with pytest.raises(ValidationError, match="subset"):
        Usage(input_tokens=1, output_tokens=1, cached_input_tokens=2)
    with pytest.raises(ValidationError, match="reviewer"):
        Review(verdict="accepted")


def test_recorder_isolates_mutation_and_enforces_lifecycle(suite, tmp_path):
    profile = run(suite)
    ticks = iter([0.0, 0.1])
    recorder = TraceRecorder(
        suite, "overview", run_id="recorded", agent=profile.agent, policy=profile.policy,
        replicate="01", cache_state="cold", repository_revision=suite.repositories[0].revision,
        clock=lambda: next(ticks),
    )
    suite.protocol.max_elapsed_ms = 1.0
    profile.policy.name = "mutated"
    profile.agent.settings["mutated"] = True
    response = {"content": ["original"]}
    recorder.record_tool("read", {}, response, elapsed_ms=2.0)
    response["content"].append("mutated")
    recorder.record_model_turn({"full_request": True}, {"answer": "original"})
    trace = recorder.finish("completed")
    assert trace.outcome == "completed"
    assert trace.policy.name == "native"
    assert trace.agent.settings == {}
    assert trace.events[0].response == {"content": ["original"]}
    assert [event.sequence for event in trace.events] == [0, 1]
    with pytest.raises(ValueError, match="already finished"):
        recorder.finish("completed")
    with pytest.raises(ValueError, match="already finished"):
        recorder.record_tool("late", {}, {}, elapsed_ms=0.0)
    output = tmp_path / "nested" / "trace.json"
    write_trace(trace, output)
    assert Trace.model_validate_json(output.read_text()) == trace
    with pytest.raises(FileExistsError):
        write_trace(trace, output)


@pytest.mark.parametrize("limit", ["time", "tools", "turns"])
def test_recorder_reports_exceeded_protocol(suite, limit):
    trace = run(suite)
    ticks = iter([0.0, 2.0 if limit == "time" else 0.1])
    recorder = TraceRecorder(
        suite, "overview", run_id="limited", agent=trace.agent, policy=trace.policy,
        replicate="01", cache_state="cold", repository_revision=suite.repositories[0].revision,
        clock=lambda: next(ticks),
    )
    for _ in range(3 if limit != "time" else 0):
        if limit == "tools":
            recorder.record_tool("read", {}, {}, elapsed_ms=0.0)
        else:
            recorder.record_model_turn({}, {})
    result = recorder.finish("completed")
    assert result.outcome == "limit_exceeded"
    assert score_run(suite, result, CharacterCounter())["status"] == "limit_exceeded"


def test_explicit_tokenizer_counts_full_payload():
    counter = PayloadCounter("cl100k_base")
    assert counter({"text": "hello", "source": "README.md"}) > counter("hello")
    assert counter({"literal": "<|endoftext|>", "unicode": "नमस्ते"}) > 0
    assert counter.description["encoding"] == "cl100k_base"
    assert counter.description["basis"] == "canonical_serialized_payload_estimate"
    assert counter.description["version"]


def test_cli_report_and_refuse_overwrite(suite, tmp_path, monkeypatch):
    monkeypatch.setattr("code_mem_graph.cli.PayloadCounter", lambda encoding: CharacterCounter())
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite.model_dump()), encoding="utf-8")
    trace_path = tmp_path / "runs" / "run.json"
    write_trace(run(suite), trace_path)
    args = ["evaluate", "--suite", str(suite_path), "--traces", str(trace_path.parent),
            "--encoding", "explicit-test-encoding", "--baseline", "native"]
    runner = CliRunner()
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["status_counts"] == {"accepted": 1}
    output = tmp_path / "reports" / "report.json"
    assert runner.invoke(main, args + ["--output", str(output)]).exit_code == 0
    original = output.read_bytes()
    assert runner.invoke(main, args + ["--output", str(output)]).exit_code != 0
    assert output.read_bytes() == original
    trace_path.unlink()
    assert "no trace JSON" in runner.invoke(main, args).output
    trace_path.write_text("not JSON", encoding="utf-8")
    assert runner.invoke(main, args).exit_code != 0


def test_onboarding_corpus_is_valid_and_has_six_categories():
    root = Path(__file__).resolve().parents[2]
    suite = Suite.model_validate(yaml.safe_load(
        (root / "tests/fixtures/evaluation/repository-onboarding.yaml").read_text(encoding="utf-8")))
    assert len({task.category for task in suite.tasks}) == 6
    for task in suite.tasks:
        for criterion in task.criteria:
            assert all((root / ref).exists() for ref in criterion.source_refs)

import asyncio
import json

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.analysis import analyze_experiment
from code_mem_graph.evaluation.experiment import Experiment, prepare_experiment, save_experiment
from code_mem_graph.evaluation.models import AgentProfile, CaptureInfo, Suite, Usage
from code_mem_graph.evaluation.report import score_run
from code_mem_graph.evaluation.runner import ModelReply, RunnerLimits, TrialSession, TrialState, run_trial, save_checkpoint
from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.serialization import serialize
from code_mem_graph.tokens import PayloadCounter


@pytest.fixture
def experiment(tmp_path):
    (tmp_path / 'README.md').write_text('A fictional trial fixture.\n', encoding='utf-8')
    (tmp_path / 'app.py').write_text('def health():\n    return "ok"\n', encoding='utf-8')
    build_repository(tmp_path, repository_id='trial-fixture')
    suite = Suite.model_validate({'schema_version': 1, 'id': 'trial-suite',
        'protocol': {'id': 'trial-protocol', 'max_tool_calls': 5, 'max_model_turns': 3, 'max_elapsed_ms': 10000.0,
                     'stopping_rule': 'Stop on answer or ceilings.'},
        'repositories': [{'id': 'fixture', 'revision': 'old'}],
        'tasks': [{'id': 'overview', 'repository_id': 'fixture', 'category': 'orientation',
                   'prompt': 'Explain the fixture.', 'acceptance_checks': {},
                   'criteria': [{'id': 'purpose', 'description': 'Private rubric.', 'critical': True, 'source_refs': ['README.md']}]}]})
    profile = AgentProfile(kind='scripted', provider='fake-transport', model='fake', model_version='1',
                           instructions_digest='fixture', settings={})
    value = prepare_experiment(tmp_path, suite, task_id='overview', mode='orient', encoding='cl100k_base',
                               ceiling=10000, split='calibration', replicates=1, cache_state='cold', agent=profile)
    data = value.model_dump(mode='json')
    data['candidates'] = data['candidates'][-1:]
    return Experiment.model_validate(data)


@pytest.fixture
def limits():
    return RunnerLimits(max_model_turns=3, max_tool_calls=5, max_elapsed_ms=10000.0,
                        max_request_tokens=12000, max_response_tokens=2000, max_visible_tokens=40000)


def capture(partial=False):
    return CaptureInfo(request_capture='visible_only' if partial else 'complete',
        provider_usage_available=not partial, model_version_pinned=not partial, host_tools_enforced=not partial,
        provider_token_budget_enforced=False, limitations=['Fake backend or explicitly partial bridge.'])


class FakeTransport:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, request, *, max_output_tokens):
        self.requests.append(request)
        assert max_output_tokens == 2000
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return ModelReply(content=json.dumps(value), usage=Usage(input_tokens=100, output_tokens=20, cached_input_tokens=0))


def tool(name='read', arguments=None):
    return {'tool_calls': [{'id': 'request-1', 'name': name, 'arguments': arguments or {'path': 'README.md'}}], 'final': None}


def final():
    return {'tool_calls': [], 'final': 'A fictional trial fixture [README.md:1].'}


@pytest.mark.asyncio
async def test_full_loop_uses_only_frozen_sources_and_records_repeated_context(experiment, limits, tmp_path):
    (tmp_path / 'README.md').write_text('Changed live tree after capture.', encoding='utf-8')
    backend = FakeTransport([tool(), final()])
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(), backend)
    assert trace.outcome == 'completed' and len(backend.requests) == 2
    assert [entry.kind for entry in trace.events] == ['model', 'tool', 'model']
    assert trace.events[1].response['text'] == 'A fictional trial fixture.'
    assert 'Changed live tree' not in json.dumps(backend.requests)
    assert backend.requests[1]['messages'][:1] == backend.requests[0]['messages']
    scored = score_run(experiment.suite, trace, PayloadCounter('cl100k_base'))
    assert scored['reported_model_total_tokens'] == 240
    assert scored['tool_calls'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('name,args', [('shell', {'command': 'anything'}), ('read', {'path': '../outside'}),
                                     ('read', {'path': 'not-catalogued.py'}), ('read', {'path': 'README.md', 'start_line': True})])
async def test_unknown_tools_paths_and_invalid_arguments_never_execute(experiment, limits, name, args):
    backend = FakeTransport([tool(name, args), final()])
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(), backend)
    assert trace.outcome == 'completed'
    assert 'error' in trace.events[1].response
    assert len(backend.requests) == 2


@pytest.mark.asyncio
async def test_model_turn_and_native_call_limits_stop_loops(experiment, limits):
    changed = limits.model_copy(update={'max_model_turns': 1})
    backend = FakeTransport([tool(), final()])
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], changed, capture(), backend)
    assert trace.outcome == 'limit_exceeded' and trace.termination_reason == 'model_turns'
    assert len(backend.requests) == 1
    changed = limits.model_copy(update={'max_tool_calls': 1})
    many = {'tool_calls': [{'id': str(index), 'name': 'list', 'arguments': {}} for index in range(2)], 'final': None}
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], changed, capture(), FakeTransport([many]))
    assert trace.outcome == 'limit_exceeded'
    assert sum(entry.kind == 'tool' for entry in trace.events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('setting', [{'max_request_tokens': 1}, {'max_visible_tokens': 100}])
async def test_reservations_refuse_a_call_before_dispatch(experiment, limits, setting):
    backend = FakeTransport([final()])
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits.model_copy(update=setting), capture(), backend)
    assert trace.outcome == 'limit_exceeded'
    assert not backend.requests and not trace.events


@pytest.mark.asyncio
async def test_transport_failure_and_cancellation_preserve_attempt_with_unknown_response(experiment, limits):
    for error in [RuntimeError('backend failed'), asyncio.CancelledError()]:
        trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(True), FakeTransport([error]))
        assert trace.outcome in {'failed', 'aborted'}
        assert len(trace.events) == 1 and not trace.events[0].response_observed
        assert trace.events[0].usage is None
        scored = score_run(experiment.suite, trace, PayloadCounter('cl100k_base'))
        assert scored['model_output_tokens_estimate'] is None
        assert scored['model_total_tokens_estimate'] is None
        assert scored['reported_model_total_tokens'] is None


@pytest.mark.asyncio
async def test_timeout_cancels_transport_and_retains_missing_usage(experiment, limits):
    class Slow:
        async def complete(self, request, *, max_output_tokens):
            await asyncio.sleep(2)
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits.model_copy(update={'max_elapsed_ms': 10.0}),
                            capture(True), Slow(), clock=lambda:0.0)
    # Freeze setup accounting so machine load cannot exhaust the 10 ms ceiling
    # before dispatch. asyncio still enforces the actual transport timeout.
    assert trace.outcome == 'limit_exceeded' and trace.termination_reason == 'transport_timeout'
    assert len(trace.events) == 1 and trace.events[0].usage is None


@pytest.mark.asyncio
async def test_expired_preflight_does_not_launch_a_model(experiment, limits):
    clock = [0.0]

    class Transport:
        def prepare(self, request, *, max_output_tokens):
            clock[0] = limits.max_elapsed_ms / 1000

        async def complete(self, request, *, max_output_tokens):
            pytest.fail('no model should launch after the preflight exhausted time')

    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(), Transport(), clock=lambda:clock[0])
    assert trace.termination_reason == 'elapsed_time_before_dispatch'
    assert [e.kind for e in trace.events] == ['control']


@pytest.mark.asyncio
async def test_oversized_replies_and_tool_results_are_explicitly_bounded(experiment, limits):
    class Oversized:
        async def complete(self, request, *, max_output_tokens):
            assert max_output_tokens == 10
            return ModelReply(content=json.dumps({'tool_calls': [], 'final': 'long reply ' * 100}))
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'],
        limits.model_copy(update={'max_response_tokens': 10}), capture(True), Oversized())
    assert trace.outcome == 'limit_exceeded' and trace.events[0].response
    probe = TrialSession.start(experiment, experiment.schedule()[0]['run_id'], limits, capture())
    changed = limits.model_copy(update={'max_request_tokens': PayloadCounter('cl100k_base')(probe.next_request()) + 1})
    backend = FakeTransport([tool('read', {'path': '.project-knowledge/map.yaml'}), final()])
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], changed, capture(), backend)
    assert trace.events[1].response['error'] == 'tool_response_exceeds_remaining_capacity'


def test_batch_reserves_space_for_later_capacity_errors(experiment, limits):
    batch = {'tool_calls': [{'id': str(i), 'name': 'read', 'arguments': {'path': '.project-knowledge/map.yaml'}}
                            for i in range(3)], 'final': None}
    probe = TrialSession.start(experiment, experiment.schedule()[0]['run_id'], limits, capture())
    probe.next_request()
    probe.accept(ModelReply(content=json.dumps(batch)), dispatch_id=probe.state.dispatch_id)
    request = probe.next_request()
    # Reproduce a boundary where one large result fits, but adding even the
    # explicit errors for the remaining calls would exceed request capacity.
    results = json.loads(request['messages'][-1]['content'])
    request['messages'][-1]['content'] = serialize(results[:1])
    cap = PayloadCounter('cl100k_base')(request) + 1
    bounded = TrialSession.start(experiment, experiment.schedule()[0]['run_id'],
                                limits.model_copy(update={'max_request_tokens': cap}), capture())
    assert bounded.next_request() is not None
    bounded.accept(ModelReply(content=json.dumps(batch)), dispatch_id=bounded.state.dispatch_id)
    following = bounded.next_request()
    assert following is not None
    assert PayloadCounter('cl100k_base')(following) <= cap
    assert bounded.state.events[1]['response']['next_start_line'] is not None


@pytest.mark.asyncio
async def test_scoped_listing_and_duplicate_reads_reuse_retained_evidence(experiment, limits):
    batch = {'tool_calls': [
        {'id': 'root', 'name': 'list', 'arguments': {}},
        {'id': 'scoped', 'name': 'list', 'arguments': {'prefix': '.project-knowledge', 'limit': 1}},
        {'id': 'read', 'name': 'read', 'arguments': {'path': 'README.md'}},
        {'id': 'repeat', 'name': 'read', 'arguments': {'path': 'README.md', 'start_line': 1}},
    ], 'final': None}
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(), FakeTransport([batch, final()]))
    root, scoped, original, repeat = [event.response for event in trace.events if event.kind == 'tool']
    assert any(entry['kind'] == 'directory' and entry['path'] == '.project-knowledge' for entry in root['entries'])
    assert not any(entry['path'] == '.project-knowledge/map.yaml' for entry in root['entries'])
    assert scoped['total'] == 2 and len(scoped['entries']) == 1 and scoped['next_offset'] == 1
    assert 'fixture' in original['text'] and repeat['duplicate_of'] == 3
    assert 'text' not in repeat


@pytest.mark.asyncio
async def test_search_pagination_is_scoped_and_cursors_are_explicit(experiment, limits):
    batch = {'tool_calls': [
        {'id': 'page', 'name': 'search', 'arguments': {'query': 'fixture', 'prefix': '.project-knowledge', 'limit': 1}},
        {'id': 'readme', 'name': 'search', 'arguments': {'query': 'fixture', 'prefix': 'README.md'}},
        {'id': 'next', 'name': 'list', 'arguments': {'prefix': '.project-knowledge', 'offset': 1}},
    ], 'final': None}
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(), FakeTransport([batch, final()]))
    page, readme, following = [event.response for event in trace.events if event.kind == 'tool']
    assert len(page['entries']) == 1 and page['total'] > 1 and page['next_offset'] == 1
    assert all(entry['path'].startswith('.project-knowledge/') for entry in page['entries'])
    assert readme['total'] == 1 and readme['entries'][0]['path'] == 'README.md'
    assert following['offset'] == 1 and following['next_offset'] is None


def test_checkpoint_replay_dispatch_binding_and_latency_freeze(experiment, limits, tmp_path):
    clock = [0.0]
    session = TrialSession.start(experiment, experiment.schedule()[0]['run_id'], limits, capture(True), clock=lambda: clock[0])
    request = session.next_request()
    assert request == session.next_request()
    checkpoint = tmp_path / 'checkpoint.json'
    save_checkpoint(session.state, checkpoint)
    restored = TrialSession(experiment, TrialState.model_validate_json(checkpoint.read_text()), clock=lambda: clock[0])
    assert restored.next_request() == request
    with pytest.raises(ValueError, match='matching pending'):
        restored.accept(ModelReply(content=json.dumps(final())), dispatch_id='wrong-dispatch')
    restored.accept(ModelReply(content=json.dumps(final())), dispatch_id=restored.state.dispatch_id)
    clock[0] = 1000.0
    assert restored.trace().elapsed_ms == 0
    with pytest.raises(FileExistsError):
        save_checkpoint(restored.state, checkpoint)
    with pytest.raises(ValueError, match='matching pending'):
        restored.accept(ModelReply(content=json.dumps(final())), dispatch_id='late')


def test_checkpoint_freezes_prompt_and_exposes_remaining_limits(experiment, limits, monkeypatch):
    import code_mem_graph.evaluation.runner as module
    session = TrialSession.start(experiment, experiment.schedule()[0]['run_id'], limits, capture())
    original_system, original_tools = session.state.system, session.state.tools.copy()
    monkeypatch.setattr(module, 'SYSTEM', 'changed after checkpoint')
    monkeypatch.setattr(module, 'TOOLS', [])
    restored = TrialSession(experiment, TrialState.model_validate_json(session.state.model_dump_json()))
    request = restored.next_request()
    assert request['system'] == original_system and request['tools'] == original_tools
    assert request['remaining']['model_turns'] == limits.max_model_turns
    restored.accept(ModelReply(content=json.dumps(tool('list', {}))), dispatch_id=restored.state.dispatch_id)
    following = restored.next_request()
    assert following['remaining']['model_turns'] == limits.max_model_turns - 1
    assert following['remaining']['native_tool_calls'] == limits.max_tool_calls - 1


@pytest.mark.asyncio
@pytest.mark.parametrize('content', ['not JSON', '{}', '{"tool_calls":[],"final":""}',
                                   '{"tool_calls":[],"final":null}'])
async def test_invalid_model_outputs_fail_without_synthetic_success(experiment, limits, content):
    class Invalid:
        async def complete(self, request, *, max_output_tokens):
            return ModelReply(content=content)
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, capture(True), Invalid())
    assert trace.outcome == 'failed' and trace.termination_reason == 'invalid_model_response'
    assert trace.events[0].response == content


@pytest.mark.asyncio
async def test_partial_host_capture_is_never_eligible_for_calibration(experiment, limits):
    traces = []
    for scheduled in experiment.schedule():
        trace = await run_trial(experiment, scheduled['run_id'], limits, capture(True), FakeTransport([final()]))
        traces.append(trace)
    report = analyze_experiment(experiment, traces)
    assert 'only_visible_requests_captured' in report['readiness_reasons']
    assert 'provider_usage_unavailable' in report['readiness_reasons']
    assert 'host_tool_isolation_not_enforced' in report['readiness_reasons']
    assert not report['comparisons'][0]['eligible_for_agent_comparison']
    assert report['default_selected'] is None


def test_cli_external_bridge_preserves_checkpoints_and_outputs_trace(experiment, limits, tmp_path):
    experiment_path = tmp_path / 'experiment.json'
    save_experiment(experiment, experiment_path)
    limits_path, capture_path = tmp_path / 'limits.json', tmp_path / 'capture.json'
    limits_path.write_text(limits.model_dump_json(), encoding='utf-8')
    capture_path.write_text(capture(True).model_dump_json(), encoding='utf-8')
    state0, state1 = tmp_path / 'state0.json', tmp_path / 'state1.json'
    runner = CliRunner()
    result = runner.invoke(main, ['trial', 'start', str(experiment_path), '--run-id', experiment.schedule()[0]['run_id'],
        '--limits', str(limits_path), '--capture', str(capture_path), '--output', str(state0)])
    assert result.exit_code == 0, result.output
    dispatched = json.loads(result.output)['dispatch_id']
    request = runner.invoke(main, ['trial', 'request', str(state0)])
    assert request.exit_code == 0 and json.loads(request.output)['dispatch_id'] == dispatched
    reply = tmp_path / 'reply.json'
    reply.write_text(json.dumps(final()), encoding='utf-8')
    trace_path = tmp_path / 'trace.json'
    result = runner.invoke(main, ['trial', 'step', str(experiment_path), '--state', str(state0), '--dispatch-id', dispatched,
        '--reply', str(reply), '--output', str(state1), '--trace', str(trace_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['outcome'] == 'completed'
    assert trace_path.exists() and state0.exists() and state1.exists()


def test_cli_invalid_utf8_is_retained_as_failed_unknown_response(experiment, limits, tmp_path):
    experiment_path = tmp_path / 'experiment.json'
    save_experiment(experiment, experiment_path)
    session = TrialSession.start(experiment, experiment.schedule()[0]['run_id'], limits, capture(True))
    session.next_request()
    state0 = tmp_path / 'state0.json'
    save_checkpoint(session.state, state0)
    reply = tmp_path / 'corrupt.json'
    raw = b'{"tool_calls":[],"final":"bad\xe9"}'
    reply.write_bytes(raw)
    state1, trace_path = tmp_path / 'state1.json', tmp_path / 'trace.json'
    result = CliRunner().invoke(main, ['trial', 'step', str(experiment_path), '--state', str(state0),
        '--dispatch-id', session.state.dispatch_id, '--reply', str(reply), '--output', str(state1), '--trace', str(trace_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['outcome'] == 'failed'
    assert json.loads(trace_path.read_text())['events'][0]['response_observed'] is False
    assert reply.read_bytes() == raw

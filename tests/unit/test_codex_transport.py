import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.codex_transport import (
    CodexDescriptor, CodexExecTransport, CodexTransportConfig, DISABLED_FEATURES, PROMPT, _parse_events, describe_codex, response_schema,
)
from code_mem_graph.evaluation.experiment import Experiment, prepare_experiment, save_experiment
from code_mem_graph.evaluation.models import AgentProfile, CaptureInfo, Suite, Trace
from code_mem_graph.evaluation.runner import RunnerLimits, run_trial
from code_mem_graph.evaluation.controls import ResourceLimits
from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.serialization import serialize


@pytest.fixture
def config():
    return CodexTransportConfig(model='gpt-6-luna', timeout_ms=5000.0, max_stdout_bytes=1000000, max_stderr_bytes=100000)


@pytest.fixture
def descriptor(config):
    return CodexDescriptor(sys.executable, 'fixture-cli', AgentProfile(kind='scripted', provider='fixture', model=config.model,
        model_version='fixture', instructions_digest='fixture', settings={}),
        CaptureInfo(request_capture='visible_only', provider_usage_available=False, model_version_pinned=False,
                    host_tools_enforced=False, provider_token_budget_enforced=False))


@pytest.fixture
def experiment(tmp_path, descriptor):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'README.md').write_text('A transport fixture.\n', encoding='utf-8')
    build_repository(root, repository_id='transport-fixture')
    suite = Suite.model_validate({'schema_version': 1, 'id': 'transport-suite',
        'protocol': {'id': 'transport-protocol', 'max_tool_calls': 5, 'max_model_turns': 3, 'max_elapsed_ms': 10000.0,
                     'stopping_rule': 'Final answer or limits.'},
        'repositories': [{'id': 'fixture', 'revision': 'placeholder'}],
        'tasks': [{'id': 'overview', 'repository_id': 'fixture', 'category': 'orientation', 'prompt': 'Explain purpose.',
                   'criteria': [{'id': 'purpose', 'description': 'Cite purpose.', 'critical': True, 'source_refs': ['README.md']}],
                   'acceptance_checks': {}}]})
    prepared = prepare_experiment(root, suite, task_id='overview', mode='orient', encoding='cl100k_base', ceiling=20000,
                                  split='calibration', replicates=1, cache_state='cold', agent=descriptor.profile)
    data = prepared.model_dump(mode='json')
    data['candidates'] = data['candidates'][-1:]
    return Experiment.model_validate(data)


@pytest.fixture
def limits():
    return RunnerLimits(max_model_turns=3, max_tool_calls=5, max_elapsed_ms=10000.0,
                        max_request_tokens=20000, max_response_tokens=2000, max_visible_tokens=60000)


@pytest.fixture
def helper(tmp_path):
    script = tmp_path / 'fake_cli.py'
    script.write_text('''import json,sys,time
mode=sys.argv[1]
prompt=sys.stdin.read()
request=json.loads(prompt.split("TRIAL_REQUEST_JSON:\\n",1)[1])
def emit(event): print(json.dumps(event),flush=True)
emit({"type":"thread.started","thread_id":"fixture-thread"})
emit({"type":"turn.started"})
if mode=="hang": time.sleep(30)
if mode=="invalid_stream": print("not json",flush=True);sys.exit(0)
if mode=="invalid_utf8": sys.stdout.buffer.write(b"\\xff\\n");sys.stdout.buffer.flush();sys.exit(0)
if mode=="overflow": print("x"*10000,flush=True);sys.exit(0)
if mode=="stderr_overflow": sys.stderr.write("x"*10000);sys.stderr.flush();sys.exit(0)
if mode=="turn_failed": emit({"type":"turn.failed","error":{"message":"fixture failure"}});sys.exit(0)
if mode=="native_tool": emit({"type":"item.completed","item":{"id":"cmd","type":"command_execution"}})
has_tool=any(m["role"]=="tool" for m in request["messages"])
answer={"tool_calls":[],"final":"A transport fixture [README.md:1]."} if has_tool else {"tool_calls":[{"id":"read-1","name":"read","arguments":{"path":"README.md","start_line":1,"end_line":1}}],"final":None}
text="not protocol JSON" if mode=="malformed_reply" else json.dumps(answer)
if mode!="missing_reply": emit({"type":"item.completed","item":{"id":"reply","type":"agent_message","text":text}})
if mode!="missing_completion": emit({"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20,"cached_input_tokens":10}})
if mode=="nonzero": sys.exit(7)
''', encoding='utf-8')
    return script


def transport(tmp_path, config, descriptor, helper, monkeypatch, mode='normal'):
    adapter = CodexExecTransport(config, tmp_path / 'artifacts', descriptor=descriptor)
    monkeypatch.setattr(adapter, '_command', lambda work, schema: [sys.executable, str(helper), mode])
    return adapter


@pytest.mark.asyncio
async def test_direct_delivery_runs_tools_without_reply_files(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    adapter = transport(tmp_path, config, descriptor, helper, monkeypatch)
    checkpoints = []
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, descriptor.capture, adapter,
                            checkpoint=lambda state: checkpoints.append(state.model_copy(deep=True)))
    assert trace.outcome == 'completed'
    assert [event.kind for event in trace.events] == ['model', 'tool', 'model']
    turns = [event for event in trace.events if event.kind == 'model']
    assert len([path for path in adapter.artifact_dir.iterdir() if path.is_dir()]) == len(turns) == 2
    assert all(event.transport.response_source == 'codex_jsonl.item.completed.agent_message' for event in turns)
    assert all(event.transport.response_item_id == 'reply' and event.usage.input_tokens == 100 for event in turns)
    assert all(event.transport.failure_reason is None for event in turns)
    assert trace.capture.provider_usage_available is False  # Aggregate reporting is not full T02 accounting.
    assert checkpoints[-1].outcome == 'completed' and checkpoints[0].pending_request
    assert not list(adapter.artifact_dir.rglob('reply*.json'))
    for event in turns:
        directory = Path(event.transport.artifact_path)
        assert (directory / 'stdin.txt').exists() and (directory / 'delivery.json').exists()
        assert (directory / 'stdin.txt').read_bytes() == (PROMPT + serialize(event.request)).encode('utf-8')
        assert json.loads((directory / 'delivery.json').read_text())['attempt_id'] == event.transport.attempt_id


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,reason,observed', [
    ('invalid_stream', 'invalid_jsonl_event_stream', False), ('invalid_utf8', 'invalid_jsonl_event_stream', False),
    ('missing_reply', 'missing_model_reply', False), ('turn_failed', 'cli_error_event', False),
    ('missing_completion', 'missing_or_ambiguous_turn_completion', True), ('nonzero', 'nonzero_process_exit', True),
    ('native_tool', 'unexpected_host_tool_activity', None), ('malformed_reply', 'invalid_model_response', True),
])
async def test_delivery_failures_retain_evidence_without_retry(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch,
                                                              mode, reason, observed):
    adapter = transport(tmp_path, config, descriptor, helper, monkeypatch, mode)
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, descriptor.capture, adapter)
    assert trace.outcome == 'failed' and trace.termination_reason == reason
    assert len([path for path in adapter.artifact_dir.iterdir() if path.is_dir()]) == 1
    event = trace.events[0]
    if observed is not None:
        assert event.response_observed is observed
    assert event.transport
    assert Path(event.transport.artifact_path, 'stdout.jsonl').exists()
    if observed is False and mode != 'missing_reply':
        assert event.usage is None


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,field,reason', [('overflow', 'max_stdout_bytes', 'stdout_byte_limit'),
                                            ('stderr_overflow', 'max_stderr_bytes', 'stderr_byte_limit')])
async def test_output_caps_are_explicit_retained_prefixes(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch,
                                                        mode, field, reason):
    adapter = transport(tmp_path, config.model_copy(update={field: 1000}), descriptor, helper, monkeypatch, mode)
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, descriptor.capture, adapter)
    assert trace.outcome == 'limit_exceeded' and trace.termination_reason == reason
    record = trace.events[0].transport
    assert record.output_truncated
    assert record.stdout_bytes <= 1000 if field == 'max_stdout_bytes' else record.stderr_bytes <= 1000


@pytest.mark.asyncio
async def test_timeout_and_cancellation_retain_attempt_and_stop_child(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    adapter = transport(tmp_path, config.model_copy(update={'timeout_ms': 200.0}), descriptor, helper, monkeypatch, 'hang')
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, descriptor.capture, adapter)
    assert trace.outcome == 'limit_exceeded' and trace.termination_reason == 'transport_timeout'
    assert trace.events[0].transport.exit_code is not None
    other = CodexExecTransport(config, tmp_path / 'cancel-artifacts', descriptor=descriptor)
    monkeypatch.setattr(other, '_command', lambda work, schema: [sys.executable, str(helper), 'hang'])
    task = asyncio.create_task(run_trial(experiment, experiment.schedule()[0]['run_id'], limits, descriptor.capture, other))
    await asyncio.sleep(.2)
    task.cancel()
    cancelled = await task
    assert cancelled.outcome == 'aborted' and cancelled.termination_reason == 'cancelled'
    assert cancelled.events[0].transport.exit_code is not None
    assert Path(cancelled.events[0].transport.artifact_path, 'delivery.json').exists()


@pytest.mark.asyncio
async def test_launch_failure_retains_unknown_response(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    adapter = transport(tmp_path, config, descriptor, helper, monkeypatch)
    monkeypatch.setattr(adapter, '_command', lambda work, schema: [str(tmp_path / 'nonexistent.exe')])
    trace = await run_trial(experiment, experiment.schedule()[0]['run_id'], limits, descriptor.capture, adapter)
    assert trace.termination_reason == 'process_or_artifact_io_failure'
    assert not trace.events[0].response_observed and trace.events[0].usage is None
    assert trace.events[0].transport.exit_code is None


def test_schema_and_command_use_explicit_model_and_no_adapter_reply_file(config, descriptor, tmp_path):
    adapter = CodexExecTransport(config, tmp_path / 'artifacts', descriptor=descriptor)
    command = adapter._command(tmp_path / 'work', tmp_path / 'schema.json')
    assert command[command.index('--model') + 1] == 'gpt-6-luna'
    assert '--output-last-message' not in command and '--json' in command and '--output-schema' in command
    assert '--ignore-user-config' in command and '--ephemeral' in command
    assert 'model_providers.code_mem_graph_trial.request_max_retries=0' in command
    assert 'model_providers.code_mem_graph_trial.stream_max_retries=0' in command
    assert 'model_provider="code_mem_graph_trial"' in command
    assert not any('model_providers.openai.' in argument for argument in command)
    assert command[-1] == '-'
    schema = response_schema()
    for variant in schema['properties']['tool_calls']['items']['anyOf']:
        args = variant['properties']['arguments']
        assert set(args['required']) == set(args['properties']) and args['additionalProperties'] is False


def test_event_decoder_preserves_final_origin_and_rejects_ambiguity():
    def stream(items, usage=None):
        return '\n'.join(json.dumps(event) for event in [*items, {'type': 'turn.completed', 'usage': usage or {}}]).encode()

    def message(text, item_id, phase=None):
        return {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text, 'id': item_id, 'phase': phase}}

    final = '{"tool_calls":[],"final":"cited answer"}'
    result = _parse_events(stream([message('commentary', 'comment'), message(final, 'final')]))
    assert result[:2] == (final, 'final') and result[3] is None
    result = _parse_events(stream([message(final, 'final', 'final_answer'), message('commentary', 'comment', 'commentary')]))
    assert result[:2] == (final, 'final')
    result = _parse_events(stream([message(final, 'a', 'final_answer'), message(final, 'b', 'final_answer')]))
    assert result[3] == 'ambiguous_final_reply'
    assert _parse_events(stream([message(final, 42)]))[3] == 'invalid_jsonl_event_stream'
    result = _parse_events(stream([message(final, 'final')], {'input_tokens': -1, 'output_tokens': 1, 'cached_input_tokens': 0}))
    assert result[2] is None and result[3] is None


def test_describe_checks_capabilities_without_model_dispatch(config, monkeypatch):
    calls = []
    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.shutil.which', lambda name: sys.executable)

    def probe(command, **kwargs):
        calls.append(command)
        from types import SimpleNamespace
        output = (b'fixture-cli' if command[-1] == '--version' else
                  '\n'.join(f'{name} stable true' for name in DISABLED_FEATURES).encode() if command[-1] == 'list' else
                  b'--json --output-schema --ignore-user-config --ephemeral')
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.subprocess.run', probe)
    described = describe_codex(config)
    assert len(calls) == 3 and all('--model' not in call for call in calls)
    assert described.profile.model == config.model and described.capture.request_capture == 'visible_only'
    assert not described.capture.model_version_pinned and not described.capture.host_tools_enforced


def test_cli_unattended_run_and_existing_output_are_not_reused(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    config_path, limits_path, exp_path = tmp_path / 'config.json', tmp_path / 'limits.json', tmp_path / 'experiment.json'
    config_path.write_text(config.model_dump_json(), encoding='utf-8')
    limits_path.write_text(limits.model_dump_json(), encoding='utf-8')
    save_experiment(experiment, exp_path)
    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.describe_codex', lambda cfg: descriptor)
    monkeypatch.setattr(CodexExecTransport, '_command', lambda self, work, schema: [sys.executable, str(helper), 'normal'])
    output = tmp_path / 'run'
    args = ['trial', 'run', str(exp_path), '--run-id', experiment.schedule()[0]['run_id'], '--limits', str(limits_path),
            '--transport-config', str(config_path), '--output', str(output)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    trace = Trace.model_validate_json((output / 'trace.json').read_text(encoding='utf-8'))
    assert trace.outcome == 'completed' and len(list((output / 'checkpoints').glob('*.json'))) == 4
    original = (output / 'trace.json').read_bytes()
    assert CliRunner().invoke(main, args).exit_code != 0
    assert (output / 'trace.json').read_bytes() == original
    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.describe_codex', lambda cfg: replace(descriptor,
        profile=descriptor.profile.model_copy(update={'model': 'changed'})))
    result = CliRunner().invoke(main, args[:-1] + [str(tmp_path / 'wrong-profile')])
    assert result.exit_code != 0 and 'differs' in result.output
    assert not (tmp_path / 'wrong-profile').exists()


def test_cli_failed_delivery_still_exports_trace(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    exp_path, config_path, limits_path = tmp_path / 'experiment.json', tmp_path / 'config.json', tmp_path / 'limits.json'
    save_experiment(experiment, exp_path)
    config_path.write_text(config.model_dump_json(), encoding='utf-8')
    limits_path.write_text(limits.model_dump_json(), encoding='utf-8')
    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.describe_codex', lambda cfg: descriptor)
    monkeypatch.setattr(CodexExecTransport, '_command', lambda self, work, schema: [sys.executable, str(helper), 'missing_reply'])
    output = tmp_path / 'failed'
    result = CliRunner().invoke(main, ['trial', 'run', str(exp_path), '--run-id', experiment.schedule()[0]['run_id'],
        '--limits', str(limits_path), '--transport-config', str(config_path), '--output', str(output)])
    assert result.exit_code == 1, result.output
    trace = Trace.model_validate_json((output / 'trace.json').read_text(encoding='utf-8'))
    assert trace.outcome == 'failed' and trace.termination_reason == 'missing_model_reply'
    assert trace.events[0].transport.response_source == 'none'
    assert trace.events[0].usage.input_tokens == 100  # Provider observation retained even without final text.


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,resources,reason,outcome', [
    ('strict', ResourceLimits(), 'strict_profile_unsupported', 'aborted'),
    ('functional', ResourceLimits(max_stdin_bytes=1), 'stdin_byte_limit', 'limit_exceeded'),
])
async def test_preflight_blocks_before_any_model_attempt(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch,
                                                       mode, resources, reason, outcome):
    cfg=config.model_copy(update={'execution_mode':mode,'resource_limits':resources})
    adapter=transport(tmp_path,cfg,descriptor,helper,monkeypatch)
    monkeypatch.setattr(adapter,'_command',lambda *args: pytest.fail('blocked dispatch must not launch'))
    trace=await run_trial(experiment,experiment.schedule()[0]['run_id'],limits,descriptor.capture,adapter)
    assert trace.outcome==outcome and trace.termination_reason==reason
    assert len(trace.events)==1 and trace.events[0].kind=='control'
    assert not [p for p in adapter.artifact_dir.iterdir() if p.is_dir()]
    assert json.loads(adapter.controls.path.read_text().splitlines()[0])['decision']=='blocked'


@pytest.mark.asyncio
async def test_dispatch_quota_stops_next_call_with_a_trace_decision(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    adapter=transport(tmp_path,config.model_copy(update={'resource_limits':ResourceLimits(max_dispatches=1)}),descriptor,helper,monkeypatch)
    trace=await run_trial(experiment,experiment.schedule()[0]['run_id'],limits,descriptor.capture,adapter)
    assert trace.termination_reason=='transport_dispatch_limit' and trace.outcome=='limit_exceeded'
    assert [e.kind for e in trace.events]==['model','tool','control']
    assert adapter.controls.dispatches==1 and trace.events[0].usage.input_tokens==100


@pytest.mark.asyncio
async def test_reported_token_overshoot_is_retained_and_stops_tools(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    adapter=transport(tmp_path,config.model_copy(update={'resource_limits':ResourceLimits(max_observed_cli_tokens=119)}),descriptor,helper,monkeypatch)
    trace=await run_trial(experiment,experiment.schedule()[0]['run_id'],limits,descriptor.capture,adapter)
    assert trace.outcome=='limit_exceeded' and trace.termination_reason=='observed_cli_token_threshold_exceeded'
    assert [e.kind for e in trace.events]==['model'] and trace.events[0].response_observed
    assert adapter.controls.cli_tokens==120 and trace.events[0].usage.output_tokens==20
    assert trace.capture.provider_token_budget_enforced is False


@pytest.mark.asyncio
async def test_total_retained_stream_budget_is_shared_across_calls(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    quota=600
    adapter=transport(tmp_path,config.model_copy(update={'resource_limits':ResourceLimits(max_total_retained_stdout_bytes=quota)}),
                      descriptor,helper,monkeypatch)
    trace=await run_trial(experiment,experiment.schedule()[0]['run_id'],limits,descriptor.capture,adapter)
    assert trace.outcome=='limit_exceeded' and trace.termination_reason=='stdout_byte_limit'
    assert sum(e.transport.stdout_bytes for e in trace.events if e.kind=='model')<=quota
    assert adapter.controls.stdout_bytes<=quota


@pytest.mark.asyncio
async def test_host_activity_is_stopped_before_later_child_side_effect(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    sentinel=tmp_path/'should-not-exist.txt'
    script=tmp_path/'host_activity.py'
    script.write_text('import sys,json,time\nfrom pathlib import Path\nsys.stdin.read()\n'
        'print(json.dumps({"type":"item.started","item":{"id":"cmd","type":"command_execution"}}),flush=True)\n'
        'time.sleep(2)\nPath(sys.argv[1]).write_text("unexpected side effect")\n',encoding='utf-8')
    adapter=transport(tmp_path,config,descriptor,helper,monkeypatch)
    monkeypatch.setattr(adapter,'_command',lambda work,schema:[sys.executable,str(script),str(sentinel)])
    trace=await run_trial(experiment,experiment.schedule()[0]['run_id'],limits,descriptor.capture,adapter)
    assert trace.termination_reason=='unexpected_host_tool_activity'
    assert not sentinel.exists() and trace.events[0].transport.exit_code is not None
    assert trace.capture.host_tools_enforced is False  # Reactive cancellation is not complete removal.


def test_cli_feature_disabling_and_temp_working_directory(config, descriptor, tmp_path):
    adapter=CodexExecTransport(config,tmp_path/'artifacts',descriptor=descriptor)
    command=adapter._command(tmp_path/'work',tmp_path/'schema.json')
    assert 'web_search="disabled"' in command
    requested={command[i+1] for i,value in enumerate(command) if value=='--disable'}
    assert requested==set(DISABLED_FEATURES)


def test_capability_report_discloses_unsupported_strict_guarantees(config, descriptor, tmp_path, monkeypatch):
    path=tmp_path/'transport.json'
    path.write_text(config.model_dump_json(),encoding='utf-8')
    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.describe_codex',lambda cfg:descriptor)
    result=CliRunner().invoke(main,['trial','capabilities','--transport-config',str(path)])
    assert result.exit_code==0,result.output
    report=json.loads(result.output)
    assert not report['capabilities']['strict_calibration_supported']
    assert not report['capture']['host_tools_enforced'] and not report['capture']['provider_token_budget_enforced']

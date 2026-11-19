import json
import sqlite3
import subprocess
import sys

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.codex_transport import CodexExecTransport
from code_mem_graph.evaluation.controls import ResourceLimits
from code_mem_graph.evaluation.coordination import (
    RunJournal, execute_coordinated, load_registration, make_registration, register_cohort, run_lock,
)
from code_mem_graph.evaluation.experiment import save_experiment
from code_mem_graph.evaluation.models import Trace
from code_mem_graph.evaluation.report import score_run
from code_mem_graph.tokens import PayloadCounter

# Reuse the synthetic CLI/snapshot fixture. These tests never contact a model.
from tests.unit.test_codex_transport import config as config, descriptor as descriptor, experiment as experiment, helper as helper, limits as limits


class Crash(BaseException):
    pass


@pytest.fixture
def registered(experiment, limits, config, descriptor, helper, tmp_path, monkeypatch):
    registration = make_registration(experiment, limits, config, descriptor)
    cohort = tmp_path/'cohort'
    register_cohort(cohort,registration)
    calls = []

    def command(self,work,schema):
        calls.append(str(schema.parent))
        return [sys.executable,str(helper),'normal']

    monkeypatch.setattr(CodexExecTransport,'_command',command)
    monkeypatch.setattr('code_mem_graph.evaluation.coordination.describe_codex',lambda cfg:descriptor)
    return registration,cohort,calls


async def start(registered, tmp_path, descriptor, *, arm=0, name='run'):
    registration,cohort,_ = registered
    return await execute_coordinated(tmp_path/name,registration=registration,
        run_id=registration.experiment.schedule()[arm]['run_id'],cohort=cohort,descriptor=descriptor)


@pytest.mark.asyncio
async def test_shared_registration_duplicate_claims_terminal_resume_and_costs(registered,tmp_path,descriptor):
    trace = await start(registered,tmp_path,descriptor)
    assert trace.outcome=='completed' and not trace.coordination['deviations']
    assert len(registered[2])==2
    assert all(row['provider_latency_ms'] is None for row in trace.coordination['dispatches'])
    original = (tmp_path/'run'/'trace.json').read_bytes()
    restored = await execute_coordinated(tmp_path/'run',resume=True)
    assert restored == trace and len(registered[2])==2
    assert (tmp_path/'run'/'trace.json').read_bytes()==original
    with pytest.raises(ValueError,match='already claimed'):
        await start(registered,tmp_path,descriptor,name='duplicate')
    assert not (tmp_path/'duplicate').exists()
    other = await start(registered,tmp_path,descriptor,arm=1,name='assisted')
    assert other.coordination['registration_digest']==trace.coordination['registration_digest']
    report = json.loads((tmp_path/'run'/'accounting-report.json').read_text())
    assert report['categories']['evaluation']['observed_model_tokens']==240


@pytest.mark.asyncio
async def test_export_interruption_resumes_from_terminal_journal_without_calls(registered,tmp_path,descriptor,monkeypatch):
    import code_mem_graph.evaluation.coordination as module
    original = module._publish

    def fail(path,content):
        if path.name=='usage-ledger.json':
            raise OSError('synthetic export interruption')
        original(path,content)

    monkeypatch.setattr(module,'_publish',fail)
    with pytest.raises(OSError,match='export interruption'):
        await start(registered,tmp_path,descriptor)
    monkeypatch.setattr(module,'_publish',original)
    trace = await execute_coordinated(tmp_path/'run',resume=True)
    assert trace.outcome=='completed' and len(registered[2])==2
    assert (tmp_path/'run'/'accounting-report.json').is_file()


@pytest.mark.asyncio
async def test_resume_after_result_before_checkpoint_reuses_delivery_and_restores_quota(registered,tmp_path,descriptor,monkeypatch):
    original = RunJournal.checkpoint

    def crash(self,state):
        if state.events and state.outcome is None:
            raise Crash()
        original(self,state)

    monkeypatch.setattr(RunJournal,'checkpoint',crash)
    with pytest.raises(Crash):
        await start(registered,tmp_path,descriptor)
    assert len(registered[2])==1
    monkeypatch.setattr(RunJournal,'checkpoint',original)
    trace = await execute_coordinated(tmp_path/'run',resume=True)
    assert trace.outcome=='completed' and len(registered[2])==2
    assert len([e for e in trace.events if e.kind=='model'])==2
    journal = RunJournal(tmp_path/'run')
    try:
        state = journal.result(journal.rows()[-1])['control_snapshot']
        assert state['dispatches']==2 and state['observed_cli_tokens']==240
    finally:
        journal.close()


@pytest.mark.asyncio
async def test_restored_dispatch_ceiling_prevents_a_second_launch(registered,tmp_path,descriptor,monkeypatch):
    registration,_,calls = registered
    limited = registration.model_copy(update={'config':registration.config.model_copy(
        update={'resource_limits':ResourceLimits(max_dispatches=1)})})
    cohort = tmp_path/'limited-cohort'
    register_cohort(cohort,limited)
    original = RunJournal.delivered

    def crash(self,*args,**kwargs):
        original(self,*args,**kwargs)
        raise Crash()

    monkeypatch.setattr(RunJournal,'delivered',crash)
    with pytest.raises(Crash):
        await execute_coordinated(tmp_path/'limited-run',registration=limited,cohort=cohort,
            run_id=limited.experiment.schedule()[0]['run_id'],descriptor=descriptor)
    monkeypatch.setattr(RunJournal,'delivered',original)
    trace = await execute_coordinated(tmp_path/'limited-run',resume=True)
    assert trace.termination_reason=='transport_dispatch_limit' and len(calls)==1
    assert trace.events[0].usage.input_tokens==100


@pytest.mark.asyncio
async def test_expired_checkpoint_still_reconciles_previously_claimed_usage(registered,tmp_path,descriptor,monkeypatch):
    original = RunJournal.checkpoint

    def crash(self,state):
        if state.events and state.outcome is None:
            raise Crash()
        original(self,state)

    monkeypatch.setattr(RunJournal,'checkpoint',crash)
    with pytest.raises(Crash):
        await start(registered,tmp_path,descriptor)
    monkeypatch.setattr(RunJournal,'checkpoint',original)
    journal = RunJournal(tmp_path/'run')
    try:
        state = journal.state()
        state.started_at-=3600
        journal.checkpoint(state)
    finally:
        journal.close()
    trace = await execute_coordinated(tmp_path/'run',resume=True)
    assert trace.outcome=='limit_exceeded' and len(registered[2])==1
    assert len(trace.events)==1 and trace.events[0].usage.input_tokens==100


@pytest.mark.asyncio
async def test_receipt_recovery_after_crash_preserves_usage_without_resending(registered,tmp_path,descriptor,monkeypatch):
    original = RunJournal.delivered
    monkeypatch.setattr(RunJournal,'delivered',lambda *args,**kwargs:(_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        await start(registered,tmp_path,descriptor)
    assert len(registered[2])==1
    monkeypatch.setattr(RunJournal,'delivered',original)
    trace = await execute_coordinated(tmp_path/'run',resume=True)
    assert trace.outcome=='completed' and len(registered[2])==2
    assert trace.events[0].usage.input_tokens==100
    assert 'delivery_recovered_after_interruption' in trace.coordination['deviations']
    scored = score_run(registered[0].experiment.suite,trace,PayloadCounter('cl100k_base'))
    assert 'delivery_recovered_after_interruption' in scored['coordination_blockers']


@pytest.mark.asyncio
async def test_ambiguous_claim_is_aborted_and_retained_as_unknown_cost(registered,tmp_path,descriptor,monkeypatch):
    original = CodexExecTransport.complete

    async def crash(*args,**kwargs):
        raise Crash()

    monkeypatch.setattr(CodexExecTransport,'complete',crash)
    with pytest.raises(Crash):
        await start(registered,tmp_path,descriptor)
    monkeypatch.setattr(CodexExecTransport,'complete',original)
    trace = await execute_coordinated(tmp_path/'run',resume=True)
    assert trace.outcome=='aborted' and trace.termination_reason=='dispatch_delivery_uncertain'
    assert registered[2]==[]
    assert len(trace.events)==1 and trace.events[0].kind=='model'
    assert trace.events[0].usage is None and trace.events[0].response_observed is False
    assert 'dispatch_delivery_uncertain' in trace.coordination['deviations']
    report = json.loads((tmp_path/'run'/'accounting-report.json').read_text())
    assert report['categories']['evaluation']['records']==1
    assert report['categories']['evaluation']['observed_model_tokens'] is None


@pytest.mark.asyncio
async def test_raw_receipt_mismatch_stops_recovery_without_retry(registered,tmp_path,descriptor,monkeypatch):
    original = RunJournal.delivered
    monkeypatch.setattr(RunJournal,'delivered',lambda *args,**kwargs:(_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        await start(registered,tmp_path,descriptor)
    monkeypatch.setattr(RunJournal,'delivered',original)
    stdout = next((tmp_path/'run'/'transport').glob('*/stdout.jsonl'))
    stdout.write_bytes(stdout.read_bytes()+b'\n')
    trace = await execute_coordinated(tmp_path/'run',resume=True)
    assert trace.outcome=='aborted' and len(registered[2])==1
    assert trace.events[0].usage.input_tokens==100  # Available counters survive uncertainty.
    assert trace.events[0].usage_capture.issues


@pytest.mark.asyncio
async def test_corrupt_durable_controls_do_not_reset_quota(registered,tmp_path,descriptor,monkeypatch):
    original = RunJournal.checkpoint

    def crash(self,state):
        if state.events and state.outcome is None:
            raise Crash()
        original(self,state)

    monkeypatch.setattr(RunJournal,'checkpoint',crash)
    with pytest.raises(Crash):
        await start(registered,tmp_path,descriptor)
    monkeypatch.setattr(RunJournal,'checkpoint',original)
    with sqlite3.connect(tmp_path/'run'/'journal.sqlite3') as db:
        db.execute("UPDATE dispatches SET controls='{}'")
    with pytest.raises(ValueError,match='snapshot checksum'):
        await execute_coordinated(tmp_path/'run',resume=True)
    assert len(registered[2])==1


def test_run_lock_excludes_other_processes_and_releases(tmp_path):
    lock = tmp_path/'run.lock'
    script = "import site; site.addsitedir(r'.venv/Lib/site-packages'); from pathlib import Path; from code_mem_graph.evaluation.coordination import run_lock;\nwith run_lock(Path(__import__('sys').argv[1])): print('acquired')"
    with run_lock(lock):
        child = subprocess.run([sys.executable,'-c',script,str(lock)],capture_output=True,text=True)
        assert child.returncode!=0 and 'another coordinator' in child.stderr
    child = subprocess.run([sys.executable,'-c',script,str(lock)],capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    crashed = subprocess.run([sys.executable,'-c',script.replace("print('acquired')","__import__('os')._exit(0)"),str(lock)],capture_output=True,text=True)
    assert crashed.returncode==0,crashed.stderr
    with run_lock(lock):
        pass  # OS releases a crashed owner's lock without deleting a lockfile.


@pytest.mark.asyncio
async def test_registration_drift_and_legacy_runs_fail_before_calls(registered,tmp_path,descriptor):
    registration,cohort,calls = registered
    changed = registration.model_copy(update={'limits':registration.limits.model_copy(update={'max_model_turns':2})})
    with pytest.raises(ValueError,match='settings differ'):
        await execute_coordinated(tmp_path/'changed',registration=changed,
            run_id=registration.experiment.schedule()[0]['run_id'],cohort=cohort,descriptor=descriptor)
    assert not calls
    legacy = tmp_path/'legacy'
    legacy.mkdir()
    with pytest.raises(ValueError,match='legacy'):
        await execute_coordinated(legacy,resume=True)
    assert load_registration(cohort)==registration


def test_cli_register_and_resume_share_durable_workflow(registered,tmp_path,descriptor,monkeypatch):
    registration,cohort,calls = registered
    exp_path,limits_path,config_path = [tmp_path/name for name in ('experiment.json','limits.json','config.json')]
    save_experiment(registration.experiment,exp_path)
    limits_path.write_text(registration.limits.model_dump_json())
    config_path.write_text(registration.config.model_dump_json())
    monkeypatch.setattr('code_mem_graph.evaluation.codex_transport.describe_codex',lambda cfg:descriptor)
    runner = CliRunner()
    second = tmp_path/'second-cohort'
    args = [str(exp_path),'--limits',str(limits_path),'--transport-config',str(config_path)]
    result = runner.invoke(main,['trial','register',*args,'--output',str(second)])
    assert result.exit_code==0,result.output
    output = tmp_path/'cli-run'
    result = runner.invoke(main,['trial','run',*args,'--run-id',registration.experiment.schedule()[0]['run_id'],
        '--registration',str(second),'--output',str(output)])
    assert result.exit_code==0,result.output
    result = runner.invoke(main,['trial','resume',str(output)])
    assert result.exit_code==0,result.output
    assert len(calls)==2
    trace = Trace.model_validate_json((output/'trace.json').read_text())
    assert trace.coordination['registration_digest']==load_registration(second).digest

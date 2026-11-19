import json
from pathlib import Path
import shutil
import sys

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.acceptance import (
    AcceptanceCheck,AcceptancePlan,CandidatePatch,register_acceptance,run_acceptance,
)
from code_mem_graph.evaluation.models import AgentProfile,ModelEvent,PolicyProfile,Suite,Trace
from code_mem_graph.evaluation.report import score_run
from code_mem_graph.evaluation.validation import grade_audit,make_audit,reviewed_trace,review_context_digest
from code_mem_graph.knowledge.build import build_repository
from tests.review_fixtures import supported_audit
from tests.unit.test_knowledge_evaluation import CharacterCounter


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'source'
    shutil.copytree(Path(__file__).parents[1]/'fixtures/validation/pricing',root)
    build_repository(root,repository_id='pricing-fixture')
    return root


def suite():
    return Suite.model_validate({'schema_version':1,'id':'change-fixture',
        'protocol':{'id':'fixture','max_model_turns':2,'max_tool_calls':2,'max_elapsed_ms':20000.0,'stopping_rule':'Fixture only.'},
        'repositories':[{'id':'pricing','revision':'unbound'}],
        'tasks':[{'id':'discount','repository_id':'pricing','category':'change','prompt':'Fix discount subtraction.',
          'criteria':[{'id':'fix','description':'Subtract the discount.','critical':True,'source_refs':['pricing.py']}],
          'acceptance_checks':{'unit-tests':'The protected regression must pass on the candidate.'}}]})


def plan(argv=None,**kwargs):
    return AcceptancePlan(allowed_changes=['pricing.py'],trusted_files=['tests/test_pricing.py'],
        checks=[AcceptanceCheck(id='unit-tests',argv=argv or [sys.executable,'-S','-m','unittest','discover','-s','tests','-v'],timeout_ms=kwargs.pop('timeout_ms',5000.0))],
        max_elapsed_ms=10000.0,max_stdout_bytes=kwargs.pop('stdout_cap',10000),max_stderr_bytes=10000,**kwargs)


def trace(registration):
    return Trace(schema_version=1,run_id='change-1',suite_digest=registration.suite.digest,task_id='discount',
        repository_revision='snapshot:'+registration.snapshot.digest,replicate='1',cache_state='cold',
        agent=AgentProfile(kind='scripted',provider='fixture',model='authored',model_version='1',instructions_digest='fixture',settings={}),
        policy=PolicyProfile(name='fixture',version='1',parameters={}),
        events=[ModelEvent(sequence=0,request={'kind':'authored fixture'},response={'tool_calls':[],
            'final':'The candidate subtracts the discount [pricing.py:1-2].'})],elapsed_ms=1.0,outcome='completed')


@pytest.mark.asyncio
async def test_bad_and_fixed_candidates_are_executed_separately_without_changing_source(source,tmp_path):
    registration=register_acceptance(source,suite(),'discount',plan(),tmp_path/'registration')
    before={str(path.relative_to(source)):path.read_bytes() for path in source.rglob('*') if path.is_file()}
    original=(source/'pricing.py').read_text(encoding='utf-8')
    for label,content,outcome in [('bug',original,'failed'),('fix','def total(price, discount):\n    return price - discount\n','passed')]:
        patch=CandidatePatch(base_snapshot_digest=registration.snapshot.digest,changes={'pricing.py':content})
        receipt=await run_acceptance(registration,patch,'change-1',tmp_path/label)
        assert receipt.outcome==outcome and receipt.checks[0].executed
        assert (tmp_path/label/'checkout/pricing.py').read_text(encoding='utf-8')==content
        change={'registration':registration.model_dump(mode='json'),'candidate':patch.model_dump(mode='json'),'receipt':receipt.model_dump(mode='json')}
        authored=trace(registration)
        audit=supported_audit(registration.suite,authored,registration.snapshot,change=change)
        summary=grade_audit(registration.suite,authored,audit)
        assert summary['change_execution']==('passed' if outcome=='passed' else 'not_verified')
        result=score_run(registration.suite,reviewed_trace(registration.suite,authored,audit),CharacterCounter())
        assert result['accepted']==(outcome=='passed')
        if outcome=='passed':
            altered=audit.model_copy(deep=True)
            altered.change['receipt']['elapsed_ms']+=1.0
            with pytest.raises(ValueError,match='context'):
                grade_audit(registration.suite,authored,altered)
            change['candidate']['changes']['pricing.py']='def total(price, discount):\n    return 0\n'
            rebound=make_audit(registration.suite,registration.snapshot,authored,change=change)
            # A different candidate cannot reuse this successful execution receipt.
            rebound.reviews=[audit.reviews[0].model_copy(update={'snapshot_digest':rebound.proof.snapshot_digest,
                                                               'context_digest':review_context_digest(rebound)})]
            assert 'change_execution_binding_mismatch' in grade_audit(registration.suite,authored,rebound)['blockers']
    assert before=={str(path.relative_to(source)):path.read_bytes() for path in source.rglob('*') if path.is_file()}


def test_review_without_execution_never_proves_a_change(source,tmp_path):
    registration=register_acceptance(source,suite(),'discount',plan(),tmp_path/'registration')
    authored=trace(registration)
    audit=supported_audit(registration.suite,authored,registration.snapshot)
    assert 'change_execution_evidence_missing' in grade_audit(registration.suite,authored,audit)['blockers']


@pytest.mark.asyncio
async def test_scope_and_trusted_tests_cannot_be_modified(source,tmp_path):
    registration=register_acceptance(source,suite(),'discount',plan(),tmp_path/'registration')
    patch=CandidatePatch(base_snapshot_digest=registration.snapshot.digest,changes={'tests/test_pricing.py':'pass\n'})
    with pytest.raises(ValueError,match='base/scope'):
        await run_acceptance(registration,patch,'change-1',tmp_path/'attempt')
    assert not (tmp_path/'attempt').exists()
    with pytest.raises(ValueError,match='distinct'):
        AcceptancePlan(allowed_changes=['tests/test_pricing.py'],trusted_files=['tests/test_pricing.py'],checks=plan().checks,
                       max_elapsed_ms=10000.0,max_stdout_bytes=10000,max_stderr_bytes=10000)


@pytest.mark.asyncio
@pytest.mark.parametrize('argv,settings,error',[
    (['-S','-c','import time;time.sleep(5)'],{'timeout_ms':100.0},'timeout'),
    (['-S','-c','print("x"*100000)'],{'stdout_cap':100},'stdout_byte_limit'),
])
async def test_time_and_output_failures_keep_evidence(source,tmp_path,argv,settings,error):
    registration=register_acceptance(source,suite(),'discount',plan([sys.executable,*argv],**settings),tmp_path/'registration')
    patch=CandidatePatch(base_snapshot_digest=registration.snapshot.digest,changes={'pricing.py':(source/'pricing.py').read_text()})
    receipt=await run_acceptance(registration,patch,'change-1',tmp_path/'attempt')
    assert receipt.outcome=='failed' and receipt.checks[0].error==error
    assert (tmp_path/'attempt/receipt.json').is_file()


@pytest.mark.asyncio
async def test_even_a_zero_exit_cannot_hide_checkout_mutation(source,tmp_path):
    registration=register_acceptance(source,suite(),'discount',plan([sys.executable,'-S','-c',
        'from pathlib import Path;Path("extra.py").write_text("changed")']),tmp_path/'registration')
    patch=CandidatePatch(base_snapshot_digest=registration.snapshot.digest,changes={'pricing.py':(source/'pricing.py').read_text()})
    receipt=await run_acceptance(registration,patch,'change-1',tmp_path/'attempt')
    assert receipt.checks[0].exit_code==0 and receipt.outcome=='failed'
    assert 'extra.py' in receipt.observed_files


def test_cli_registration_never_executes_commands(source,tmp_path):
    suite_path=tmp_path/'suite.json'
    suite_path.write_text(suite().model_dump_json(),encoding='utf-8')
    plan_path=tmp_path/'plan.json'
    plan_path.write_text(plan().model_dump_json(),encoding='utf-8')
    output=tmp_path/'registered'
    result=CliRunner().invoke(main,['acceptance','register','--root',str(source),'--suite',str(suite_path),
        '--task-id','discount','--plan',str(plan_path),'--output',str(output)])
    assert result.exit_code==0,result.output
    assert json.loads(result.output)['executed_checks']==0
    assert list(output.iterdir())==[output/'registration.json']

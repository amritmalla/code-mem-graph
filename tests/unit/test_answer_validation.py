import json

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.models import Review
from code_mem_graph.evaluation.report import build_report,score_run
from code_mem_graph.evaluation.validation import (
    AnswerAudit,grade_audit,make_audit,reviewed_trace,validation_summary,review_context_digest,
)
from tests.review_fixtures import source_snapshot,supported_audit
from tests.unit.test_knowledge_evaluation import CharacterCounter,run,suite as suite


@pytest.mark.parametrize('citation,status',[
    ('README.md:1','valid'),('README.md:1-1','valid'),('README.md#L1-L1','valid'),
    ('README.md:L1','valid'),('README.md:0','invalid_line_range'),('README.md:1-2','invalid_line_range'),
    ('README.md:2-1','invalid_line_range'),('README.md:bad','malformed_reference'),
    ('missing.py:1','outside_frozen_catalog'),('../README.md:1','unsafe_path'),
    ('%2e%2e/README.md#L1','unsafe_path'),('C:\\repo\\README.md:1','unsafe_path'),
    ('https://example.test/README.md#L1','unsafe_path'),
])
def test_citation_addresses_are_checked_against_exact_frozen_sources(suite,citation,status):
    trace=run(suite)
    trace.events[-1].response['final']='A project ['+citation+'].'
    audit=make_audit(suite,source_snapshot(),trace)
    assert len(audit.citations)==1 and audit.citations[0].status==status
    assert audit.citations[0].excerpt==('A fixture project.' if status=='valid' else None)


def test_citation_validity_never_substitutes_for_semantic_review(suite):
    trace=run(suite)
    trace.events[-1].response['final']='The CTO approves all releases [README.md:1].'
    audit=supported_audit(suite,trace,source_snapshot())
    assert audit.citations[0].status=='valid'
    audit.reviews[0].units[0].verdict='unsupported'
    audit.reviews[0].units[0].note='The introduction names no release approver.'
    trace.review.validation=audit.model_dump(mode='json')
    result=score_run(suite,trace,CharacterCounter())
    assert result['accepted'] is False and 'unsupported_or_uncertain_answer_unit' in result['validation_blockers']
    assert trace.review.verdict=='accepted'  # An optimistic outer label cannot bypass the audit.


def test_omitted_critical_guidance_missing_units_and_disagreements_block_acceptance(suite):
    trace=run(suite)
    audit=AnswerAudit.model_validate(trace.review.validation)
    audit.reviews[0].criteria[0].status='not_met'
    assert 'critical_or_required_criterion_unmet' in grade_audit(suite,trace,audit)['blockers']
    audit=AnswerAudit.model_validate(trace.review.validation)
    audit.reviews[0].units=[]
    assert 'review_coverage_incomplete' in grade_audit(suite,trace,audit)['blockers']
    audit=AnswerAudit.model_validate(trace.review.validation)
    dissent=audit.reviews[0].model_copy(deep=True)
    dissent.reviewer.reference='independent-second-opinion'
    dissent.units[0].verdict='unsupported'
    audit.reviews.append(dissent)
    summary=grade_audit(suite,trace,audit)
    assert summary['status']=='disputed' and summary['disagreements']
    assert reviewed_trace(suite,trace,audit).review.verdict=='rejected'


@pytest.mark.parametrize('mutation',[
    lambda audit:audit.units[0].__setattr__('text','Different statement'),
    lambda audit:audit.citations[0].__setattr__('excerpt','Invented source text'),
    lambda audit:audit.proof.sources[0].__setattr__('content','Different source bytes'),
    lambda audit:audit.reviews[0].__setattr__('answer_digest','b'*64),
])
def test_mutated_audit_evidence_cannot_be_reused(suite,mutation):
    trace=run(suite)
    audit=AnswerAudit.model_validate(trace.review.validation)
    mutation(audit)
    trace.review.validation=audit.model_dump(mode='json')
    assert validation_summary(suite,trace)['status']=='invalid'


def test_legacy_accepted_labels_and_answer_edits_do_not_qualify(suite):
    baseline=run(suite)
    candidate=run(suite,name='knowledge',run_id='candidate')
    candidate.review.validation=None
    assert 'validation' not in candidate.review.model_dump(mode='json')
    result=build_report(suite,[baseline,candidate],CharacterCounter(),baseline='native')
    assert result['runs'][1]['status']=='unverified'
    assert result['comparisons'][0]['eligible'] is False
    assert 'answer_validation_undeclared' in result['comparisons'][0]['capture_blockers']
    candidate=run(suite,name='knowledge',run_id='candidate')
    candidate.events[-1].response['final']+=' Edited after review.'
    assert validation_summary(suite,candidate)['status']=='invalid'


def test_no_review_does_not_silently_become_a_pass(suite):
    trace=run(suite)
    trace.review=Review()
    audit=make_audit(suite,source_snapshot(),trace)
    summary=grade_audit(suite,trace,audit)
    assert summary['status']=='unreviewed' and 'semantic_review_missing' in summary['blockers']
    assert reviewed_trace(suite,trace,audit).review.verdict=='unreviewed'


def test_review_cannot_transfer_to_a_different_rubric_even_with_same_answer_and_sources(suite):
    trace=run(suite)
    original=AnswerAudit.model_validate(trace.review.validation)
    data=suite.model_dump(mode='json')
    data['tasks'][0]['criteria'][0]['description']='A different critical question under the same criterion ID.'
    changed=type(suite).model_validate(data)
    trace.suite_digest=changed.digest
    audit=make_audit(changed,source_snapshot(),trace,reviews=original.reviews)
    with pytest.raises(ValueError,match='rubric context'):
        grade_audit(changed,trace,audit)


def test_context_unbound_reviews_remain_readable_but_cannot_verify(suite):
    trace=run(suite)
    audit=AnswerAudit.model_validate(trace.review.validation)
    audit.reviews[0].context_digest=None
    assert 'context_digest' not in audit.reviews[0].model_dump(mode='json')
    assert 'review_context_unbound' in grade_audit(suite,trace,audit)['blockers']


def test_discovery_cannot_launder_an_overlay_under_change_metadata(suite):
    trace=run(suite)
    audit=AnswerAudit.model_validate(trace.review.validation)
    audit.change={}
    audit.proof.policy_digest='b'*64
    audit.reviews[0].snapshot_digest=audit.proof.snapshot_digest
    audit.reviews[0].context_digest=review_context_digest(audit)
    with pytest.raises(ValueError,match='discovery proof'):
        grade_audit(suite,trace,audit)
    with pytest.raises(ValueError,match='discovery reviews'):
        make_audit(suite,source_snapshot(),trace,change={})


def test_factual_units_require_valid_citations_even_with_positive_review(suite):
    trace=run(suite)
    trace.events[-1].response['final']='A project without any source location.'
    audit=supported_audit(suite,trace,source_snapshot())
    assert 'factual_unit_without_valid_evidence' in grade_audit(suite,trace,audit)['blockers']


def test_audit_serialization_does_not_change_legacy_review_shape():
    original={'verdict':'unreviewed','reviewer':None,'criteria':{},'acceptance_checks':{},'unsupported_high_impact_claims':[]}
    assert Review.model_validate(original).model_dump(mode='json')==original


def test_review_packet_withholds_previous_opinions_and_costs(tmp_path,monkeypatch,suite):
    trace=run(suite)
    trace_path=tmp_path/'trace.json'
    trace_path.write_text(trace.model_dump_json(),encoding='utf-8')
    monkeypatch.setattr('code_mem_graph.cli._review_inputs',lambda *args:(suite,source_snapshot()))
    output=tmp_path/'packet.json'
    result=CliRunner().invoke(main,['review','packet','--experiment',str(trace_path),'--trace',str(trace_path),'--output',str(output)])
    assert result.exit_code==0,result.output
    packet=json.loads(output.read_text(encoding='utf-8'))
    assert packet['audit']['reviews']==[]
    assert 'agent' not in packet and 'policy' not in packet and 'usage' not in packet
    opinion=tmp_path/'opinion.json'
    opinion.write_text(AnswerAudit.model_validate(trace.review.validation).reviews[0].model_dump_json(),encoding='utf-8')
    # Attach to an unreviewed original, so the same reviewer is not duplicated.
    original=trace.model_copy(deep=True)
    original.review=Review()
    trace_path.write_text(original.model_dump_json(),encoding='utf-8')
    before=trace_path.read_bytes()
    derived=tmp_path/'derived.json'
    result=CliRunner().invoke(main,['review','attach','--experiment',str(trace_path),'--trace',str(trace_path),
        '--review',str(opinion),'--output',str(derived)])
    assert result.exit_code==0,result.output
    assert trace_path.read_bytes()==before and derived.is_file()

from datetime import date
import json

from click.testing import CliRunner
import pytest

from code_mem_graph.cli import main
from code_mem_graph.evaluation.accounting import (
    CATEGORIES, CostLedger, CostRecord, PriceCard, capture_codex_usage, ledger_from_traces, ledger_report,
    merge_ledgers, normalize_codex_counter, usage_diagnostics,
)
from code_mem_graph.evaluation.models import AgentProfile, ModelEvent, PolicyProfile, Trace, Usage, UsageCapture, UsageObservation


def usage(**changes):
    values=dict(input_tokens=100,output_tokens=20,cached_input_tokens=40,cache_write_input_tokens=10,reasoning_output_tokens=5)
    values.update(changes)
    return Usage.model_validate(values)


def capture(value=None, *, call='call-1', expected=1, complete=True):
    return UsageCapture(scope='provider_calls',source='fictional test provider',expected_calls=expected,
        internal_calls_complete=complete,observations=[UsageObservation(id=call,reference='fixture response',
            outcome='completed',usage=value if value is not None else usage())])


def record(**changes):
    values=dict(id='record-1',category='evaluation',provider='fixture',model='fictional',reference='fixture response',
                outcome='completed',usage=usage(),usage_capture=capture(),billing_mode='api',price_card_id='fixture-rates')
    values.update(changes)
    return CostRecord.model_validate(values)


def price(**changes):
    values=dict(id='fixture-rates',version='fictional-v1',source='fixture://rates-not-real-prices',effective_date=date(2026,11,17),
                provider='fixture',model='fictional',currency='USD',input_per_million='2',cached_input_per_million='0.5',
                output_per_million='8',cache_write_policy='separate_rate',cache_write_per_million='3.5')
    values.update(changes)
    return PriceCard.model_validate(values)


def test_inclusive_totals_and_exact_cache_write_pricing():
    ledger=CostLedger(records=[record()],prices=[price()],coverage={key:'complete' if key=='evaluation' else 'not_applicable' for key in CATEGORIES})
    report=ledger_report(ledger)
    assert report['categories']['evaluation']['complete_provider_tokens']==120
    assert report['records'][0]['observed_token_rate_estimate']=='0.000315'
    assert report['complete_estimated_cost']=='0.000315'  # No cached or reasoning double count.
    assert report['prices'][0]['version']=='fictional-v1' and report['prices'][0]['effective_date']=='2026-11-17'
    ordinary=CostLedger(records=[record()],prices=[price(cache_write_policy='standard_input',cache_write_per_million=None)])
    assert ledger_report(ordinary)['records'][0]['observed_token_rate_estimate']=='0.0003'


def test_unknown_costs_categories_and_subscription_are_not_zero():
    report=ledger_report(CostLedger(records=[record(price_card_id=None,billing_mode='subscription')]))
    assert report['currency'] is None and report['complete_estimated_cost'] is None
    assert report['categories']['review']['coverage']=='not_recorded'
    assert report['categories']['review']['observed_model_tokens'] is None
    assert report['categories']['evaluation']['observed_token_rate_subtotal'] is None
    assert report['records'][0]['pricing_unknown_reason']=='billing_mode_not_token_priced'
    assert ledger_report(CostLedger())['complete_estimated_cost'] is None


def test_failed_and_internal_calls_are_retained_without_double_counting():
    calls=capture()
    calls.expected_calls=2
    calls.observations.append(UsageObservation(id='failed-call',reference='fixture error usage',outcome='failed',
        usage=Usage(input_tokens=70,output_tokens=10,cached_input_tokens=0,cache_write_input_tokens=0,reasoning_output_tokens=0)))
    combined=calls.observed_usage()
    row=record(usage=combined,usage_capture=calls,outcome='failed')
    report=ledger_report(CostLedger(records=[row],prices=[price()],coverage={'evaluation':'complete'}))
    assert report['categories']['evaluation']['complete_provider_tokens']==200
    assert report['records'][0]['usage']['provider_complete']
    assert report['records'][0]['usage']['observation_records'][1]['outcome']=='failed'
    assert report['records'][0]['estimated_total_cost']=='0.000535'


def test_unknown_internal_retry_usage_keeps_known_spend_but_blocks_totals():
    calls=capture(expected=2)
    calls.observations.append(UsageObservation(id='unknown-retry',reference='provider disconnect',outcome='failed',usage=None))
    report=ledger_report(CostLedger(records=[record(usage_capture=calls)],prices=[price()],coverage={'evaluation':'complete'}))
    assert report['records'][0]['observed_token_rate_estimate']=='0.000315'
    assert report['records'][0]['estimated_total_cost'] is None
    assert report['categories']['evaluation']['complete_provider_tokens'] is None
    assert report['categories']['evaluation']['coverage']=='partial'
    assert report['categories']['evaluation']['observed_model_tokens']==120


def test_counter_mismatch_cannot_be_priced_or_treated_as_complete():
    row=record(usage=Usage(input_tokens=999,output_tokens=20,cached_input_tokens=40))
    result=ledger_report(CostLedger(records=[row],prices=[price()],coverage={'evaluation':'complete'}))
    assert result['records'][0]['usage']['observed_usage'] is None
    assert 'usage_reconciliation_mismatch' in result['records'][0]['usage']['issues']
    assert result['records'][0]['observed_token_rate_estimate'] is None


@pytest.mark.parametrize('detail,reason',[('cached_input_tokens','cache_read_usage_unknown'),
                                        ('cache_write_input_tokens','cache_write_usage_unknown')])
def test_missing_cache_bucket_never_becomes_a_free_bucket(detail,reason):
    data=usage().model_dump()
    data.pop(detail)
    tokens=Usage.model_validate(data)
    result=ledger_report(CostLedger(records=[record(usage=tokens,usage_capture=capture(tokens))],prices=[price()]))
    assert result['records'][0]['pricing_unknown_reason']==reason
    assert result['records'][0]['observed_token_rate_estimate'] is None
    assert result['categories']['evaluation']['observed_model_tokens']==120


def test_unknown_usage_does_not_hide_known_category_or_details():
    records=[record(),record(id='unknown',price_card_id=None,usage=None,usage_capture=UsageCapture(scope='provider_calls',
        source='fixture timeout',expected_calls=1,internal_calls_complete=True,observations=[UsageObservation(
            id='timeout',reference='fixture request',outcome='failed',usage=None)]),outcome='failed')]
    result=ledger_report(CostLedger(records=records,prices=[price()],coverage={'evaluation':'complete'}))
    group=result['categories']['evaluation']
    assert group['known_usage_records']==1 and group['provider_incomplete_records']==1
    assert group['observed_model_tokens']==120 and group['estimated_total_cost'] is None


def test_codex_optional_details_and_corrupt_later_output_preserve_counters():
    raw={'input_tokens':100,'output_tokens':20,'cached_input_tokens':40,'cache_write_input_tokens':10,'reasoning_output_tokens':5}
    stream=json.dumps({'type':'turn.completed','usage':raw}).encode()+b'\n\xff\n'
    observed=capture_codex_usage(stream)
    assert observed.observed_usage()==usage()
    assert observed.observations[0].raw_usage==raw
    assert observed.observations[0].reference=='stdout.jsonl:line:1'
    assert 'invalid_event_stream' in observed.issues
    diagnostic=usage_diagnostics(observed.observed_usage(),observed)
    assert not diagnostic['provider_complete'] and not diagnostic['recorded_scope_complete']
    assert diagnostic['observed_usage']['reasoning_output_tokens']==5


def test_multiple_cli_totals_stay_ambiguous_and_never_double_count():
    line=json.dumps({'type':'turn.completed','usage':usage().model_dump()}).encode()
    observed=capture_codex_usage(line+b'\n'+line)
    assert len(observed.observations)==2 and observed.observed_usage() is None
    assert 'ambiguous_cli_usage' in observed.issues
    assert not observed.provider_complete


def test_valid_cli_total_is_not_full_provider_capture():
    observed=capture_codex_usage(json.dumps({'type':'turn.completed','usage':usage().model_dump()}).encode())
    diagnostic=usage_diagnostics(usage(),observed)
    assert diagnostic['recorded_scope_complete'] and not diagnostic['provider_complete']
    assert 'provider_call_usage_unavailable' in diagnostic['blockers']
    assert usage_diagnostics(usage(),None)['scope']=='legacy'
    assert not usage_diagnostics(usage(),None)['provider_complete']


def test_malformed_optional_details_preserve_known_input_and_output():
    observed,issues=normalize_codex_counter({'input_tokens':100,'output_tokens':20,'cached_input_tokens':101,
                                           'reasoning_output_tokens':21})
    assert observed.input_tokens==100 and observed.output_tokens==20 and observed.cached_input_tokens is None
    assert set(issues)=={'invalid_cached_input_tokens','invalid_reasoning_output_tokens'}
    assert normalize_codex_counter({'input_tokens':True,'output_tokens':0})[0] is None
    malformed=capture_codex_usage(b'{"type":"turn.completed","usage":{"input_tokens":NaN,"output_tokens":0}}')
    assert 'invalid_event_stream' in malformed.issues and malformed.observed_usage() is None


@pytest.mark.parametrize('changes',[
    {'input_per_million':'NaN'}, {'output_per_million':'-1'}, {'cached_input_per_million':0.5},
    {'cache_write_per_million':None}, {'cache_write_policy':'standard_input'},
])
def test_rate_cards_require_exact_explicit_nonnegative_rates(changes):
    with pytest.raises(ValueError):
        price(**changes)


def test_duplicate_records_provider_calls_and_mixed_currencies_are_rejected():
    with pytest.raises(ValueError,match='duplicate cost'):
        CostLedger(records=[record(),record()],prices=[price()])
    with pytest.raises(ValueError,match='more than one cost'):
        CostLedger(records=[record(),record(id='other',category='review')],prices=[price()])
    with pytest.raises(ValueError,match='mixed currencies'):
        CostLedger(prices=[price(),price(id='euros',currency='EUR')])
    with pytest.raises(ValueError,match='match the exact'):
        CostLedger(records=[record(model='other-model')],prices=[price()])
    with pytest.raises(ValueError,match='duplicate usage'):
        UsageCapture(scope='provider_calls',source='fixture',observations=[capture().observations[0]]*2)


def test_categories_and_rate_versions_remain_separate_on_merge():
    evaluation=CostLedger(records=[record()],prices=[price()],coverage={'evaluation':'complete'})
    review=CostLedger(records=[record(id='review',category='review',usage_capture=capture(call='review-call'))],prices=[price()])
    combined=merge_ledgers([evaluation,review])
    result=ledger_report(combined)
    assert result['categories']['evaluation']['records']==1 and result['categories']['review']['records']==1
    assert result['categories']['build']['observed_model_tokens'] is None
    assert combined.coverage=={} and result['complete_estimated_cost'] is None
    with pytest.raises(ValueError,match='conflicting versions'):
        merge_ledgers([evaluation,CostLedger(prices=[price(version='different')])])


def test_cli_reports_and_merges_without_overwriting(tmp_path):
    path=tmp_path/'ledger.json'
    ledger=CostLedger(records=[record()],prices=[price()])
    path.write_text(ledger.model_dump_json(),encoding='utf-8')
    runner=CliRunner()
    out=tmp_path/'report.json'
    result=runner.invoke(main,['accounting','report',str(path),'--output',str(out)])
    assert result.exit_code==0,result.output
    assert json.loads(result.output)['categories']['evaluation']['observed_model_tokens']==120
    before=out.read_bytes()
    assert runner.invoke(main,['accounting','report',str(path),'--output',str(out)]).exit_code!=0
    assert out.read_bytes()==before
    assert runner.invoke(main,['accounting','merge',str(path),str(path),'--output',str(tmp_path/'duplicate.json')]).exit_code!=0
    assert not (tmp_path/'duplicate.json').exists()


def test_import_traces_requires_explicit_category_and_preserves_pricing_provenance(tmp_path):
    trace=Trace(schema_version=1,run_id='fixture-run',suite_digest='fixture-suite',task_id='fixture-task',
        repository_revision='fixture-revision',replicate='1',cache_state='cold',elapsed_ms=1.0,outcome='failed',
        agent=AgentProfile(kind='scripted',provider='fixture',model='fictional',model_version='fixture',
                           instructions_digest='fixture',settings={}),
        policy=PolicyProfile(name='fixture',version='1',parameters={}),
        events=[ModelEvent(sequence=0,request={},response=None,response_observed=False,usage=usage(),usage_capture=capture())])
    ledger=ledger_from_traces([trace],category='review',billing_mode='api',price=price())
    assert ledger.records[0].category=='review' and ledger.records[0].outcome=='failed'
    trace_path,price_path,out=tmp_path/'trace.json',tmp_path/'price.json',tmp_path/'ledger.json'
    trace_path.write_text(trace.model_dump_json(),encoding='utf-8')
    price_path.write_text(price().model_dump_json(),encoding='utf-8')
    args=['accounting','from-traces','--trace',str(trace_path),'--category','review','--billing-mode','api',
          '--price-card',str(price_path),'--output',str(out)]
    result=CliRunner().invoke(main,args)
    assert result.exit_code==0,result.output
    imported=CostLedger.model_validate_json(out.read_text(encoding='utf-8'))
    assert imported==ledger and imported.prices[0].source=='fixture://rates-not-real-prices'
    assert CliRunner().invoke(main,args).exit_code!=0
    with pytest.raises(ValueError,match='more than one model event'):
        ledger_from_traces([trace,trace],category='evaluation')

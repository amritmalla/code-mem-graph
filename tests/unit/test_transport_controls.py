import json

import pytest

from code_mem_graph.evaluation.accounting import capture_codex_usage
from code_mem_graph.evaluation.controls import ControlLedger, DispatchBlocked, ResourceLimits


def usage():
    return capture_codex_usage(json.dumps({'type':'turn.completed','usage':{
        'input_tokens':100,'output_tokens':20,'cached_input_tokens':0}}).encode())


def test_reservation_is_idempotent_until_activation_and_cannot_be_replayed(tmp_path):
    with pytest.raises(ValueError,match='execution mode'):
        ControlLedger(tmp_path/'invalid.jsonl',ResourceLimits(),'typo')
    ledger=ControlLedger(tmp_path/'controls.jsonl',ResourceLimits(max_dispatches=2),'functional')
    first=ledger.reserve('request',50,200,100)
    assert first==ledger.reserve('request',50,200,100) and ledger.dispatches==0
    ledger.activate('request')
    with pytest.raises(DispatchBlocked,match='unsettled'):
        ledger.reserve('request',50,200,100)
    ledger.settle(100,20,usage())
    assert ledger.dispatches==1 and ledger.stdin_bytes==50
    ledger.reserve('request',50,200,100)
    ledger.activate('request')
    ledger.settle(100,20,usage())
    with pytest.raises(DispatchBlocked,match='dispatch_limit'):
        ledger.reserve('request',50,200,100)


def test_exact_input_reservation_and_remaining_stream_quota(tmp_path):
    ledger=ControlLedger(tmp_path/'controls.jsonl',ResourceLimits(max_stdin_bytes=60,max_total_stdin_bytes=100,
        max_total_retained_stdout_bytes=250,max_total_retained_stderr_bytes=150),'functional')
    with pytest.raises(DispatchBlocked,match='stdin_byte_limit'):
        ledger.reserve('too-large',61,200,100)
    ledger.reserve('first',60,200,100)
    ledger.activate('first')
    ledger.settle(190,90,usage())
    with pytest.raises(DispatchBlocked,match='total_stdin'):
        ledger.reserve('too-large-total',41,200,100)
    second=ledger.reserve('second',40,200,100)
    assert second['stdout_cap']==60 and second['stderr_cap']==60
    ledger.activate('second')
    ledger.settle(60,60,usage())
    assert ledger.stdin_bytes==100 and ledger.stdout_bytes==250 and ledger.stderr_bytes==150


def test_reported_usage_threshold_and_unknown_usage_fail_closed(tmp_path):
    ledger=ControlLedger(tmp_path/'controls.jsonl',ResourceLimits(max_observed_cli_tokens=120),'functional')
    ledger.reserve('first',30,200,100)
    ledger.activate('first')
    assert ledger.settle(100,20,usage()) is False
    with pytest.raises(DispatchBlocked,match='observed_cli_token_threshold'):
        ledger.reserve('second',30,200,100)
    unknown=ControlLedger(tmp_path/'unknown.jsonl',ResourceLimits(max_observed_cli_tokens=1000),'functional')
    unknown.reserve('first',30,200,100)
    unknown.activate('first')
    unknown.settle(10,10,capture_codex_usage(b''))
    assert unknown.usage_unknown
    with pytest.raises(DispatchBlocked,match='cli_usage_unknown'):
        unknown.reserve('second',30,200,100)


def test_audit_failure_prevents_activation_and_failed_settlement_cannot_reuse_slot(tmp_path):
    ledger=ControlLedger(tmp_path,ResourceLimits(),'functional')
    with pytest.raises(DispatchBlocked,match='control_audit_io_failure'):
        ledger.reserve('request',30,200,100)
    assert ledger.dispatches==0
    ledger.path=tmp_path/'controls.jsonl'
    ledger.reserve('request',30,200,100)
    ledger.activate('request')
    ledger.path=tmp_path
    with pytest.raises(DispatchBlocked,match='control_audit_io_failure'):
        ledger.settle(10,10,usage())
    with pytest.raises(DispatchBlocked,match='unsettled'):
        ledger.reserve('request',30,200,100)

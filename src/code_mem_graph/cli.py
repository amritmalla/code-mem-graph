from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import click
from pydantic import ValidationError
import yaml

from .evaluation.models import Suite, Trace
from .evaluation.report import PayloadCounter, build_report
from .evaluation.analysis import analyze_experiment
from .evaluation.experiment import check_partitions, load_experiment, preparation_summary, prepare_experiment, save_experiment
from .evaluation.models import AgentProfile
from .evaluation.models import CaptureInfo
from .evaluation.runner import ModelReply, RunnerLimits, TrialSession, TrialState, save_checkpoint
from .evaluation.trace import write_trace
from .evaluation.study_cli import study_group
from .knowledge.freshness import assess_map
from .knowledge.io import load_map
from .knowledge.models import KnowledgeMap
from .knowledge.build import build_repository, load_repository_map
from .knowledge.contributions import (
    add_contribution,
    load_contribution,
    render_contribution_template,
    retire_contribution,
    update_contribution,
)
from .knowledge.orient import OrientRequest, orientation
from .knowledge.context import ContextRequest, task_context
from .knowledge.workflow import record_review, refresh_repository, resolve_conflict, show_item, status_report
from .serialization import serialize
from .onboarding_cli import init_command, doctor_command, config_command


@click.group()
def main() -> None:
    """code-mem-graph — project knowledge and evidence-based context evaluation."""


main.add_command(study_group)
main.add_command(init_command)
main.add_command(doctor_command)
main.add_command(config_command)


@main.command("memory", context_settings={"ignore_unknown_options": True}, add_help_option=False)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def memory_command(ctx: click.Context, arguments: tuple[str, ...]):
    """Index, retrieve, and maintain durable memory; use memory --help for commands."""
    # Keep optional retrieval backends out of knowledge-only CLI startup.
    from .engine.cli import main as memory_main

    return memory_main.main(args=list(arguments), prog_name=ctx.command_path, standalone_mode=False)


@main.group("experiment")
def experiment_group() -> None:
    """Prepare frozen discovery experiments and analyze imported traces; no model calls."""


@main.group("review")
def review_group() -> None:
    """Prepare frozen answer evidence and attach recorded independent reviews; no model calls."""


@review_group.command("schema")
def review_schema():
    """Print the recorded semantic-review schema; no model calls."""
    from .evaluation.validation import SemanticReview
    click.echo(json.dumps(SemanticReview.model_json_schema(),indent=2))


def _review_inputs(experiment_path,registration_path):
    if (experiment_path is None)==(registration_path is None):
        raise ValueError("choose one frozen experiment or acceptance registration")
    if experiment_path:
        experiment=load_experiment(experiment_path)
        return experiment.suite,experiment.snapshot
    from .evaluation.acceptance import AcceptanceRegistration
    registration=AcceptanceRegistration.model_validate_json(registration_path.read_text(encoding="utf-8"))
    return registration.suite,registration.snapshot


@review_group.command("packet")
@click.option("--experiment", "experiment_path", type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--registration", "registration_path", type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--trace", "trace_path", required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--change-evidence",type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--output",required=True,type=click.Path(dir_okay=False,path_type=Path))
def review_packet(experiment_path,registration_path,trace_path,change_evidence,output):
    """Create numbered answer units, checked citations and source proof; withhold approach/cost/previous opinions."""
    from .evaluation.validation import make_audit,grade_audit,review_context_digest
    try:
        suite,snapshot=_review_inputs(experiment_path,registration_path)
        trace=Trace.model_validate_json(trace_path.read_text(encoding="utf-8"))
        change=json.loads(change_evidence.read_text(encoding="utf-8")) if change_evidence else None
        if change is None and trace.review.validation:
            change=trace.review.validation.get("change")
        audit=make_audit(suite,snapshot,trace,change=change)
        packet={"schema_version":1,"audit":audit.model_dump(mode="json"),"task":suite.task(trace.task_id).model_dump(mode="json"),
            "deterministic_checks":grade_audit(suite,trace,audit),
            "instructions":"Review every nonempty answer unit, including headings/code as nonclaims where appropriate. "
                "A supported factual unit must have valid citation IDs whose frozen content supports all its assertions. "
                "Check every criterion and acceptance-check ID; omitted critical guidance is not met. "
                "Do not invent responsibilities or approvers; scoped unknowns can be limitations. "
                "Return SemanticReview data bound to answer_digest, evidence_snapshot_digest and review_context_digest. "
                "Disclose reviewer identity, selection and independence; do not execute commands. "
                "Review does not establish that a code change works. Previous opinions/approach/cost are withheld."}
        # Proof snapshot digest is computed rather than a stored field.
        packet["evidence_snapshot_digest"]=audit.proof.snapshot_digest
        packet["review_context_digest"]=review_context_digest(audit)
        with output.open("x",encoding="utf-8") as stream:
            json.dump(packet,stream,indent=2,ensure_ascii=False)
        click.echo(json.dumps({"units":len(audit.units),"citations":len(audit.citations),"output":str(output),"model_calls":0}))
    except (OSError,ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@review_group.command("attach")
@click.option("--experiment", "experiment_path", type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--registration", "registration_path", type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--trace", "trace_path", required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--review", "reviews", multiple=True,required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--change-evidence",type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--output",required=True,type=click.Path(dir_okay=False,path_type=Path))
def review_attach(experiment_path,registration_path,trace_path,reviews,change_evidence,output):
    """Write a derived trace retaining all previous/new opinions and disagreements; never overwrite the original."""
    from .evaluation.validation import AnswerAudit,SemanticReview,make_audit,grade_audit,reviewed_trace
    try:
        suite,snapshot=_review_inputs(experiment_path,registration_path)
        trace=Trace.model_validate_json(trace_path.read_text(encoding="utf-8"))
        previous=AnswerAudit.model_validate(trace.review.validation) if trace.review.validation else None
        change=json.loads(change_evidence.read_text(encoding="utf-8")) if change_evidence else previous.change if previous else None
        opinions=[*(previous.reviews if previous else []),*(SemanticReview.model_validate_json(path.read_text(encoding="utf-8")) for path in reviews)]
        audit=make_audit(suite,snapshot,trace,reviews=opinions,change=change)
        summary=grade_audit(suite,trace,audit)
        write_trace(reviewed_trace(suite,trace,audit),output)
        click.echo(json.dumps(summary,indent=2))
        if summary['status']!='verified':
            raise click.exceptions.Exit(1)
    except (OSError,ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.group("acceptance")
def acceptance_group() -> None:
    """Register explicit checks and run trusted change fixtures in separate snapshot directories."""


@acceptance_group.command("register")
@click.option("--root",required=True,type=click.Path(exists=True,file_okay=False,path_type=Path))
@click.option("--suite", "suite_path",required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--task-id",required=True)
@click.option("--plan", "plan_path",required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--output",required=True,type=click.Path(file_okay=False,path_type=Path))
def acceptance_register(root,suite_path,task_id,plan_path,output):
    """Freeze source, trusted tests, allowed changes, executable hashes, argv and resource limits; execute nothing."""
    from .evaluation.acceptance import AcceptancePlan,register_acceptance
    try:
        suite=Suite.model_validate(yaml.safe_load(suite_path.read_text(encoding="utf-8")))
        plan=AcceptancePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        registration=register_acceptance(root,suite,task_id,plan,output)
        click.echo(json.dumps({"registration_digest":registration.digest,"snapshot":registration.snapshot.digest,"executed_checks":0}))
    except (OSError,ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@acceptance_group.command("run")
@click.option("--registration", "registration_path",required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--candidate", "candidate_path",required=True,type=click.Path(exists=True,dir_okay=False,path_type=Path))
@click.option("--run-id",required=True)
@click.option("--output",required=True,type=click.Path(file_okay=False,path_type=Path))
def acceptance_run(registration_path,candidate_path,run_id,output):
    """Run only registered argv against a source overlay; retain checkout/logs/receipt and unchanged trusted tests."""
    import asyncio
    from .evaluation.acceptance import AcceptanceRegistration,CandidatePatch,run_acceptance
    try:
        registration=AcceptanceRegistration.model_validate_json(registration_path.read_text(encoding="utf-8"))
        patch=CandidatePatch.model_validate_json(candidate_path.read_text(encoding="utf-8"))
        receipt=asyncio.run(run_acceptance(registration,patch,run_id,output))
        with (output/'evidence.json').open("x",encoding="utf-8") as stream:
            json.dump({"registration":registration.model_dump(mode="json"),"candidate":patch.model_dump(mode="json"),
                       "receipt":receipt.model_dump(mode="json")},stream,indent=2)
        click.echo(json.dumps({"outcome":receipt.outcome,"candidate_digest":patch.digest,"checks":len(receipt.checks),
                               "evidence":str(output/'evidence.json')},indent=2))
        if receipt.outcome!='passed':
            raise click.exceptions.Exit(1)
    except (OSError,ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.group("accounting")
def accounting_group() -> None:
    """Build separate usage/cost ledgers and reports without model calls."""


@accounting_group.command("from-traces")
@click.option("--trace", "trace_paths", multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--category", type=click.Choice(["task", "evaluation", "review", "build", "generation"]), required=True)
@click.option("--billing-mode", type=click.Choice(["api", "subscription", "unknown"]), default="unknown", show_default=True)
@click.option("--price-card", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def accounting_from_traces(trace_paths: tuple[Path, ...], category: str, billing_mode: str, price_card: Path | None, output: Path):
    """Import explicit traces into one category; unknown billing/prices remain unknown."""
    from .evaluation.accounting import PriceCard, ledger_from_traces
    try:
        traces = [Trace.model_validate_json(path.read_text(encoding="utf-8")) for path in trace_paths]
        price = PriceCard.model_validate_json(price_card.read_text(encoding="utf-8")) if price_card else None
        ledger = ledger_from_traces(traces, category=category, billing_mode=billing_mode, price=price)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(ledger.model_dump_json(indent=2) + "\n")
        click.echo(json.dumps({"records": len(ledger.records), "category": category, "output": str(output)}))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@accounting_group.command("merge")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def accounting_merge(paths: tuple[Path, ...], output: Path):
    """Merge category ledgers; reject duplicate costs and conflicting rates/currencies."""
    from .evaluation.accounting import CostLedger, merge_ledgers
    try:
        ledger = merge_ledgers([CostLedger.model_validate_json(path.read_text(encoding="utf-8")) for path in paths])
        with output.open("x", encoding="utf-8") as stream:
            stream.write(ledger.model_dump_json(indent=2) + "\n")
        click.echo(json.dumps({"records": len(ledger.records), "output": str(output)}))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@accounting_group.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
def accounting_report(path: Path, output: Path | None):
    """Report observed usage, missing coverage and versioned cost estimates by category."""
    from .evaluation.accounting import CostLedger, ledger_report
    try:
        ledger = CostLedger.model_validate_json(path.read_text(encoding="utf-8"))
        text = json.dumps(ledger_report(ledger), indent=2, ensure_ascii=False, allow_nan=False)
        if output is not None:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        click.echo(text)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.group("trial")
def trial_group() -> None:
    """Run controlled trials or checkpoint externally supplied replies."""


@trial_group.command("agent-profile")
@click.option("--transport-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
def trial_agent_profile(transport_config: Path, output: Path | None) -> None:
    """Probe the installed transport and describe its profile without model calls."""
    from .evaluation.codex_transport import CodexTransportConfig, describe_codex
    try:
        config = CodexTransportConfig.model_validate_json(transport_config.read_text(encoding="utf-8"))
        descriptor = describe_codex(config)
        text = descriptor.profile.model_dump_json(indent=2)
        if output is not None:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        click.echo(text)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@trial_group.command("capabilities")
@click.option("--transport-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def trial_capabilities(transport_config: Path) -> None:
    """Inspect requested controls and unsupported guarantees without model calls."""
    from .evaluation.codex_transport import CodexTransportConfig, DISABLED_FEATURES, describe_codex
    from .evaluation.controls import codex_capabilities
    try:
        config = CodexTransportConfig.model_validate_json(transport_config.read_text(encoding="utf-8"))
        descriptor = describe_codex(config)
        click.echo(json.dumps({"client_version": descriptor.version, "model": config.model,
                              "execution_mode": config.execution_mode, "resource_limits": config.resource_limits.model_dump(),
                              "requested_disabled_features": list(DISABLED_FEATURES), "web_search": "disabled",
                              "capabilities": codex_capabilities(), "capture": descriptor.capture.model_dump()}, indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@trial_group.command("run")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--run-id", required=True)
@click.option("--limits", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--transport-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="New directory for checkpoints, transport artifacts, and the terminal trace; refuses reuse.")
@click.option("--registration", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Shared frozen cohort registration. Omission is reported as an efficiency-blocking deviation.")
def trial_run(experiment_path: Path, run_id: str, limits: Path, transport_config: Path, output: Path,
              registration: Path | None) -> None:
    """Invoke the configured model and mediate frozen tools until a terminal outcome."""
    import asyncio
    from .evaluation.codex_transport import CodexTransportConfig, describe_codex
    from .evaluation.coordination import execute_coordinated, make_registration
    try:
        config = CodexTransportConfig.model_validate_json(transport_config.read_text(encoding="utf-8"))
        descriptor = describe_codex(config)
        experiment = load_experiment(experiment_path)
        declared_limits = RunnerLimits.model_validate_json(limits.read_text(encoding="utf-8"))
        frozen = make_registration(experiment, declared_limits, config, descriptor)
        TrialSession.start(experiment, run_id, declared_limits, descriptor.capture)
        trace = asyncio.run(execute_coordinated(output, registration=frozen, run_id=run_id, cohort=registration,
                                               descriptor=descriptor))
        _direct_trial_summary(trace, output)
        if trace.outcome != "completed":
            raise click.exceptions.Exit(1)
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise click.ClickException(str(exc)) from exc


def _direct_trial_summary(trace, output):
    click.echo(json.dumps({"run_id": trace.run_id, "outcome": trace.outcome,
                          "reason": trace.termination_reason, "trace": str((output / "trace.json").resolve()),
                          "model_dispatches": sum(event.kind == "model" for event in trace.events),
                          "tool_calls": sum(event.kind == "tool" for event in trace.events),
                          "coordination": trace.coordination}, indent=2))


@trial_group.command("register")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--limits", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--transport-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True)
def trial_register(experiment_path, limits, transport_config, output):
    """Freeze shared controls and no-retry rules before running any cohort arm; no model calls."""
    from .evaluation.codex_transport import CodexTransportConfig, describe_codex
    from .evaluation.coordination import make_registration, register_cohort
    try:
        config = CodexTransportConfig.model_validate_json(transport_config.read_text(encoding="utf-8"))
        descriptor = describe_codex(config)
        registration = make_registration(load_experiment(experiment_path),
            RunnerLimits.model_validate_json(limits.read_text(encoding="utf-8")), config, descriptor)
        register_cohort(output, registration)
        click.echo(json.dumps({"registration_digest":registration.digest,"scheduled_runs":len(registration.experiment.schedule()),
                               "output":str(output.resolve()),"retry_policy":registration.retry_policy},indent=2))
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise click.ClickException(str(exc)) from exc


@trial_group.command("resume")
@click.argument("output", type=click.Path(exists=True, file_okay=False, path_type=Path))
def trial_resume(output):
    """Resume the original durable run; uncertain deliveries are retained and never resent."""
    import asyncio
    from .evaluation.coordination import execute_coordinated
    try:
        trace = asyncio.run(execute_coordinated(output, resume=True))
        _direct_trial_summary(trace,output)
        if trace.outcome != "completed":
            raise click.exceptions.Exit(1)
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise click.ClickException(str(exc)) from exc


def _trial_summary(session: TrialSession) -> dict:
    return {"run_id": session.state.run_id, "dispatch_id": session.state.dispatch_id,
            "outcome": session.state.outcome, "termination_reason": session.state.termination_reason,
            "logical_model_turns": sum(event["kind"] == "model" for event in session.state.events),
            "native_tool_calls": sum(event["kind"] == "tool" for event in session.state.events),
            "visible_packet_tokens_spent": session.state.visible_tokens_spent,
            "provider_usage_available": session.state.capture.provider_usage_available}


@trial_group.command("start")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--run-id", required=True)
@click.option("--limits", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--capture", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def trial_start(experiment_path: Path, run_id: str, limits: Path, capture: Path, output: Path) -> None:
    """Prepare the first bounded request without invoking a model."""
    try:
        session = TrialSession.start(load_experiment(experiment_path), run_id,
            RunnerLimits.model_validate_json(limits.read_text(encoding="utf-8")),
            CaptureInfo.model_validate_json(capture.read_text(encoding="utf-8")))
        session.next_request()
        save_checkpoint(session.state, output)
        click.echo(json.dumps(_trial_summary(session), indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@trial_group.command("request")
@click.argument("state_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def trial_request(state_path: Path) -> None:
    """Print the pending logical request; dispatch and host measurement are external."""
    try:
        state = TrialState.model_validate_json(state_path.read_text(encoding="utf-8"))
        if state.outcome is not None or state.pending_request is None:
            raise ValueError("checkpoint has no pending request")
        click.echo(json.dumps({"dispatch_id": state.dispatch_id, "request": state.pending_request}, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@trial_group.command("step")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--state", "state_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--dispatch-id", required=True)
@click.option("--reply", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--usage", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Actual provider usage only; omit when unavailable. Never use model-generated estimates.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--trace", "trace_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
def trial_step(experiment_path: Path, state_path: Path, dispatch_id: str, reply: Path,
               usage: Path | None, output: Path, trace_path: Path | None) -> None:
    """Record an actual reply, mediate frozen tools, and emit the next immutable checkpoint."""
    try:
        from .evaluation.models import Usage
        session = TrialSession(load_experiment(experiment_path), TrialState.model_validate_json(state_path.read_text(encoding="utf-8")))
        observed_usage = Usage.model_validate_json(usage.read_text(encoding="utf-8")) if usage else None
        if session.state.dispatch_id != dispatch_id:
            raise ValueError("response has no matching pending dispatch")
        try:
            content = reply.read_text(encoding="utf-8-sig")
        except UnicodeError:
            session.fail_pending("invalid_utf8_transport_reply")
        else:
            session.accept(ModelReply(content=content, usage=observed_usage), dispatch_id=dispatch_id)
        session.next_request()
        save_checkpoint(session.state, output)
        if trace_path is not None and session.state.outcome is not None:
            write_trace(session.trace(), trace_path)
        click.echo(json.dumps(_trial_summary(session), indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@trial_group.command("abort")
@click.argument("experiment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--state", "state_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--reason", required=True)
@click.option("--dispatched", is_flag=True, help="A model request was actually sent but its response is unavailable.")
@click.option("--trace", "trace_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def trial_abort(experiment_path: Path, state_path: Path, reason: str, dispatched: bool, trace_path: Path) -> None:
    """Retain interrupted trials; missing provider usage/response remains unknown."""
    try:
        session = TrialSession(load_experiment(experiment_path), TrialState.model_validate_json(state_path.read_text(encoding="utf-8")))
        if dispatched:
            session.fail_pending(reason, outcome="aborted")
        else:
            session.abort(reason)
        write_trace(session.trace(), trace_path)
        click.echo(json.dumps(_trial_summary(session), indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@experiment_group.command("prepare")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--suite", "suite_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--task-id", required=True)
@click.option("--mode", type=click.Choice(["orient", "context"]), required=True)
@click.option("--encoding", required=True)
@click.option("--ceiling", type=click.IntRange(min=1), required=True, help="Explicit experimental payload ceiling; not a default.")
@click.option("--split", type=click.Choice(["calibration", "validation", "heldout"]), required=True)
@click.option("--replicates", type=click.IntRange(min=1), required=True)
@click.option("--cache-state", type=click.Choice(["cold", "warm"]), required=True)
@click.option("--agent-profile", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--representation", "representations", multiple=True, type=click.Choice(["detailed", "compact"]),
              help="Include measured representation alternatives; defaults to detailed only.")
@click.option("--detail", "details", multiple=True, type=click.Choice(["full", "focused"]),
              help="Include measured content views; defaults to full only.")
def experiment_prepare(root: Path, suite_path: Path, task_id: str, mode: str, encoding: str, ceiling: int,
                       split: str, replicates: int, cache_state: str, agent_profile: Path | None, output: Path,
                       representations: tuple, details: tuple) -> None:
    """Freeze source evidence and measured bundle-boundary alternatives without launching agents."""
    try:
        suite = Suite.model_validate(yaml.safe_load(suite_path.read_text(encoding="utf-8")))
        profile = AgentProfile.model_validate_json(agent_profile.read_text(encoding="utf-8")) if agent_profile else None
        private_paths = [str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
                         for path in (suite_path, agent_profile) if path is not None and path.resolve().is_relative_to(root.resolve())]
        result = prepare_experiment(root, suite, task_id=task_id, mode=mode, encoding=encoding, ceiling=ceiling,
                                    split=split, replicates=replicates, cache_state=cache_state, agent=profile,
                                    private_paths=private_paths, representations=representations or ("detailed",),
                                    details=details or ("full",))
        save_experiment(result, output)
        click.echo(json.dumps(preparation_summary(result), ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@experiment_group.command("inspect")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def experiment_inspect(path: Path) -> None:
    """Verify artifact hashes, measurements, and schedule; print a compact readiness summary."""
    try:
        result = load_experiment(path)
        summary = preparation_summary(result)
        summary["schedule"] = result.schedule()
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@experiment_group.command("packet")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--run-id", required=True)
def experiment_packet(path: Path, run_id: str) -> None:
    """Print the task/knowledge packet for a scheduled run, without the private review rubric."""
    try:
        click.echo(json.dumps(load_experiment(path).agent_packet(run_id), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@experiment_group.command("assess")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--traces", type=click.Path(exists=True, path_type=Path), required=True)
def experiment_assess(path: Path, traces: Path) -> None:
    """Analyze every planned trial, retaining missing/unreviewed/failed cases. Never select a default."""
    try:
        paths = sorted(traces.glob("*.json")) if traces.is_dir() else [traces]
        runs = [Trace.model_validate_json(entry.read_text(encoding="utf-8")) for entry in paths]
        report = analyze_experiment(load_experiment(path), runs)
        click.echo(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@experiment_group.command("partitions")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def experiment_partitions(paths: tuple[Path, ...]) -> None:
    """Check task/repository/source overlap between declared experiment splits."""
    try:
        click.echo(json.dumps(check_partitions([load_experiment(path) for path in paths]), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.group()
def knowledge() -> None:
    """Inspect and validate versionable project knowledge."""


@knowledge.command("build")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", show_default=True)
@click.option("--repository-id", default=None, help="Stable identity, required for the first build.")
def knowledge_build(root: Path, repository_id: str | None) -> None:
    """Extract deterministic facts into .project-knowledge/map.yaml and README.md."""
    try:
        document = build_repository(root, repository_id=repository_id)
        click.echo(f"Built knowledge for {document.repository_id}: {len(document.generated.items)} generated claims, "
                   f"{len(document.curated.items)} maintainer claims. Files: .project-knowledge/map.yaml and README.md")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("orient")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", show_default=True)
@click.option("--budget", type=click.IntRange(min=1), required=True, help="Explicit prototype token cap; no automatic default.")
@click.option("--encoding", required=True, help="Explicit tiktoken encoding for payload estimates.")
@click.option("--representation", type=click.Choice(["detailed", "compact"]), default="detailed", show_default=True)
@click.option("--detail", type=click.Choice(["full", "focused"]), default="full", show_default=True)
def orient_command(root: Path, budget: int, encoding: str, representation: str, detail: str) -> None:
    """Return cited project orientation with coverage and source status."""
    try:
        click.echo(serialize(orientation(root, OrientRequest(budget=budget, encoding=encoding,
                                                           representation=representation, detail=detail))), nl=False)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("context")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", show_default=True)
@click.option("--task", required=True)
@click.option("--budget", type=click.IntRange(min=1), required=True)
@click.option("--encoding", required=True)
@click.option("--path", "paths", multiple=True, help="Repository-relative target file or directory; may be a proposed new path.")
@click.option("--component", "components", multiple=True, help="Known component ID.")
@click.option("--application", help="Eligible manifest directory relative to the repository, or .; path hints alone do not filter siblings.")
@click.option("--item", "items", multiple=True, help="Known knowledge ID for focused inspection.")
@click.option("--representation", type=click.Choice(["detailed", "compact"]), default="detailed", show_default=True)
@click.option("--detail", type=click.Choice(["full", "focused"]), default="full", show_default=True)
def context_command(root: Path, task: str, budget: int, encoding: str, paths: tuple, components: tuple, items: tuple,
                    representation: str, detail: str, application: str | None) -> None:
    """Return bounded task context, dependencies, practices, checks, and gaps."""
    try:
        request = ContextRequest(task=task, budget=budget, encoding=encoding, paths=list(paths),
                                 component_ids=list(components), item_ids=list(items), representation=representation, detail=detail,
                                 application=application)
        click.echo(serialize(task_context(root, request)), nl=False)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("status")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
@click.option("--check", is_flag=True, help="Exit unsuccessfully for stale/unknown active evidence or a pending review queue.")
def knowledge_status(root: Path, check: bool) -> None:
    """Inspect current evidence, dependent claims, conflicts, and review requirements."""
    try:
        report = status_report(root)
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        if check and (report["source_status"]["stale"] or report["source_status"]["unknown"] or report["review_queue"]
                      or report["refresh"]["status"] == "failed"):
            raise click.exceptions.Exit(1)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("refresh")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
def knowledge_refresh(root: Path) -> None:
    """Rebuild generated facts, preserve curated knowledge, and report changed/retired claims."""
    try:
        click.echo(json.dumps(refresh_repository(root), ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("show")
@click.argument("item_id")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
def knowledge_show(root: Path, item_id: str) -> None:
    """Inspect a full item, its version, current source fingerprints, and dependents."""
    try:
        click.echo(json.dumps(show_item(root, item_id), ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("review")
@click.argument("item_id")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
@click.option("--expected-digest", required=True)
@click.option("--reviewer", required=True)
@click.option("--status", type=click.Choice(["accepted", "rejected"]), required=True)
@click.option("--note", default=None)
@click.option("--refresh-evidence", is_flag=True, help="Explicitly recapture curated evidence after reviewing the cited changes.")
def knowledge_review(root: Path, item_id: str, expected_digest: str, reviewer: str,
                     status: str, note: str | None, refresh_evidence: bool) -> None:
    """Record a named, version-bound review. No automatic approvals."""
    try:
        record_review(root, item_id, expected_digest=expected_digest, reviewer=reviewer,
                      status=status, note=note, refresh_evidence=refresh_evidence)
        click.echo(json.dumps(show_item(root, item_id), ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("resolve-conflict")
@click.argument("conflict_id")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
@click.option("--expected-map-digest", required=True)
@click.option("--reviewer", required=True)
@click.option("--explanation", required=True)
def knowledge_resolve_conflict(root: Path, conflict_id: str, expected_map_digest: str,
                              reviewer: str, explanation: str) -> None:
    """Record a human conflict explanation against the inspected map and claim versions."""
    try:
        document = resolve_conflict(root, conflict_id, expected_map_digest=expected_map_digest,
                                    reviewer=reviewer, explanation=explanation)
        click.echo(json.dumps(status_report(root, document), ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("template")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".",
              help="Repository root when exporting an existing curated item with --from-item.")
@click.option("--from-item", default=None,
              help="Export an existing active curated item into editable contribution YAML/JSON.")
@click.option("--format", "output_format", type=click.Choice(["yaml", "json"]), default="yaml", show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write the template to a new file; refuses to overwrite an existing file.")
@click.option("--id", "item_id", default=None, help="Starter item ID for a new contribution template.")
@click.option("--topic", type=click.Choice(["overview", "architecture", "domain", "conventions", "workflow", "decision"]),
              default=None, help="Starter topic for a new contribution template.")
@click.option("--author", default=None, help="Starter author name for a new contribution template.")
@click.option("--basis", type=click.Choice(["documented", "inferred", "maintainer"]), default=None,
              help="Starter epistemic basis for a new contribution template.")
@click.option("--gap", "gap_ids", multiple=True, help="Existing gap ID to include in addresses_gap_ids.")
def knowledge_template(root: Path, from_item: str | None, output_format: str, output: Path | None,
                       item_id: str | None, topic: str | None, author: str | None, basis: str | None,
                       gap_ids: tuple[str, ...]) -> None:
    """Print a starter maintainer contribution file or export an existing curated item for editing."""
    try:
        if from_item is not None:
            if any(value is not None for value in (item_id, topic, author, basis)) or gap_ids:
                raise ValueError("--from-item cannot be combined with starter template overrides (--id/--topic/--author/--basis/--gap)")
            document = load_repository_map(root)
            item = next((entry for entry in document.items if entry.id == from_item), None)
            if item is None:
                raise ValueError(f"unknown knowledge item ID: {from_item}")
            rendered = render_contribution_template(output_format=output_format, item=item)
        else:
            rendered = render_contribution_template(
                output_format=output_format,
                item_id=item_id,
                topic=topic,
                author=author,
                basis=basis,
                addresses_gap_ids=list(gap_ids),
            )
        if output is not None:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
            click.echo(json.dumps({"output": str(output), "format": output_format}, ensure_ascii=False))
        else:
            click.echo(rendered, nl=False)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("add")
@click.argument("contribution_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", show_default=True)
@click.option("--expected-map-digest", default=None,
              help="Optional optimistic guard matching the inspected knowledge map digest.")
@click.option("--dry-run", is_flag=True,
              help="Validate, capture evidence/dependencies, and preview the resulting claim without writing files.")
def knowledge_add(contribution_path: Path, root: Path, expected_map_digest: str | None, dry_run: bool) -> None:
    """Add a new unreviewed maintainer contribution from a YAML/JSON file."""
    try:
        contribution = load_contribution(contribution_path)
        result = add_contribution(root, contribution, expected_map_digest=expected_map_digest, dry_run=dry_run)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("update")
@click.argument("contribution_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", show_default=True)
@click.option("--expected-digest", required=True,
              help="Current content_digest of the curated item being updated (from `knowledge show`).")
@click.option("--expected-map-digest", default=None,
              help="Optional optimistic guard matching the inspected knowledge map digest.")
@click.option("--dry-run", is_flag=True,
              help="Validate, capture evidence/dependencies, and preview the revised claim without writing files.")
def knowledge_update(contribution_path: Path, root: Path, expected_digest: str,
                     expected_map_digest: str | None, dry_run: bool) -> None:
    """Update an existing curated contribution with version protection; invalidates prior acceptance."""
    try:
        contribution = load_contribution(contribution_path)
        result = update_contribution(
            root,
            contribution,
            expected_digest=expected_digest,
            expected_map_digest=expected_map_digest,
            dry_run=dry_run,
        )
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@knowledge.command("retire")
@click.argument("item_id")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".", show_default=True)
@click.option("--expected-digest", required=True,
              help="Current content_digest of the curated item being retired (from `knowledge show`).")
@click.option("--expected-map-digest", default=None,
              help="Optional optimistic guard matching the inspected knowledge map digest.")
def knowledge_retire(item_id: str, root: Path, expected_digest: str, expected_map_digest: str | None) -> None:
    """Retire an active curated contribution while keeping its historical record and review."""
    try:
        result = retire_contribution(
            root,
            item_id,
            expected_digest=expected_digest,
            expected_map_digest=expected_map_digest,
        )
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("serve")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
def serve(root: Path) -> None:
    """Serve read-only knowledge tools over MCP stdio; root is explicitly pinned."""
    import asyncio
    from .mcp import serve_stdio
    asyncio.run(serve_stdio(root))


@knowledge.command("schema")
def knowledge_schema() -> None:
    """Print the versioned knowledge-map JSON Schema."""
    click.echo(json.dumps(KnowledgeMap.model_json_schema(), ensure_ascii=False, indent=2))


@knowledge.command("validate")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Also assess current source fingerprints in this repository directory.")
def knowledge_validate(path: Path, root: Path | None) -> None:
    """Validate a knowledge map; optionally report source freshness."""
    try:
        document = load_map(path)
        result = {
            "schema_version": document.schema_version, "repository_id": document.repository_id,
            "valid": True, "generated_items": len(document.generated.items),
            "curated_items": len(document.curated.items), "refresh": document.refresh.model_dump(),
            "gaps": [gap.model_dump() for layer in (document.generated, document.curated) for gap in layer.gaps],
            "conflicts": [conflict.model_dump()
                          for layer in (document.generated, document.curated) for conflict in layer.conflicts],
            "assessments": [assessment.model_dump() for assessment in assess_map(document, root)]
                           if root is not None else None,
        }
        click.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--traces", type=click.Path(exists=True, path_type=Path), required=True,
              help="One run JSON file or a directory containing run JSON files.")
@click.option("--encoding", required=True, help="Explicit tiktoken encoding used for labeled estimates.")
@click.option("--baseline", required=True, help="Name of the native/reference policy to pair against.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
def evaluate(suite_path: Path, traces: Path, encoding: str, baseline: str, output: Path | None) -> None:
    """Validate recorded runs and report quality, usage, and comparable pairs.

    Does not invoke an agent, model, or any commands from the suite.
    """
    try:
        suite = Suite.model_validate(yaml.safe_load(suite_path.read_text(encoding="utf-8")))
        paths = sorted(traces.glob("*.json")) if traces.is_dir() else [traces]
        if not paths:
            raise ValueError("no trace JSON files found")
        runs = [Trace.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]
        report = build_report(suite, runs, PayloadCounter(encoding), baseline=baseline)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
            click.echo(f"Report: {output}")
        else:
            click.echo(rendered, nl=False)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()

"""Durable, single-owner Codex trials. Ambiguous deliveries are never retried."""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Literal

from pydantic import Field

from code_mem_graph.serialization import fingerprint, serialize

from .accounting import capture_codex_usage, ledger_from_traces, ledger_report, usage_diagnostics
from .codex_transport import (
    PROMPT, VERSION, CodexDescriptor, CodexExecTransport, CodexTransportConfig, _parse_events,
    describe_codex, response_schema,
)
from .controls import DispatchBlocked
from .experiment import Experiment
from .models import CaptureInfo, Contract, TransportProvenance
from .runner import ModelReply, RunnerLimits, SYSTEM, TOOLS, TransportFailure, TrialSession, TrialState, run_trial

COORDINATOR_VERSION = "1"
RETRY_POLICY = {"adapter_attempts_per_dispatch": 1, "fallback": "none", "retry": "none",
                "ambiguous_delivery": "retain evidence and abort; never resend"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Registration(Contract):
    schema_version: Literal[1] = 1
    coordinator_version: str = COORDINATOR_VERSION
    experiment: Experiment
    limits: RunnerLimits
    config: CodexTransportConfig
    capture: CaptureInfo
    executable: str
    executable_sha256: str
    system: str = SYSTEM
    tools: list[dict] = Field(default_factory=lambda: json.loads(serialize(TOOLS)))
    transport_prompt: str = PROMPT
    output_schema: dict = Field(default_factory=response_schema)
    transport_version: str = VERSION
    retry_policy: dict = Field(default_factory=lambda: RETRY_POLICY.copy())
    notes: list[str] = Field(default_factory=lambda: [
        "Backend model revision and provider cache state are unverified; requested alias is not a pinned revision.",
        "Provider latency is unexposed; local client timings include startup, networking and execution."])

    @property
    def digest(self):
        return fingerprint(self.model_dump(mode="json"))

    def verify_runtime(self, descriptor: CodexDescriptor):
        if (self.coordinator_version != COORDINATOR_VERSION or self.transport_version != VERSION
                or self.system != SYSTEM or self.tools != TOOLS or self.transport_prompt != PROMPT
                or self.output_schema != response_schema() or self.retry_policy != RETRY_POLICY):
            raise ValueError("registered instructions/implementation/retry policy differ; no model request sent")
        if self.experiment.agent != descriptor.profile or self.capture != descriptor.capture:
            raise ValueError("transport profile/capture differs from registration; no model request sent")
        if self.executable != descriptor.executable or self.executable_sha256 != sha256(Path(descriptor.executable)):
            raise ValueError("registered client executable changed; no model request sent")


def make_registration(experiment, limits, config, descriptor):
    if experiment.agent != descriptor.profile:
        raise ValueError("transport profile differs from the prepared experiment; prepare with trial agent-profile")
    for entry in experiment.schedule():
        TrialSession.start(experiment, entry["run_id"], limits, descriptor.capture)
    return Registration(experiment=experiment, limits=limits, config=config, capture=descriptor.capture,
                        executable=descriptor.executable, executable_sha256=sha256(Path(descriptor.executable)))


def _connection(path):
    connection = sqlite3.connect(path, timeout=0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def register_cohort(path: Path, registration: Registration):
    """Freeze one registration shared by every scheduled arm; no model calls."""
    path.mkdir(parents=True, exist_ok=False)
    _publish(path / "registration.json", registration.model_dump_json(indent=2) + "\n")
    with closing(_connection(path / "registry.sqlite3")) as db, db:
        db.execute("CREATE TABLE registration (digest TEXT NOT NULL)")
        db.execute("INSERT INTO registration VALUES (?)", (registration.digest,))
        db.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, output TEXT NOT NULL UNIQUE)")


def load_registration(path: Path) -> Registration:
    registration = Registration.model_validate_json((path / "registration.json").read_text(encoding="utf-8"))
    with closing(_connection(path / "registry.sqlite3")) as db, db:
        row = db.execute("SELECT digest FROM registration").fetchone()
    if row is None or row[0] != registration.digest:
        raise ValueError("cohort registration digest mismatch")
    return registration


def _claim_run(path: Path, registration: Registration, run_id: str, output: Path, *, resume: bool):
    with closing(_connection(path / "registry.sqlite3")) as db, db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT digest FROM registration").fetchone()[0] != registration.digest:
            raise ValueError("cohort registration changed")
        row = db.execute("SELECT output FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is not None:
            if not resume or row[0] != str(output.resolve()):
                raise ValueError("scheduled run already claimed; resume its original directory")
        elif resume:
            raise ValueError("scheduled run has no registered claim")
        else:
            db.execute("INSERT INTO runs VALUES (?,?)", (run_id, str(output.resolve())))


@contextmanager
def run_lock(path: Path):
    """OS-released lock, held for the whole run; no stale PID/lease guessing."""
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("another coordinator owns this run") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _publish(path: Path, content: str):
    """Recoverable exports; the SQLite journal remains the source of truth."""
    data = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different export: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RunJournal:
    def __init__(self, path: Path, *, create=False):
        self.path = path
        self.db = _connection(path / "journal.sqlite3")
        if create:
            with self.db:
                self.db.execute("CREATE TABLE manifest (json TEXT NOT NULL, digest TEXT NOT NULL)")
                self.db.execute("CREATE TABLE states (seq INTEGER PRIMARY KEY, json TEXT NOT NULL, digest TEXT NOT NULL)")
                self.db.execute("""CREATE TABLE dispatches (id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE,
                    request TEXT NOT NULL, max_output INTEGER NOT NULL, claimed_at REAL NOT NULL,
                    result TEXT, result_digest TEXT, controls TEXT, elapsed_ms REAL, recovered INTEGER NOT NULL DEFAULT 0)""")
                self.db.execute("CREATE TABLE terminal (json TEXT NOT NULL, digest TEXT NOT NULL)")

    def close(self):
        self.db.close()

    def save_manifest(self, manifest):
        with self.db:
            self.db.execute("INSERT INTO manifest VALUES (?,?)", (serialize(manifest), fingerprint(manifest)))

    def manifest(self):
        row = self.db.execute("SELECT * FROM manifest").fetchone()
        value = json.loads(row["json"])
        if fingerprint(value) != row["digest"]:
            raise ValueError("run manifest checksum mismatch")
        return value

    def checkpoint(self, state):
        content = state.model_dump(mode="json")
        previous = self.db.execute("SELECT digest FROM states ORDER BY seq DESC LIMIT 1").fetchone()
        if previous is not None and previous[0] == fingerprint(content):
            return
        with self.db:
            cursor = self.db.execute("INSERT INTO states(json,digest) VALUES (?,?)", (serialize(content),fingerprint(content)))
        _publish(self.path / "checkpoints" / f"state-{cursor.lastrowid - 1:04d}.json", state.model_dump_json(indent=2) + "\n")

    def state(self):
        row = self.db.execute("SELECT * FROM states ORDER BY seq DESC LIMIT 1").fetchone()
        if row is None or fingerprint(json.loads(row["json"])) != row["digest"]:
            raise ValueError("missing or damaged journal checkpoint")
        return TrialState.model_validate_json(row["json"])

    def dispatch(self, dispatch_id):
        return self.db.execute("SELECT * FROM dispatches WHERE id=?", (dispatch_id,)).fetchone()

    def rows(self):
        return self.db.execute("SELECT * FROM dispatches ORDER BY ordinal").fetchall()

    def claim(self, dispatch_id, request, max_output):
        with self.db:
            self.db.execute("INSERT INTO dispatches(id,ordinal,request,max_output,claimed_at) VALUES (?,?,?,?,?)",
                (dispatch_id, len(self.rows()), serialize(request), max_output, time.time()))

    def delivered(self, dispatch_id, result, controls, elapsed_ms, *, recovered=False):
        result = {**result, "control_snapshot": controls}
        with self.db:
            self.db.execute("UPDATE dispatches SET result=?,result_digest=?,controls=?,elapsed_ms=?,recovered=? WHERE id=? AND result IS NULL",
                (serialize(result),fingerprint(result),serialize(controls),elapsed_ms,int(recovered),dispatch_id))

    def result(self, row):
        result = json.loads(row["result"])
        if fingerprint(result) != row["result_digest"]:
            raise ValueError("dispatch result checksum mismatch; refusing replay")
        if result["control_snapshot"] != json.loads(row["controls"]):
            raise ValueError("control snapshot checksum mismatch; refusing replay")
        return result

    def mark_replayed(self, dispatch_id):
        with self.db:
            self.db.execute("UPDATE dispatches SET recovered=1 WHERE id=?", (dispatch_id,))


def _envelope(value, reason=None):
    return {"failure": reason, "reply": value.model_dump(mode="json")}


def _unpack(value):
    if "control_refusal" in value:
        raise DispatchBlocked(value["failure"], value["control_refusal"], outcome=value["control_outcome"])
    reply = ModelReply.model_validate(value["reply"])
    if value["failure"] is not None:
        raise TransportFailure(value["failure"], transport=reply.transport, response=reply.content or None,
                               usage=reply.usage, usage_capture=reply.usage_capture)
    return reply


def recover_attempt(adapter: CodexExecTransport, dispatch_id: str, request: dict):
    """Use a validated receipt, else retain observed bytes as an uncertain failure."""
    attempt = adapter.artifact_dir / dispatch_id
    stdout = (attempt / "stdout.jsonl").read_bytes() if (attempt / "stdout.jsonl").is_file() else b""
    stderr = (attempt / "stderr.log").read_bytes() if (attempt / "stderr.log").is_file() else b""
    response, item_id, _, parsed_failure = _parse_events(stdout)
    reason = "dispatch_delivery_uncertain"
    record = None
    try:
        candidate = TransportProvenance.model_validate_json((attempt / "delivery.json").read_text(encoding="utf-8"))
        valid = (candidate.name == "codex-exec-jsonl" and candidate.version == VERSION
                and candidate.attempt_id == dispatch_id and candidate.request_digest == fingerprint(request)
                and candidate.requested_model == adapter.config.model and candidate.client_version == adapter.descriptor.version
                and Path(candidate.artifact_path).resolve() == attempt.resolve()
                and candidate.stdout_sha256 == hashlib.sha256(stdout).hexdigest()
                and candidate.stderr_sha256 == hashlib.sha256(stderr).hexdigest()
                and candidate.stdout_bytes == len(stdout) and candidate.stderr_bytes == len(stderr)
                and candidate.response_item_id == item_id
                and (attempt / "stdin.txt").read_bytes() == (PROMPT + serialize(request)).encode("utf-8")
                and json.loads((attempt / "response-schema.json").read_text(encoding="utf-8")) == response_schema())
        if not valid or not (candidate.failure_reason is not None or (candidate.exit_code == 0 and parsed_failure is None and response is not None)):
            raise ValueError("invalid recovery receipt")
        record, reason = candidate, candidate.failure_reason
    except (OSError, ValueError):
        pass
    usage_capture = capture_codex_usage(stdout, truncated=record.output_truncated if record else True)
    if record is None:
        record = TransportProvenance(name="codex-exec-jsonl", version=VERSION, attempt_id=dispatch_id,
            request_digest=fingerprint(request), requested_model=adapter.config.model, client_version=adapter.descriptor.version,
            response_source="recovered_raw_stdout" if response else "none", response_item_id=item_id,
            artifact_path=str(attempt), stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr).hexdigest(), stdout_bytes=len(stdout), stderr_bytes=len(stderr),
            output_truncated=True, exit_code=None, failure_reason=reason)
    reply = ModelReply(content=response or "", transport=record, usage=usage_capture.observed_usage(), usage_capture=usage_capture)
    return _envelope(reply, reason)


class JournaledTransport:
    def __init__(self, journal: RunJournal, adapter: CodexExecTransport):
        self.journal, self.adapter = journal, adapter
        self.dispatch_id = None
        self.last_failure = None
        # Rebuild local quotas from durable delivery snapshots, never reset them.
        rows = journal.rows()
        settled = [row for row in rows if row["controls"] is not None]
        if settled:
            state = journal.result(settled[-1])["control_snapshot"]
            for attribute, key in [("dispatches","dispatches"),("stdin_bytes","reserved_stdin_bytes"),
                ("stdout_bytes","retained_stdout_bytes"),("stderr_bytes","retained_stderr_bytes"),
                ("cli_tokens","observed_cli_tokens"),("usage_unknown","cli_usage_unknown")]:
                setattr(adapter.controls, attribute, state[key])

    def prepare_dispatch(self, dispatch_id, request, *, max_output_tokens):
        self.dispatch_id = dispatch_id
        row = self.journal.dispatch(dispatch_id)
        if row is not None:
            self.journal.mark_replayed(self.dispatch_id)
            if row["request"] != serialize(request) or row["max_output"] != max_output_tokens:
                raise ValueError("dispatch ID reused with different instructions/settings")
            return
        self.adapter.prepare(request, max_output_tokens=max_output_tokens)

    def can_replay(self):
        return self.journal.dispatch(self.dispatch_id) is not None

    async def complete(self, request, *, max_output_tokens):
        row = self.journal.dispatch(self.dispatch_id)
        if row is not None:
            if row["result"] is None:
                result = recover_attempt(self.adapter, self.dispatch_id, request)
                # A claimed-but-unsettled call consumes a slot/input reservation
                # conservatively, even if launch cannot be established.
                controls = self.adapter.controls
                controls.dispatches += 1
                controls.stdin_bytes += len((PROMPT + serialize(request)).encode("utf-8"))
                reply = ModelReply.model_validate(result["reply"])
                controls.stdout_bytes += reply.transport.stdout_bytes
                controls.stderr_bytes += reply.transport.stderr_bytes
                diagnostic = usage_diagnostics(reply.usage, reply.usage_capture)
                if diagnostic["recorded_scope_complete"]:
                    controls.cli_tokens += reply.usage.input_tokens + reply.usage.output_tokens
                else:
                    controls.usage_unknown = True
                ceiling = controls.limits.max_observed_cli_tokens
                if ceiling is not None and controls.cli_tokens > ceiling:
                    result["failure"] = result["failure"] or "observed_cli_token_threshold_exceeded"
                self.journal.delivered(self.dispatch_id, result, controls.snapshot(), None, recovered=True)
                row = self.journal.dispatch(self.dispatch_id)
            return _unpack(self.journal.result(row))
        self.journal.claim(self.dispatch_id, request, max_output_tokens)
        began = time.perf_counter()
        try:
            reply = await self.adapter.complete(request, max_output_tokens=max_output_tokens, dispatch_id=self.dispatch_id)
            result = _envelope(reply)
        except DispatchBlocked as exc:
            result = {"failure":exc.reason,"control_refusal":exc.details,"control_outcome":exc.outcome}
        except TransportFailure as exc:
            reply = ModelReply(content=exc.response or "", transport=exc.transport, usage=exc.usage, usage_capture=exc.usage_capture)
            result = _envelope(reply, exc.reason)
        except asyncio.CancelledError:
            self.last_failure = self.adapter.last_failure
            if self.last_failure is not None:
                failure = self.last_failure
                result = _envelope(ModelReply(content=failure.response or "",transport=failure.transport,
                    usage=failure.usage,usage_capture=failure.usage_capture),failure.reason)
                self.journal.delivered(self.dispatch_id,result,self.adapter.controls.snapshot(),(time.perf_counter()-began)*1000)
            raise
        self.journal.delivered(self.dispatch_id, result, self.adapter.controls.snapshot(), (time.perf_counter()-began)*1000)
        return _unpack(result)


async def execute_coordinated(output: Path, *, registration: Registration | None = None, run_id: str | None = None,
                              cohort: Path | None = None, resume=False, descriptor: CodexDescriptor | None = None):
    """Create or resume one durable run; the run lock spans all provider activity."""
    output = output.resolve()
    if not resume:
        if registration is None or run_id is None:
            raise ValueError("new run needs a registration and scheduled run ID")
        if output.exists():
            raise ValueError("run directory already exists; use trial resume")
        if cohort is not None:
            frozen = load_registration(cohort)
            if frozen != registration:
                raise ValueError("run settings differ from cohort registration")
            _claim_run(cohort, registration, run_id, output, resume=False)
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "journal.sqlite3").is_file():
        raise ValueError("run has no durable journal; legacy trials cannot be auto-resumed")
    with run_lock(output / "run.lock"):
        journal = RunJournal(output, create=not resume)
        try:
            if resume:
                manifest = journal.manifest()
                registration = Registration.model_validate(manifest["registration"])
                run_id = manifest["run_id"]
                cohort = Path(manifest["cohort"]) if manifest["cohort"] else None
                if cohort is not None:
                    if load_registration(cohort) != registration:
                        raise ValueError("cohort registration changed")
                    _claim_run(cohort, registration, run_id, output, resume=True)
                state = (journal.state() if journal.db.execute("SELECT 1 FROM states LIMIT 1").fetchone()
                         else TrialState.model_validate(manifest["initial_state"]))
            else:
                state = TrialSession.start(registration.experiment, run_id, registration.limits, registration.capture).state
                manifest = {"registration": registration.model_dump(mode="json"), "run_id": run_id,
                            "cohort": str(cohort.resolve()) if cohort else None,"initial_state":state.model_dump(mode="json")}
                journal.save_manifest(manifest)
                journal.checkpoint(state)
            terminal = journal.db.execute("SELECT * FROM terminal").fetchone()
            if terminal is not None:
                from .models import Trace
                trace = Trace.model_validate_json(terminal[0])
                if fingerprint(json.loads(terminal[0])) != terminal["digest"]:
                    raise ValueError("terminal trace checksum mismatch")
            else:
                descriptor = descriptor or describe_codex(registration.config)
                registration.verify_runtime(descriptor)
                _publish(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                _publish(output / "capture.json", registration.capture.model_dump_json(indent=2))
                transport_path = output / "transport"
                adapter = CodexExecTransport(registration.config, transport_path, descriptor=descriptor,
                                             resume=resume and transport_path.exists())
                wrapped = JournaledTransport(journal, adapter)
                trace = await run_trial(registration.experiment, run_id, registration.limits, registration.capture,
                                        wrapped, checkpoint=journal.checkpoint, state=state)
                deviations = [] if cohort else ["cohort_registration_missing"]
                records = []
                for row in journal.rows():
                    if row["recovered"]:
                        deviations.append("delivery_recovered_after_interruption")
                    result = journal.result(row) if row["result"] else None
                    if result is None or result["failure"] == "dispatch_delivery_uncertain":
                        deviations.append("dispatch_delivery_uncertain")
                    records.append({"dispatch_id":row["id"],"request_digest":fingerprint(json.loads(row["request"])),
                        "claimed_at":row["claimed_at"],"client_call_elapsed_ms":row["elapsed_ms"],
                        "provider_latency_ms":None,"recovered":bool(row["recovered"]),
                        "failure":result["failure"] if result else "unsettled"})
                trace.coordination = {"registration_digest":registration.digest if cohort else None,
                    "deviations":sorted(set(deviations)),"dispatches":records,"resumed":resume,
                    "retry_policy":registration.retry_policy,"provider_latency_ms":None,
                    "timing":{"recorded_client_call_ms":sum(row["client_call_elapsed_ms"] for row in records
                                                             if row["client_call_elapsed_ms"] is not None),
                              "calls_with_unknown_latency":sum(row["client_call_elapsed_ms"] is None for row in records),
                              "other_wall_time_ms":max(0.0,trace.elapsed_ms-sum(row["client_call_elapsed_ms"] for row in records))
                              if all(row["client_call_elapsed_ms"] is not None for row in records) else None},
                    "timing_basis":"wall-clock trial includes downtime; client call time excludes runner coordination; provider latency unknown"}
                with journal.db:
                    journal.db.execute("INSERT INTO terminal VALUES (?,?)",(trace.model_dump_json(),fingerprint(trace.model_dump(mode="json"))))
            _publish(output / "trace.json",trace.model_dump_json(indent=2)+"\n")
            ledger = ledger_from_traces([trace], category="evaluation")
            _publish(output / "usage-ledger.json",ledger.model_dump_json(indent=2)+"\n")
            _publish(output / "accounting-report.json",json.dumps(ledger_report(ledger),indent=2,allow_nan=False)+"\n")
            return trace
        finally:
            journal.close()

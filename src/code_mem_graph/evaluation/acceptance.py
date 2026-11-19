"""Explicit acceptance commands in a separate frozen-source checkout.

This is a trusted-code test harness, not a security sandbox. Suite descriptions
are never interpreted as commands; only a separately registered argv plan runs.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Literal

from pydantic import Field, model_validator

from code_mem_graph.knowledge.models import Digest, SourcePath
from code_mem_graph.knowledge.sources import _safe_path
from code_mem_graph.serialization import fingerprint

from .models import Contract, Suite
from .snapshot import Snapshot, SnapshotFile, capture_snapshot


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class AcceptanceCheck(Contract):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    argv: list[str] = Field(min_length=1)
    timeout_ms: float = Field(gt=0)


class AcceptancePlan(Contract):
    allowed_changes: list[SourcePath] = Field(min_length=1)
    trusted_files: list[SourcePath] = Field(min_length=1)
    checks: list[AcceptanceCheck] = Field(min_length=1)
    max_elapsed_ms: float = Field(gt=0)
    max_stdout_bytes: int = Field(gt=0)
    max_stderr_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def nonoverlapping_paths(self):
        if ("." in self.allowed_changes or "." in self.trusted_files
                or len(set(self.allowed_changes)) != len(self.allowed_changes)
                or len(set(self.trusted_files)) != len(self.trusted_files)
                or set(self.allowed_changes)&set(self.trusted_files)
                or any(path.startswith(".project-knowledge/") for path in self.allowed_changes)):
            raise ValueError("allowed changes and trusted files must be distinct explicit file paths")
        if len({entry.id for entry in self.checks}) != len(self.checks):
            raise ValueError("acceptance check IDs must be unique")
        return self


class CandidatePatch(Contract):
    schema_version: Literal[1] = 1
    base_snapshot_digest: Digest
    changes: dict[SourcePath, str | None] = Field(min_length=1)

    @property
    def digest(self):
        return fingerprint(self.model_dump(mode="json"))


class AcceptanceRegistration(Contract):
    schema_version: Literal[1] = 1
    harness_version: Literal["1"] = "1"
    suite: Suite
    task_id: str
    snapshot: Snapshot
    plan: AcceptancePlan
    executable_hashes: dict[str,Digest]

    @model_validator(mode="after")
    def declared_change(self):
        task = self.suite.task(self.task_id)
        revision = next(entry.revision for entry in self.suite.repositories if entry.id==task.repository_id)
        if task.category != "change" or revision != "snapshot:"+self.snapshot.digest:
            raise ValueError("acceptance registration requires a frozen change task")
        if set(task.acceptance_checks) != {entry.id for entry in self.plan.checks}:
            raise ValueError("registered commands must cover exactly the task's acceptance check IDs")
        paths = {entry.path for entry in self.snapshot.files}
        if set(self.plan.trusted_files)-paths:
            raise ValueError("trusted acceptance files must belong to the frozen snapshot")
        if set(self.executable_hashes) != {entry.argv[0] for entry in self.plan.checks}:
            raise ValueError("all registered executables must be fingerprinted")
        return self

    @property
    def digest(self):
        return fingerprint(self.model_dump(mode="json"))


class CheckExecution(Contract):
    id: str
    argv: list[str]
    executed: bool
    elapsed_ms: float = Field(ge=0)
    exit_code: int | None
    error: str | None
    stdout_b64: str
    stderr_b64: str
    stdout_sha256: Digest
    stderr_sha256: Digest


class AcceptanceReceipt(Contract):
    schema_version: Literal[1] = 1
    harness_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1)
    task_id: str
    suite_digest: Digest
    registration_digest: Digest
    base_snapshot_digest: Digest
    candidate_digest: Digest
    outcome: Literal["passed", "failed", "aborted"]
    checks: list[CheckExecution]
    expected_files: dict[SourcePath,Digest]
    observed_files: dict[str,str]
    elapsed_ms: float = Field(ge=0)


def register_acceptance(root: Path, suite: Suite, task_id: str, plan: AcceptancePlan, output: Path) -> AcceptanceRegistration:
    snapshot = capture_snapshot(root)
    task = suite.task(task_id)
    data = suite.model_dump(mode="json")
    for entry in data["repositories"]:
        if entry["id"]==task.repository_id:
            entry["revision"]="snapshot:"+snapshot.digest
    bound = Suite.model_validate(data)
    normalized = plan.model_copy(deep=True)
    hashes = {}
    for check in normalized.checks:
        executable = shutil.which(check.argv[0])
        if executable is None:
            raise ValueError("registered acceptance executable not found")
        executable = str(Path(executable).resolve())
        if os.name == "nt" and Path(executable).suffix.casefold() != ".exe":
            raise ValueError("Windows acceptance commands require a native executable, not a shell wrapper")
        check.argv[0]=executable
        hashes[executable]=_sha(Path(executable).read_bytes())
    registration = AcceptanceRegistration(suite=bound,task_id=task_id,snapshot=snapshot,plan=normalized,executable_hashes=hashes)
    output.mkdir(parents=True,exist_ok=False)
    (output/'registration.json').write_text(registration.model_dump_json(indent=2),encoding='utf-8')
    return registration


def materialized_files(registration,patch):
    if patch.base_snapshot_digest != registration.snapshot.digest or set(patch.changes)-set(registration.plan.allowed_changes):
        raise ValueError("candidate changes differ from the registered base/scope")
    files = {entry.path:entry.content for entry in registration.snapshot.files}
    for path,content in patch.changes.items():
        if content is None:
            files.pop(path,None)
        else:
            files[path]=content
    # A new child cannot traverse through a file already in the snapshot.
    for path in files:
        if any('/'.join(path.split('/')[:index]) in files for index in range(1,len(path.split('/')))):
            raise ValueError("candidate has conflicting file/directory paths")
    return files


def candidate_snapshot(registration,patch):
    files = materialized_files(registration,patch)
    policy = fingerprint({"kind":"registered-acceptance-overlay-v1","base_policy":registration.snapshot.policy_digest,
                          "allowed_changes":registration.plan.allowed_changes})
    return Snapshot(policy_digest=policy,files=[SnapshotFile(path=path,content=content,digest=_sha(content.encode('utf-8')))
                                              for path,content in sorted(files.items())])


async def _execute(check,work,output,plan,remaining_ms):
    from .codex_transport import _terminate
    process = None
    buffers = {"stdout":bytearray(),"stderr":bytearray()}
    error = None
    started = time.perf_counter()
    tasks = []
    try:
        env = {key:value for key,value in os.environ.items() if key.upper() in {"PATH","SYSTEMROOT","WINDIR","TEMP","TMP"}}
        env.update(PYTHONPATH=str(work),PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        kwargs = ({"creationflags":subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=="nt"
                  else {"start_new_session":True})
        process = await asyncio.create_subprocess_exec(*check.argv,cwd=work,env=env,stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,**kwargs)

        async def drain(stream,name,cap):
            while chunk := await stream.read(16384):
                retain = chunk[:max(0,cap-len(buffers[name]))]
                buffers[name].extend(retain)
                if len(retain)!=len(chunk):
                    raise OverflowError(name+"_byte_limit")

        tasks = [asyncio.create_task(drain(process.stdout,'stdout',plan.max_stdout_bytes)),
                 asyncio.create_task(drain(process.stderr,'stderr',plan.max_stderr_bytes)),asyncio.create_task(process.wait())]
        async with asyncio.timeout(min(check.timeout_ms,remaining_ms)/1000):
            await asyncio.gather(*tasks)
    except TimeoutError:
        error = "timeout"
    except asyncio.CancelledError:
        error = "cancelled"
    except OverflowError as exc:
        error = str(exc)
    except OSError:
        error = "process_launch_or_io_failure"
    finally:
        if process is not None and process.returncode is None:
            try:
                await _terminate(process)
            except (OSError,TimeoutError):
                error = error or "process_cleanup_failure"
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    for name,value in buffers.items():
        (output/f'{check.id}.{name}').write_bytes(value)
    return CheckExecution(id=check.id,argv=check.argv,executed=process is not None,elapsed_ms=(time.perf_counter()-started)*1000,
        exit_code=process.returncode if process else None,error=error,
        stdout_b64=base64.b64encode(buffers['stdout']).decode('ascii'),stderr_b64=base64.b64encode(buffers['stderr']).decode('ascii'),
        stdout_sha256=_sha(buffers['stdout']),stderr_sha256=_sha(buffers['stderr']))


async def run_acceptance(registration: AcceptanceRegistration, patch: CandidatePatch, run_id: str, output: Path) -> AcceptanceReceipt:
    files = materialized_files(registration,patch)
    for path,digest in registration.executable_hashes.items():
        if _sha(Path(path).read_bytes()) != digest:
            raise ValueError("registered acceptance executable changed")
    output.mkdir(parents=True,exist_ok=False)
    work = (output/'checkout').resolve()
    work.mkdir()
    for path,content in files.items():
        target = work.joinpath(*path.split('/'))
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content.encode('utf-8'))
    expected = {path:_sha(content.encode('utf-8')) for path,content in files.items()}
    started = time.perf_counter()
    results = []
    aborted = False
    for check in registration.plan.checks:
        remaining = registration.plan.max_elapsed_ms-(time.perf_counter()-started)*1000
        if remaining<=0 or aborted:
            results.append(CheckExecution(id=check.id,argv=check.argv,executed=False,elapsed_ms=0.0,exit_code=None,
                error='not_run_after_stop',stdout_b64='',stderr_b64='',stdout_sha256=_sha(b''),stderr_sha256=_sha(b'')))
            continue
        result = await _execute(check,work,output,registration.plan,remaining)
        results.append(result)
        aborted = result.error in {'cancelled','process_cleanup_failure'}
    observed = {}
    for directory,directories,names in os.walk(work,followlinks=False):
        for name in list(directories):
            path=(Path(directory)/name).relative_to(work).as_posix()
            try:
                _safe_path(work,path)
            except (OSError,ValueError):
                observed[path]='unsafe_directory'
                directories.remove(name)
        for name in names:
            path=(Path(directory)/name).relative_to(work).as_posix()
            try:
                observed[path]=_sha(_safe_path(work,path).read_bytes())
            except (OSError,ValueError):
                observed[path]='unavailable_or_unsafe'
    for path in expected:
        observed.setdefault(path,'missing')
    passed = observed==expected and all(entry.executed and entry.exit_code==0 and entry.error is None for entry in results)
    receipt = AcceptanceReceipt(run_id=run_id,task_id=registration.task_id,suite_digest=registration.suite.digest,
        registration_digest=registration.digest,base_snapshot_digest=registration.snapshot.digest,candidate_digest=patch.digest,
        outcome='aborted' if aborted else 'passed' if passed else 'failed',checks=results,expected_files=expected,
        observed_files=observed,elapsed_ms=(time.perf_counter()-started)*1000)
    (output/'receipt.json').write_text(receipt.model_dump_json(indent=2),encoding='utf-8')
    (output/'candidate.json').write_text(patch.model_dump_json(indent=2),encoding='utf-8')
    return receipt


def verify_change_evidence(suite,trace,proof,evidence):
    try:
        registration = AcceptanceRegistration.model_validate(evidence['registration'])
        patch = CandidatePatch.model_validate(evidence['candidate'])
        receipt = AcceptanceReceipt.model_validate(evidence['receipt'])
        expected = {path:_sha(content.encode('utf-8')) for path,content in materialized_files(registration,patch).items()}
        if (registration.suite != suite or registration.task_id!=trace.task_id
                or trace.repository_revision!='snapshot:'+registration.snapshot.digest
                or candidate_snapshot(registration,patch).digest!=proof.snapshot_digest
                or receipt.run_id!=trace.run_id or receipt.task_id!=trace.task_id or receipt.suite_digest!=suite.digest
                or receipt.registration_digest!=registration.digest or receipt.base_snapshot_digest!=registration.snapshot.digest
                or receipt.candidate_digest!=patch.digest or receipt.expected_files!=expected):
            return ['change_execution_binding_mismatch']
        issues = []
        if receipt.observed_files!=expected:
            issues.append('acceptance_checkout_modified')
        if receipt.elapsed_ms>registration.plan.max_elapsed_ms:
            issues.append('acceptance_elapsed_limit_exceeded')
        if [entry.id for entry in receipt.checks] != [entry.id for entry in registration.plan.checks]:
            issues.append('acceptance_checks_missing')
        for result,check in zip(receipt.checks,registration.plan.checks):
            if result.argv!=check.argv or not result.executed or result.exit_code!=0 or result.error is not None:
                issues.append('acceptance_command_not_passed')
            for name,cap in (('stdout',registration.plan.max_stdout_bytes),('stderr',registration.plan.max_stderr_bytes)):
                value = base64.b64decode(getattr(result,name+'_b64'),validate=True)
                if len(value)>cap or _sha(value)!=getattr(result,name+'_sha256'):
                    issues.append('acceptance_output_evidence_invalid')
        if receipt.outcome!='passed':
            issues.append('change_acceptance_not_passed')
        return sorted(set(issues))
    except (ValueError,KeyError,TypeError):
        return ['change_execution_evidence_invalid']

"""Frozen-source citation checks and explicit, answer-bound semantic review.

Syntax/range checks never establish entailment. Every nonempty answer line needs
a recorded finding; independent reviewers judge meaning and rubric coverage.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from code_mem_graph.knowledge.models import Digest, SourcePath, portable_path
from code_mem_graph.serialization import fingerprint

from .models import AgentProfile, Contract, ModelEvent, Review, Suite, Trace
from .snapshot import Snapshot, SnapshotFile


class SourceIdentity(Contract):
    path: SourcePath
    digest: Digest


class SourceProof(Contract):
    schema_version: Literal[1] = 1
    policy_digest: Digest
    catalog: list[SourceIdentity]
    sources: list[SnapshotFile]

    @model_validator(mode="after")
    def valid_sources(self):
        paths = [entry.path for entry in self.catalog]
        if paths != sorted(set(paths)):
            raise ValueError("proof catalog must be sorted and unique")
        expected = {entry.path:entry.digest for entry in self.catalog}
        supplied = [entry.path for entry in self.sources]
        if len(supplied) != len(set(supplied)) or any(expected.get(entry.path) != entry.digest for entry in self.sources):
            raise ValueError("proof sources differ from the frozen catalog")
        return self

    @property
    def snapshot_digest(self):
        return fingerprint({"schema_version":1,"policy_digest":self.policy_digest,
                            "files":[entry.model_dump(mode="json") for entry in self.catalog]})


class AnswerUnit(Contract):
    id: str
    line: int = Field(ge=1)
    text: str = Field(min_length=1)


class CitationCheck(Contract):
    id: str
    unit_id: str
    raw: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    status: Literal["valid", "unsafe_path", "outside_frozen_catalog", "invalid_line_range", "malformed_reference", "missing_source_proof"]
    source_digest: str | None = None
    excerpt: str | None = None


class ReviewerIdentity(Contract):
    name: str = Field(min_length=1)
    kind: Literal["human", "model"]
    independent: bool
    selection_basis: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    profile: AgentProfile | None = None

    @model_validator(mode="after")
    def model_identity(self):
        if self.kind == "model" and self.profile is None:
            raise ValueError("model reviews require a recorded reviewer profile")
        return self


class UnitFinding(Contract):
    unit_id: str
    kind: Literal["factual", "limitation", "nonclaim"]
    verdict: Literal["supported", "unsupported", "contradicted", "unknown", "not_claim"]
    citation_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)


class CriterionFinding(Contract):
    criterion_id: str
    status: Literal["met", "not_met", "unknown"]
    unit_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)


class CheckFinding(Contract):
    check_id: str
    status: Literal["passed", "failed", "not_run"]
    unit_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)


class SemanticReview(Contract):
    reviewer: ReviewerIdentity
    answer_digest: Digest
    snapshot_digest: Digest
    context_digest: Digest | None = None
    units: list[UnitFinding]
    criteria: list[CriterionFinding]
    checks: list[CheckFinding] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def omit_unbound_context(self,handler):
        data=handler(self)
        if data.get('context_digest') is None:
            data.pop('context_digest',None)
        return data

    @model_validator(mode="after")
    def unique_findings(self):
        for entries,key in ((self.units,"unit_id"),(self.criteria,"criterion_id"),(self.checks,"check_id")):
            values = [getattr(entry,key) for entry in entries]
            if len(values) != len(set(values)):
                raise ValueError("duplicate review finding ID")
        return self


class AnswerAudit(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    task_id: str
    suite_digest: Digest
    answer: str = Field(min_length=1)
    answer_digest: Digest
    base_snapshot_digest: Digest
    proof: SourceProof
    units: list[AnswerUnit]
    citations: list[CitationCheck]
    reviews: list[SemanticReview] = Field(default_factory=list)
    # Change validation is optional; the acceptance harness validates these data.
    change: dict | None = None


def answer_digest(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def review_context_digest(audit: AnswerAudit) -> str:
    context={"run_id":audit.run_id,"task_id":audit.task_id,"suite_digest":audit.suite_digest,
        "answer_digest":audit.answer_digest,"base_snapshot_digest":audit.base_snapshot_digest,
        "evidence_snapshot_digest":audit.proof.snapshot_digest}
    if audit.change is not None:
        context['change_evidence_digest']=fingerprint(audit.change)
    return fingerprint(context)


def final_answer(trace: Trace) -> str | None:
    for event in reversed(trace.events):
        if not isinstance(event,ModelEvent) or not event.response_observed:
            continue
        try:
            value = json.loads(event.response) if isinstance(event.response,str) else event.response
        except ValueError:
            return None
        if (isinstance(value,dict) and value.get("tool_calls") == [] and isinstance(value.get("final"),str)
                and value["final"].strip()):
            return value["final"]
        return None
    return None


def answer_units(answer: str) -> list[AnswerUnit]:
    return [AnswerUnit(id=f"u{index}",line=line,text=text)
            for index,(line,text) in enumerate(((line,text) for line,text in enumerate(answer.splitlines(),1) if text.strip()),1)]


# Preserve the whole path token so external URLs, encoded traversal and Windows
# absolute paths cannot be mistaken for a safe suffix inside the catalog.
_REFERENCE = re.compile(r"[^\s\[\]()`<>{}\"']+")


def check_citations(units: list[AnswerUnit], proof: SourceProof) -> list[CitationCheck]:
    from urllib.parse import unquote
    catalog = {entry.path:entry.digest for entry in proof.catalog}
    sources = {entry.path:entry for entry in proof.sources}
    results = []
    for unit in units:
        for match in _REFERENCE.finditer(unit.text):
            token = match[0].rstrip(".,;!")
            if "#L" in token:
                path,address = token.rsplit("#L",1)
            elif ":" in token:
                path,address = token.rsplit(":",1)
            else:
                continue
            path = unquote(path)
            if not any(character in path for character in (".","/","\\")):
                continue  # Ordinary times/numeric ratios are not file references.
            numbers = re.fullmatch(r"L?(-?\d+)(?:-L?(-?\d+))?",address) if len(address)<=32 else None
            start = int(numbers[1]) if numbers else None
            end = int(numbers[2] or numbers[1]) if numbers else None
            status,excerpt,digest = "valid",None,None
            try:
                portable_path(path)
            except ValueError:
                status = "unsafe_path"
            if status == "valid":
                if path not in catalog:
                    status = "outside_frozen_catalog"
                elif numbers is None:
                    status = "malformed_reference"
                elif path not in sources:
                    status = "missing_source_proof"
                else:
                    digest = catalog[path]
                    lines = sources[path].content.splitlines()
                    if start < 1 or end < start or end > len(lines):
                        status = "invalid_line_range"
                    else:
                        excerpt = "\n".join(lines[start-1:end])
            results.append(CitationCheck(id=f"c{len(results)+1}",unit_id=unit.id,raw=match[0],path=path,
                start_line=start,end_line=end,status=status,source_digest=digest,excerpt=excerpt))
    return results


def make_audit(suite: Suite, snapshot: Snapshot, trace: Trace, *, reviews: list[SemanticReview] | None = None,
               change: dict | None = None) -> AnswerAudit:
    answer = final_answer(trace)
    if answer is None:
        raise ValueError("trace has no structured final answer to audit")
    task = suite.task(trace.task_id)
    revision = next(entry.revision for entry in suite.repositories if entry.id == task.repository_id)
    if trace.suite_digest != suite.digest or trace.repository_revision != revision or revision != "snapshot:" + snapshot.digest:
        raise ValueError("audit must use the trace's frozen suite and source snapshot")
    if change is not None and task.category != 'change':
        raise ValueError("discovery reviews cannot substitute a candidate source overlay")
    review_snapshot = snapshot
    if change is not None:
        from .acceptance import AcceptanceRegistration, CandidatePatch, candidate_snapshot
        registration = AcceptanceRegistration.model_validate(change['registration'])
        if registration.snapshot.digest != snapshot.digest:
            raise ValueError("change registration differs from the trace base")
        review_snapshot = candidate_snapshot(registration,CandidatePatch.model_validate(change['candidate']))
    proof = SourceProof(policy_digest=review_snapshot.policy_digest,
        catalog=[SourceIdentity(path=entry.path,digest=entry.digest) for entry in review_snapshot.files],sources=review_snapshot.files)
    units = answer_units(answer)
    citations = check_citations(units,proof)
    needed = {entry.path for entry in citations if entry.path in {source.path for source in review_snapshot.files}}
    needed.update(source.path for source in review_snapshot.files for criterion in task.criteria for ref in criterion.source_refs
                  if ref=='.' or source.path==ref or source.path.startswith(ref.rstrip('/')+'/'))
    proof = proof.model_copy(update={"sources":[entry for entry in review_snapshot.files if entry.path in needed]})
    return AnswerAudit(run_id=trace.run_id,task_id=trace.task_id,suite_digest=suite.digest,answer=answer,
        answer_digest=answer_digest(answer),base_snapshot_digest=snapshot.digest,proof=proof,units=units,citations=citations,
        reviews=reviews or [],change=change)


def grade_audit(suite: Suite, trace: Trace, audit: AnswerAudit) -> dict:
    audit = AnswerAudit.model_validate(audit.model_dump(mode="json"))
    task = suite.task(trace.task_id)
    revision = next(entry.revision for entry in suite.repositories if entry.id == task.repository_id)
    if (audit.run_id != trace.run_id or audit.task_id != trace.task_id or audit.suite_digest != suite.digest
            or trace.suite_digest != suite.digest or trace.repository_revision != revision
            or revision != "snapshot:"+audit.base_snapshot_digest or final_answer(trace) != audit.answer
            or answer_digest(audit.answer) != audit.answer_digest):
        raise ValueError("answer audit binding mismatch")
    if task.category != 'change' and (audit.change is not None or audit.base_snapshot_digest != audit.proof.snapshot_digest):
        raise ValueError("discovery proof must use only the frozen base source")
    if audit.change is None and audit.base_snapshot_digest != audit.proof.snapshot_digest:
        raise ValueError("discovery proof differs from the frozen base")
    if audit.units != answer_units(audit.answer) or audit.citations != check_citations(audit.units,audit.proof):
        raise ValueError("citation/unit results differ from deterministic frozen evidence checks")
    issues,disagreements = [],[]
    citations = {entry.id:entry for entry in audit.citations}
    units = {entry.id for entry in audit.units}
    criteria = {entry.id for entry in task.criteria}
    checks = set(task.acceptance_checks)
    invalid = [entry.id for entry in audit.citations if entry.status != "valid"]
    if invalid:
        issues.append("invalid_citations")
    catalog_paths={entry.path for entry in audit.proof.catalog}
    supplied_paths={entry.path for entry in audit.proof.sources}
    for criterion in task.criteria:
        required={path for path in catalog_paths for ref in criterion.source_refs
                  if ref=='.' or path==ref or path.startswith(ref.rstrip('/')+'/')}
        if not required or required-supplied_paths:
            issues.append("criterion_source_evidence_unavailable")
    if not audit.reviews:
        issues.append("semantic_review_missing")
    signatures = {"units":{},"criteria":{},"checks":{}}
    reviewers = []
    for review in audit.reviews:
        identity = fingerprint({"kind":review.reviewer.kind,"reference":review.reviewer.reference})
        if identity in reviewers:
            issues.append("duplicate_reviewer")
        reviewers.append(identity)
        if not review.reviewer.independent:
            issues.append("independent_review_missing")
        if review.answer_digest != audit.answer_digest or review.snapshot_digest != audit.proof.snapshot_digest:
            raise ValueError("semantic review belongs to a different answer/snapshot")
        if review.context_digest is None:
            issues.append("review_context_unbound")
        elif review.context_digest != review_context_digest(audit):
            raise ValueError("semantic review belongs to a different run/task/rubric context")
        actual_units = {entry.unit_id for entry in review.units}
        actual_criteria = {entry.criterion_id for entry in review.criteria}
        actual_checks = {entry.check_id for entry in review.checks}
        if actual_units != units or actual_criteria != criteria or actual_checks != checks:
            issues.append("review_coverage_incomplete")
        for entry in review.units:
            if entry.unit_id not in units or set(entry.citation_ids)-citations.keys():
                raise ValueError("review references an unknown unit/citation")
            signatures["units"].setdefault(entry.unit_id,[]).append((entry.kind,entry.verdict))
            if entry.verdict not in {"supported","not_claim"}:
                issues.append("unsupported_or_uncertain_answer_unit")
            if entry.kind == "nonclaim" and entry.verdict != "not_claim" or entry.kind != "nonclaim" and entry.verdict == "not_claim":
                issues.append("inconsistent_unit_classification")
            if entry.kind == "factual" and (not entry.citation_ids or any(citations[key].status != "valid" for key in entry.citation_ids)):
                issues.append("factual_unit_without_valid_evidence")
        for entry in review.criteria:
            if entry.criterion_id not in criteria or set(entry.unit_ids)-units:
                raise ValueError("review references an unknown criterion/unit")
            signatures["criteria"].setdefault(entry.criterion_id,[]).append(entry.status)
            if entry.status != "met" or not entry.unit_ids:
                issues.append("critical_or_required_criterion_unmet")
        for entry in review.checks:
            if entry.check_id not in checks or set(entry.unit_ids)-units:
                raise ValueError("review references an unknown check/unit")
            signatures["checks"].setdefault(entry.check_id,[]).append(entry.status)
            if task.category != "change" and (entry.status != "passed" or not entry.unit_ids):
                issues.append("answer_acceptance_check_unmet")
    for kind,entries in signatures.items():
        for key,opinions in entries.items():
            if len(set(opinions)) > 1:
                disagreements.append({"kind":kind,"id":key,"opinions":[list(value) if isinstance(value,tuple) else value for value in opinions]})
    if disagreements:
        issues.append("review_disagreement")
    change_status = "not_applicable"
    if task.category == "change":
        if audit.change is None:
            issues.append("change_execution_evidence_missing")
            change_status = "not_run"
        else:
            from .acceptance import verify_change_evidence
            change_issues = verify_change_evidence(suite,trace,audit.proof,audit.change)
            issues.extend(change_issues)
            change_status = "passed" if not change_issues else "not_verified"
    return {"status":"disputed" if disagreements else "verified" if not issues else "unreviewed" if not audit.reviews else "rejected",
        "blockers":sorted(set(issues)),"answer_digest":audit.answer_digest,"snapshot_digest":audit.proof.snapshot_digest,
        "units":len(units),"citations":len(citations),"invalid_citation_ids":invalid,
        "reviewers":[entry.reviewer.model_dump(mode="json") for entry in audit.reviews],"disagreements":disagreements,
        "change_execution":change_status,"basis":"frozen citation checks plus recorded independent semantic review; not an automatic entailment proof"}


def validation_summary(suite: Suite, trace: Trace) -> dict:
    if trace.review.validation is None:
        return {"status":"unverified","blockers":["answer_validation_undeclared"],"change_execution":"not_verified" if suite.task(trace.task_id).category=="change" else "not_applicable"}
    try:
        return grade_audit(suite,trace,AnswerAudit.model_validate(trace.review.validation))
    except (ValueError,KeyError,TypeError) as exc:
        return {"status":"invalid","blockers":["answer_validation_invalid"],"error":str(exc),"change_execution":"not_verified"}


def reviewed_trace(suite: Suite, trace: Trace, audit: AnswerAudit) -> Trace:
    summary = grade_audit(suite,trace,audit)
    criteria = {entry.id:("met" if audit.reviews and all(any(f.criterion_id==entry.id and f.status=="met" for f in review.criteria)
                 for review in audit.reviews) else "unknown") for entry in suite.task(trace.task_id).criteria}
    checks = {key:("passed" if (summary["change_execution"]=="passed" if suite.task(trace.task_id).category=="change" else
              audit.reviews and all(any(f.check_id==key and f.status=="passed" for f in review.checks) for review in audit.reviews))
              else "not_run") for key in suite.task(trace.task_id).acceptance_checks}
    review = Review(verdict="accepted" if summary["status"]=="verified" else "rejected" if audit.reviews else "unreviewed",
        reviewer=", ".join(entry.reviewer.name for entry in audit.reviews) or None,criteria=criteria,acceptance_checks=checks,
        validation=audit.model_dump(mode="json"))
    return trace.model_copy(update={"review":review},deep=True)

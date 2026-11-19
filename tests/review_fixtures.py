"""Explicit fictional review data for report tests; no model/human calls."""
import hashlib

from code_mem_graph.evaluation.models import AgentProfile
from code_mem_graph.evaluation.snapshot import Snapshot, SnapshotFile
from code_mem_graph.evaluation.validation import (
    CheckFinding, CriterionFinding, ReviewerIdentity, SemanticReview, UnitFinding, make_audit, review_context_digest,
)


def source_snapshot():
    sources={'README.md':'A fixture project.\n','CONTRIBUTING.md':'Run pytest tests to validate it.\n'}
    return Snapshot(policy_digest='a'*64,files=[SnapshotFile(path=path,content=text,digest=hashlib.sha256(text.encode()).hexdigest())
                                               for path,text in sorted(sources.items())])


def supported_audit(suite,trace,snapshot,*,change=None):
    audit=make_audit(suite,snapshot,trace,change=change)
    citations=[entry.id for entry in audit.citations if entry.status=='valid']
    units=[entry.id for entry in audit.units]
    task=suite.task(trace.task_id)
    reviewer=ReviewerIdentity(name='fictional-independent-reviewer',kind='model',independent=True,
        selection_basis='Synthetic unit-test findings, not an actual human/model assessment.',reference='fixture:'+trace.run_id,
        profile=AgentProfile(kind='scripted',provider='fixture',model='fictional-reviewer',model_version='1',instructions_digest='fixture',settings={}))
    opinion=SemanticReview(reviewer=reviewer,answer_digest=audit.answer_digest,snapshot_digest=audit.proof.snapshot_digest,
        context_digest=review_context_digest(audit),
        units=[UnitFinding(unit_id=entry.id,kind='factual',verdict='supported',citation_ids=citations,note='Fictional test assessment.') for entry in audit.units],
        criteria=[CriterionFinding(criterion_id=entry.id,status='met',unit_ids=units,note='Fictional test assessment.') for entry in task.criteria],
        checks=[CheckFinding(check_id=key,status='passed',unit_ids=units,note='Answer criterion only; not command execution.') for key in task.acceptance_checks])
    audit.reviews=[opinion]
    return audit

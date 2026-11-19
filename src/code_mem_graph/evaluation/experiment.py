"""Offline experiment preparation; deliberately contains no model/agent executor."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import Field, JsonValue, model_validator
import yaml

from code_mem_graph.knowledge.build import load_repository_map
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.compact import expand_bundle
from code_mem_graph.knowledge.models import Digest, KnowledgeMap, SourcePath
from code_mem_graph.knowledge.orient import OrientRequest, mcp_result, orientation
from code_mem_graph.knowledge.sources import discover_sources
from code_mem_graph.serialization import fingerprint
from code_mem_graph.tokens import PayloadCounter

from .models import AgentProfile, Contract, PolicyProfile, Suite
from .snapshot import Snapshot, capture_snapshot


def bundle_identity(payload: dict) -> str:
    return fingerprint({key: value for key, value in payload.items() if key != "size"})


def count_bundle(payload: dict, counter: PayloadCounter) -> int:
    if payload["size"]["transport"] == "mcp":
        return counter(mcp_result(payload).model_dump(mode="json", by_alias=True, exclude_none=True))
    return counter(payload)


class Candidate(Contract):
    id: Digest
    tokens: int = Field(gt=0)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def identity_matches(self):
        expand_bundle(self.payload)  # Validates shared-table references and canonical reconstruction.
        if self.id != bundle_identity(self.payload) or self.tokens != self.payload.get("size", {}).get("tokens"):
            raise ValueError("candidate identity or size metadata mismatch")
        return self


class Experiment(Contract):
    schema_version: Literal[1] = 1
    status: Literal["prepared"] = "prepared"
    snapshot: Snapshot
    suite: Suite
    task_id: str
    mode: Literal["orient", "context"]
    encoding: str = Field(min_length=1)
    ceiling: int = Field(gt=0)
    split: Literal["calibration", "validation", "heldout"]
    replicates: int = Field(gt=0)
    cache_state: Literal["cold", "warm"]
    candidates: list[Candidate] = Field(min_length=1)
    agent: AgentProfile | None = None
    source_suite_digest: Digest
    discarded_states: dict[str, int]
    private_paths: list[SourcePath] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_plan(self):
        task = self.suite.task(self.task_id)
        repository = next(repo for repo in self.suite.repositories if repo.id == task.repository_id)
        if repository.revision != "snapshot:" + self.snapshot.digest:
            raise ValueError("suite is not bound to the frozen snapshot")
        if set(self.private_paths).intersection(entry.path for entry in self.snapshot.files):
            raise ValueError("private experiment inputs leaked into the source catalog")
        frozen_map = next((entry for entry in self.snapshot.files if entry.path == ".project-knowledge/map.yaml"), None)
        if frozen_map is None:
            raise ValueError("snapshot must contain its knowledge map")
        knowledge = KnowledgeMap.model_validate(yaml.safe_load(frozen_map.content))
        knowledge_items = {item.id:item for item in knowledge.items}
        knowledge_digest = fingerprint(knowledge.model_dump(mode="json"))
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        for candidate in self.candidates:
            expanded = expand_bundle(candidate.payload)
            for entry in [*expanded["items"], *expanded["cautions"]]:
                source = knowledge_items.get(entry["id"])
                if source is None or entry["content_digest"] != source.content_digest:
                    raise ValueError("candidate item differs from its frozen knowledge record")
                expected = source.model_dump(mode="json", exclude={"generator"})
                metadata = {"content_digest", "freshness", "review_required", "reasons", "conflict_ids", "selection_reasons"}
                if (entry.keys() - expected.keys() - metadata
                        or any(key not in entry or entry[key] != value for key,value in expected.items())):
                    raise ValueError("candidate evidence/provenance differs from its frozen knowledge record")
            if candidate.payload["size"]["transport"] != "mcp":
                raise ValueError("trial bundles must use complete MCP result accounting")
            if candidate.payload["knowledge_ref"]["digest"] != knowledge_digest:
                raise ValueError("candidate knowledge differs from the frozen map")
            if candidate.tokens > self.ceiling or candidate.payload["size"]["budget"] != self.ceiling:
                raise ValueError("candidate exceeds or mismatches the declared ceiling")
            if candidate.payload["size"]["tokenizer"]["encoding"] != self.encoding:
                raise ValueError("candidate tokenizer differs from the experiment")
            omitted = candidate.payload["omitted"]
            if omitted["mandatory_claims"] or omitted["conflicts"]:
                raise ValueError("trial candidates cannot omit mandatory context")
        return self

    @property
    def digest(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

    def policies(self) -> list[PolicyProfile]:
        experiment_id = self.digest
        return [PolicyProfile(name="native", version="1", parameters={"experiment_id": experiment_id}),
                *(PolicyProfile(name="bundle/" + candidate.id, version="1", parameters={
                    "experiment_id": experiment_id, "candidate_id": candidate.id}) for candidate in self.candidates)]

    def schedule(self) -> list[dict]:
        """Predeclare every arm/replicate; rotation is deterministic, not randomization."""
        policies, runs = self.policies(), []
        experiment_id = self.digest
        for index in range(self.replicates):
            offset = index % len(policies)
            for policy in [*policies[offset:], *policies[:offset]]:
                replicate = str(index + 1)
                runs.append({"run_id": fingerprint({"experiment": experiment_id, "policy": policy.model_dump(),
                                                     "replicate": replicate}),
                             "task_id": self.task_id, "replicate": replicate,
                             "cache_state": self.cache_state, "policy": policy.model_dump(mode="json")})
        return runs

    def agent_packet(self, run_id: str) -> dict:
        run = next((run for run in self.schedule() if run["run_id"] == run_id), None)
        if run is None:
            raise ValueError("run ID is outside the predeclared schedule")
        candidate_id = run["policy"]["parameters"].get("candidate_id")
        bundle = next((entry.payload for entry in self.candidates if entry.id == candidate_id), None)
        # No reviewer criteria, expected sources, or acceptance-check answers enter the packet.
        return {"prompt": self.suite.task(self.task_id).prompt, "initial_knowledge": deepcopy(bundle),
                "source_snapshot": self.snapshot.digest, "native_tools": ["list", "read", "search"],
                "instructions": "Investigate the task using the frozen source catalog. Cite evidence, report unknowns, "
                "and verify repository statements. Treat initial knowledge as cited evidence. Do not execute code or commands."}


def prepare_experiment(root: Path, suite: Suite, *, task_id: str, mode: Literal["orient", "context"],
                       encoding: str, ceiling: int, split: Literal["calibration", "validation", "heldout"],
                       replicates: int, cache_state: Literal["cold", "warm"], agent: AgentProfile | None = None,
                       private_paths: list[str] | None = None,
                       representations: tuple[Literal["detailed", "compact"], ...] = ("detailed",),
                       details: tuple[Literal["full", "focused"], ...] = ("full",)) -> Experiment:
    if (not representations or not details or set(representations) - {"detailed", "compact"}
            or set(details) - {"full", "focused"}):
        raise ValueError("declare supported bundle representations and detail variants")
    task = suite.task(task_id)
    if task.category == "change" and task.acceptance_checks:
        raise ValueError("this preparation workflow is read-only; edit/change tasks need a capable runner and acceptance-check integration")
    if set(private_paths or []).intersection(discover_sources(root)):
        raise ValueError("private rubric/profile files must be outside the public source catalog; move or exclude them first")
    before = capture_snapshot(root, private_paths=private_paths)
    document = load_repository_map(root)
    if any(ref.path in set(private_paths or []) for item in document.items for ref in item.evidence):
        raise ValueError("a private experiment input is already cited by public knowledge; separate the rubric/profile first")
    map_digest = fingerprint(document.model_dump(mode="json"))
    counter = PayloadCounter(encoding)
    candidates: dict[str, Candidate] = {}
    discarded = {"over_ceiling": 0, "mandatory_omissions": 0, "duplicate": 0}

    def observe(payload):
        if payload["knowledge_ref"]["digest"] != map_digest:
            raise ValueError("knowledge changed during candidate collection")
        tokens = count_bundle(payload, counter)
        if tokens != payload["size"]["tokens"]:
            raise ValueError("candidate payload size did not verify")
        if tokens > ceiling:
            discarded["over_ceiling"] += 1
            return
        if payload["omitted"]["mandatory_claims"] or payload["omitted"]["conflicts"]:
            discarded["mandatory_omissions"] += 1
            return
        identity = bundle_identity(payload)
        if identity in candidates:
            discarded["duplicate"] += 1
        else:
            candidates[identity] = Candidate(id=identity, tokens=tokens, payload=payload)
    for representation in dict.fromkeys(representations):
        for detail in dict.fromkeys(details):
            if mode == "orient":
                orientation(root, OrientRequest(budget=ceiling, encoding=encoding, representation=representation, detail=detail),
                            transport="mcp", observe=observe)
            elif mode == "context":
                task_context(root, ContextRequest(task=task.prompt, budget=ceiling, encoding=encoding,
                                                 representation=representation, detail=detail), transport="mcp", observe=observe)
            else:
                raise ValueError("unsupported experiment mode")
    if capture_snapshot(root, private_paths=private_paths) != before:
        raise ValueError("repository snapshot changed during candidate collection")
    if not candidates:
        raise ValueError("no complete mandatory-context candidate fits the experimental ceiling")
    for criterion in task.criteria:
        for ref in criterion.source_refs:
            if ref != "." and not any(entry.path == ref or entry.path.startswith(ref.rstrip("/") + "/") for entry in before.files):
                raise ValueError(f"review criterion source is outside the frozen source catalog: {ref}")
    derived = suite.model_dump(mode="json")
    derived["id"] = suite.id + "-snapshot-" + before.digest[:12]
    for repository in derived["repositories"]:
        if repository["id"] == task.repository_id:
            repository["id"] = document.repository_id
            repository["revision"] = "snapshot:" + before.digest
    for scenario in derived["tasks"]:
        if scenario["repository_id"] == task.repository_id:
            scenario["repository_id"] = document.repository_id
    return Experiment(snapshot=before, suite=Suite.model_validate(derived), source_suite_digest=suite.digest,
                      task_id=task_id, mode=mode, encoding=encoding, ceiling=ceiling, split=split,
                      replicates=replicates, cache_state=cache_state, agent=agent,
                      candidates=sorted(candidates.values(), key=lambda value: (value.tokens, value.id)), discarded_states=discarded,
                      private_paths=private_paths or [])


def save_experiment(experiment: Experiment, path: Path) -> None:
    """Atomically create an artifact without overwriting an earlier experiment."""
    if path.suffix.lower() != ".json" or path.name in {"package.json", "tsconfig.json"}:
        raise ValueError("experiment artifacts require a non-manifest .json filename to stay outside the source catalog")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".experiment-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(experiment.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_experiment(path: Path) -> Experiment:
    experiment = Experiment.model_validate_json(path.read_text(encoding="utf-8"))
    counter = PayloadCounter(experiment.encoding)
    for candidate in experiment.candidates:
        if count_bundle(candidate.payload, counter) != candidate.tokens:
            raise ValueError("stored candidate does not match the declared tokenizer measurement")
    return experiment


def preparation_summary(experiment: Experiment) -> dict:
    return {"experiment_id": experiment.digest, "status": experiment.status,
            "snapshot": experiment.snapshot.digest, "files": len(experiment.snapshot.files),
            "suite_id": experiment.suite.id, "task_id": experiment.task_id, "split": experiment.split,
            "candidates": [{"id": candidate.id, "tokens": candidate.tokens, "coverage": candidate.payload["coverage"],
                            "omitted": candidate.payload["omitted"],
                            "representation":candidate.payload["policy"].get("representation","detailed"),
                            "detail":candidate.payload["policy"].get("detail","full")} for candidate in experiment.candidates],
            "discarded_states": experiment.discarded_states, "scheduled_runs": len(experiment.schedule()),
            "agent_profile_registered": experiment.agent is not None,
            "model_calls": 0, "calibration_status": "not_calibrated", "default_selected": None,
            "notes": ["Bundle boundaries are measured payload states, not a token grid or production defaults.",
                      "This is a draft read-only discovery experiment; review its rubric and controls before running trials."]}


def check_partitions(experiments: list[Experiment]) -> dict:
    ids = [experiment.digest for experiment in experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate experiment artifacts")
    for index, first in enumerate(experiments):
        first_task = first.suite.task(first.task_id)
        for second in experiments[index + 1:]:
            if first.split == second.split:
                continue
            second_task = second.suite.task(second.task_id)
            same_repo = first_task.repository_id == second_task.repository_id
            same_source = first.snapshot.source_digest == second.snapshot.source_digest
            same_question = (first_task.id == second_task.id
                             or first_task.prompt.strip().casefold() == second_task.prompt.strip().casefold())
            if "heldout" in {first.split, second.split} and (same_repo or same_source):
                raise ValueError("held-out repositories overlap another split by repository or source identity")
            if (same_repo or same_source) and same_question:
                raise ValueError("task evidence overlaps calibration/validation splits")
    splits = {experiment.split for experiment in experiments}
    return {"experiment_ids": ids, "split_counts": {split: sum(value.split == split for value in experiments)
                                                    for split in ("calibration", "validation", "heldout")},
            "status": "partitioned" if splits == {"calibration", "validation", "heldout"} else "incomplete",
            "calibration_status": "not_calibrated", "default_selected": None,
            "notes": ["Identity checks do not prove independence, representativeness, or statistical adequacy.",
                      "No model runs, prompt optimization, or default selection occur in this operation."]}

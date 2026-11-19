from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_serializer, model_validator

from code_mem_graph.serialization import fingerprint, serialize as serialize


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Criterion(Contract):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    critical: bool
    source_refs: list[str] = Field(min_length=1)


class RepositorySpec(Contract):
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class TaskSpec(Contract):
    id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    category: Literal["orientation", "architecture", "change", "conventions", "workflow", "missing-evidence"]
    prompt: str = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)
    acceptance_checks: dict[str, str]

    @model_validator(mode="after")
    def unique_criteria(self):
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique within a task")
        return self


class Protocol(Contract):
    id: str = Field(min_length=1)
    max_tool_calls: int = Field(gt=0)
    max_model_turns: int = Field(gt=0)
    max_elapsed_ms: float = Field(gt=0)
    stopping_rule: str = Field(min_length=1)


class Suite(Contract):
    schema_version: Literal[1]
    id: str = Field(min_length=1)
    protocol: Protocol
    repositories: list[RepositorySpec] = Field(min_length=1)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_references(self):
        repos = [repo.id for repo in self.repositories]
        tasks = [task.id for task in self.tasks]
        if len(repos) != len(set(repos)) or len(tasks) != len(set(tasks)):
            raise ValueError("repository and task IDs must be unique")
        if any(task.repository_id not in repos for task in self.tasks):
            raise ValueError("every task must reference a declared repository")
        return self

    @property
    def digest(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

    def task(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"unknown task: {task_id}")


class AgentProfile(Contract):
    kind: Literal["agent", "scripted"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    instructions_digest: str = Field(min_length=1)
    settings: dict[str, JsonValue]


class PolicyProfile(Contract):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parameters: dict[str, JsonValue]


class Usage(Contract):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)

    @model_serializer(mode="wrap")
    def omit_unknown_details(self, handler):
        data = handler(self)
        for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def cache_is_subset(self):
        if self.cached_input_tokens is not None and self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens must be a subset of input tokens")
        if self.cache_write_input_tokens is not None and self.cache_write_input_tokens + (self.cached_input_tokens or 0) > self.input_tokens:
            raise ValueError("cache read/write buckets must be disjoint subsets of input tokens")
        if self.reasoning_output_tokens is not None and self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning tokens must be a subset of output tokens")
        return self


class UsageObservation(Contract):
    id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    outcome: Literal["completed", "failed", "cancelled", "unknown"]
    usage: Usage | None
    raw_usage: JsonValue = None


class UsageCapture(Contract):
    scope: Literal["provider_calls", "cli_aggregate", "legacy", "unknown"]
    source: str = Field(min_length=1)
    observations: list[UsageObservation] = Field(default_factory=list)
    expected_calls: int | None = Field(default=None, ge=0)
    internal_calls_complete: bool = False
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_observations(self):
        ids = [entry.id for entry in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate usage observation IDs")
        return self

    def observed_usage(self) -> Usage | None:
        # Multiple CLI terminal counters may be duplicate or cumulative. Keep
        # each observation but do not invent a total from an ambiguous stream.
        if self.scope in {"cli_aggregate", "legacy"} and len(self.observations) != 1:
            return None
        if self.scope == "unknown":
            return None
        values = [entry.usage for entry in self.observations if entry.usage is not None]
        if not values:
            if self.scope == "provider_calls" and self.expected_calls == 0 and self.internal_calls_complete and not self.observations:
                return Usage(input_tokens=0, output_tokens=0, cached_input_tokens=0,
                             cache_write_input_tokens=0, reasoning_output_tokens=0)
            return None
        details = {key: sum(getattr(value, key) for value in values)
                   if all(getattr(value, key) is not None for value in values) else None
                   for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens")}
        return Usage(input_tokens=sum(value.input_tokens for value in values),
                     output_tokens=sum(value.output_tokens for value in values),
                     **details)

    @property
    def provider_complete(self) -> bool:
        return (self.scope == "provider_calls" and self.internal_calls_complete and not self.issues
                and self.expected_calls == len(self.observations)
                and all(entry.usage is not None and entry.usage.cached_input_tokens is not None for entry in self.observations))


class ToolEvent(Contract):
    sequence: int = Field(ge=0)
    kind: Literal["tool"] = "tool"
    name: str = Field(min_length=1)
    request: JsonValue
    response: JsonValue
    elapsed_ms: float = Field(ge=0)


class TransportProvenance(Contract):
    name: str
    version: str
    attempt_id: str
    request_digest: str
    requested_model: str
    client_version: str
    response_source: str
    response_item_id: str | None = None
    artifact_path: str
    stdout_sha256: str
    stderr_sha256: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)
    output_truncated: bool
    exit_code: int | None
    failure_reason: str | None = None


class ModelEvent(Contract):
    sequence: int = Field(ge=0)
    kind: Literal["model"] = "model"
    request: JsonValue
    response: JsonValue
    usage: Usage | None = None
    response_observed: bool = True
    transport: TransportProvenance | None = None
    usage_capture: UsageCapture | None = None


class ControlEvent(Contract):
    sequence: int = Field(ge=0)
    kind: Literal["control"] = "control"
    decision: Literal["blocked"] = "blocked"
    reason: str = Field(min_length=1)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Review(Contract):
    verdict: Literal["accepted", "rejected", "unreviewed"] = "unreviewed"
    reviewer: str | None = None
    criteria: dict[str, Literal["met", "not_met", "unknown"]] = Field(default_factory=dict)
    acceptance_checks: dict[str, Literal["passed", "failed", "not_run"]] = Field(default_factory=dict)
    unsupported_high_impact_claims: list[str] = Field(default_factory=list)
    validation: dict[str, JsonValue] | None = None

    @model_serializer(mode="wrap")
    def omit_legacy_validation(self, handler):
        data = handler(self)
        if data.get("validation") is None:
            data.pop("validation",None)
        return data

    @model_validator(mode="after")
    def requires_reviewer(self):
        if self.verdict != "unreviewed" and not (self.reviewer or "").strip():
            raise ValueError("a scored review must name its reviewer")
        return self


class CaptureInfo(Contract):
    request_capture: Literal["complete", "visible_only"]
    provider_usage_available: bool
    model_version_pinned: bool
    host_tools_enforced: bool
    provider_token_budget_enforced: bool
    limitations: list[str] = Field(default_factory=list)


class Trace(Contract):
    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    suite_digest: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    repository_revision: str = Field(min_length=1)
    replicate: str = Field(min_length=1)
    cache_state: Literal["cold", "warm"]
    agent: AgentProfile
    policy: PolicyProfile
    events: list[Annotated[ToolEvent | ModelEvent | ControlEvent, Field(discriminator="kind")]]
    elapsed_ms: float = Field(ge=0)
    outcome: Literal["completed", "failed", "limit_exceeded", "aborted"]
    review: Review = Field(default_factory=Review)
    capture: CaptureInfo | None = None
    termination_reason: str | None = None
    coordination: dict[str, JsonValue] | None = None

    @model_serializer(mode="wrap")
    def omit_legacy_coordination(self, handler):
        data = handler(self)
        if data["coordination"] is None:
            data.pop("coordination")
        return data

    @model_validator(mode="after")
    def ordered_events(self):
        if [event.sequence for event in self.events] != list(range(len(self.events))):
            raise ValueError("event sequences must be contiguous and ordered from zero")
        return self

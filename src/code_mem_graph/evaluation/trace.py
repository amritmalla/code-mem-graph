from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Literal

from pydantic import JsonValue

from .models import AgentProfile, ModelEvent, PolicyProfile, Review, Suite, ToolEvent, Trace, Usage, UsageCapture


class TraceRecorder:
    """Record caller-supplied interactions without invoking tools or models."""

    def __init__(
        self, suite: Suite, task_id: str, *, run_id: str, agent: AgentProfile,
        policy: PolicyProfile, replicate: str, cache_state: Literal["cold", "warm"],
        repository_revision: str, clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        task = suite.task(task_id)
        expected = next(repo.revision for repo in suite.repositories if repo.id == task.repository_id)
        if repository_revision != expected:
            raise ValueError("repository revision does not match the evaluation suite")
        self._metadata = dict(
            schema_version=1, run_id=run_id, suite_digest=suite.digest, task_id=task_id,
            repository_revision=repository_revision, agent=agent.model_copy(deep=True),
            policy=policy.model_copy(deep=True),
            replicate=replicate, cache_state=cache_state,
        )
        self._events: list[ToolEvent | ModelEvent] = []
        self._protocol = suite.protocol.model_copy(deep=True)
        self._clock = clock
        self._started = clock()
        self._finished = False

    def _check_open(self) -> None:
        if self._finished:
            raise ValueError("trace is already finished")

    def record_tool(self, name: str, request: JsonValue, response: JsonValue, *, elapsed_ms: float) -> None:
        self._check_open()
        event = ToolEvent(sequence=len(self._events), name=name, request=request, response=response, elapsed_ms=elapsed_ms)
        self._events.append(event.model_copy(deep=True))

    def record_model_turn(self, request: JsonValue, response: JsonValue, usage: Usage | None = None,
                          *, usage_capture: UsageCapture | None = None) -> None:
        self._check_open()
        event = ModelEvent(sequence=len(self._events), request=request, response=response, usage=usage, usage_capture=usage_capture)
        self._events.append(event.model_copy(deep=True))

    def finish(
        self, outcome: Literal["completed", "failed", "limit_exceeded", "aborted"],
        *, review: Review | None = None,
    ) -> Trace:
        self._check_open()
        elapsed_ms = (self._clock() - self._started) * 1000
        tools = sum(isinstance(event, ToolEvent) for event in self._events)
        turns = len(self._events) - tools
        if (elapsed_ms > self._protocol.max_elapsed_ms or tools > self._protocol.max_tool_calls
                or turns > self._protocol.max_model_turns):
            outcome = "limit_exceeded"
        result = Trace(**self._metadata, events=self._events, elapsed_ms=elapsed_ms, outcome=outcome,
                       review=review or Review())
        self._finished = True
        return result.model_copy(deep=True)


def write_trace(trace: Trace, path: Path) -> None:
    """Write one trace; refuse accidental overwrite of another run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(trace.model_dump_json(indent=2) + "\n")

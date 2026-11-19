"""Local dispatch reservations; unsupported provider guarantees fail closed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .accounting import usage_diagnostics
from .models import Contract, UsageCapture


class ResourceLimits(Contract):
    max_dispatches: int | None = Field(default=None, gt=0)
    max_stdin_bytes: int | None = Field(default=None, gt=0)
    max_total_stdin_bytes: int | None = Field(default=None, gt=0)
    max_total_retained_stdout_bytes: int | None = Field(default=None, gt=0)
    max_total_retained_stderr_bytes: int | None = Field(default=None, gt=0)
    max_observed_cli_tokens: int | None = Field(default=None, gt=0)


class DispatchBlocked(Exception):
    def __init__(self, reason: str, details: dict, *, outcome: Literal["aborted", "limit_exceeded"] = "limit_exceeded"):
        super().__init__(reason)
        self.reason, self.details, self.outcome = reason, details, outcome


def codex_capabilities() -> dict:
    return {"strict_calibration_supported": False,
            "supported_local_controls": ["stdin byte checks", "dispatch count", "retained stdout/stderr quotas",
                                        "timeouts and cancellation", "frozen virtual-tool allowlist"],
            "reactive_controls": ["terminate on detected host-tool/error events", "stop after observed CLI token threshold"],
            "unsupported_guarantees": ["complete host-tool removal", "provider output-token cap", "provider total-token/spend cap"],
            "notes": ["Named feature disabling is requested, not proof of complete tool isolation.",
                      "Observed CLI token thresholds can be exceeded by the current call; they are not provider reservations."]}


class ControlLedger:
    def __init__(self, path: Path, limits: ResourceLimits, mode: str):
        if mode not in {"functional", "strict"}:
            raise ValueError("execution mode must be functional or strict")
        self.path, self.limits, self.mode = path, limits.model_copy(deep=True), mode
        self.dispatches = self.stdin_bytes = self.stdout_bytes = self.stderr_bytes = self.cli_tokens = 0
        self.usage_unknown = False
        self.pending: dict | None = None
        self.activated = False

    def snapshot(self) -> dict:
        return {"dispatches": self.dispatches, "reserved_stdin_bytes": self.stdin_bytes,
                "retained_stdout_bytes": self.stdout_bytes, "retained_stderr_bytes": self.stderr_bytes,
                "observed_cli_tokens": self.cli_tokens, "cli_usage_unknown": self.usage_unknown}

    def audit(self, decision: str, **details):
        try:
            with self.path.open("ab") as stream:
                stream.write((json.dumps({"decision": decision, **details}, sort_keys=True) + "\n").encode("utf-8"))
        except OSError as exc:
            raise DispatchBlocked("control_audit_io_failure", {"stage": decision}, outcome="aborted") from exc

    def reserve(self, request_digest: str, stdin_bytes: int, stdout_cap: int, stderr_cap: int) -> dict:
        if self.pending is not None:
            if not self.activated and self.pending["request_digest"] == request_digest:
                return self.pending.copy()
            raise DispatchBlocked("unsettled_dispatch_reservation", self.snapshot(), outcome="aborted")
        reason = None
        details = {"mode": self.mode, "request_digest": request_digest, "stdin_bytes": stdin_bytes,
                   "limits": self.limits.model_dump(mode="json"), **self.snapshot()}
        if self.mode == "strict":
            reason = "strict_profile_unsupported"
            details["unsupported_guarantees"] = codex_capabilities()["unsupported_guarantees"]
        elif self.limits.max_dispatches is not None and self.dispatches >= self.limits.max_dispatches:
            reason = "transport_dispatch_limit"
        elif self.limits.max_stdin_bytes is not None and stdin_bytes > self.limits.max_stdin_bytes:
            reason = "stdin_byte_limit"
        elif self.limits.max_total_stdin_bytes is not None and self.stdin_bytes + stdin_bytes > self.limits.max_total_stdin_bytes:
            reason = "total_stdin_byte_reservation"
        elif self.limits.max_observed_cli_tokens is not None:
            if self.usage_unknown:
                reason = "cli_usage_unknown_with_threshold"
            elif self.cli_tokens >= self.limits.max_observed_cli_tokens:
                reason = "observed_cli_token_threshold"
        if self.limits.max_total_retained_stdout_bytes is not None:
            stdout_cap = min(stdout_cap, self.limits.max_total_retained_stdout_bytes - self.stdout_bytes)
        if self.limits.max_total_retained_stderr_bytes is not None:
            stderr_cap = min(stderr_cap, self.limits.max_total_retained_stderr_bytes - self.stderr_bytes)
        if reason is None and (stdout_cap <= 0 or stderr_cap <= 0):
            reason = "retained_stream_quota_exhausted"
        if reason:
            self.audit("blocked", reason=reason, **details)
            raise DispatchBlocked(reason, details, outcome="aborted" if reason == "strict_profile_unsupported" else "limit_exceeded")
        permit = {"request_digest": request_digest, "stdin_bytes": stdin_bytes,
                  "stdout_cap": stdout_cap, "stderr_cap": stderr_cap}
        self.audit("reserved", permit=permit, state=self.snapshot())
        self.pending = permit
        return permit.copy()

    def activate(self, request_digest: str):
        if self.activated or self.pending is None or self.pending["request_digest"] != request_digest:
            raise DispatchBlocked("missing_dispatch_reservation", self.snapshot(), outcome="aborted")
        self.audit("activated", permit=self.pending, state=self.snapshot())
        self.dispatches += 1
        self.activated = True
        # Conservative charge even if child launch subsequently fails. This
        # reserves local request capacity, not provider-billed input tokens.
        self.stdin_bytes += self.pending["stdin_bytes"]

    def settle(self, stdout_bytes: int, stderr_bytes: int, usage_capture: UsageCapture) -> bool:
        self.stdout_bytes += stdout_bytes
        self.stderr_bytes += stderr_bytes
        diagnostic = usage_diagnostics(usage_capture.observed_usage(), usage_capture)
        if diagnostic["recorded_scope_complete"]:
            usage = usage_capture.observed_usage()
            self.cli_tokens += usage.input_tokens + usage.output_tokens
        else:
            self.usage_unknown = True
        self.audit("settled", permit=self.pending, state=self.snapshot(), usage_scope=diagnostic["scope"])
        self.pending = None
        self.activated = False
        return self.limits.max_observed_cli_tokens is not None and self.cli_tokens > self.limits.max_observed_cli_tokens

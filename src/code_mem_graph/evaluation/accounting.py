"""Evidence-scoped usage and separate, optionally priced cost categories."""
from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext
import json
import re
from typing import Literal

from pydantic import Field, model_validator

from .models import Contract, ModelEvent, Trace, Usage, UsageCapture, UsageObservation

CostCategory = Literal["task", "evaluation", "review", "build", "generation"]
CATEGORIES = ("task", "evaluation", "review", "build", "generation")


def reject_json_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def normalize_codex_counter(raw) -> tuple[Usage | None, list[str]]:
    """Normalize documented inclusive totals; unknown detail buckets stay unknown."""
    if not isinstance(raw, dict):
        return None, ["missing_or_invalid_usage"]
    try:
        value = Usage(input_tokens=raw["input_tokens"], output_tokens=raw["output_tokens"])
    except (KeyError, ValueError):
        return None, ["missing_or_invalid_usage"]
    issues, updates = [], {}
    for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens"):
        if key not in raw:
            continue
        try:
            Usage.model_validate({**value.model_dump(), **updates, key: raw[key]})
        except ValueError:
            issues.append("invalid_" + key)
        else:
            updates[key] = raw[key]
    return Usage.model_validate({**value.model_dump(), **updates}), issues


def capture_codex_usage(raw: bytes, *, truncated: bool = False) -> UsageCapture:
    observations, issues = [], ["transcript_truncated"] if truncated else []
    # Decode lines independently so a corrupt later event cannot erase earlier
    # counters. Multiple terminal counters remain ambiguous, not silently summed.
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line.decode("utf-8"), parse_constant=reject_json_constant)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
        except (UnicodeError, ValueError):
            issues.append("invalid_event_stream")
            continue
        kind = event.get("type")
        if kind in {"turn.completed", "turn.failed"}:
            usage, problems = normalize_codex_counter(event.get("usage"))
            issues.extend(problems)
            observations.append(UsageObservation(id=f"terminal-{number}", reference=f"stdout.jsonl:line:{number}",
                outcome="completed" if kind == "turn.completed" else "failed", usage=usage, raw_usage=event.get("usage")))
        if kind == "error":
            issues.append("cli_error_or_retry_seen")
    if len(observations) > 1:
        issues.append("ambiguous_cli_usage")
    if not observations:
        issues.append("no_terminal_usage")
    return UsageCapture(scope="cli_aggregate", source="Codex JSONL terminal usage", observations=observations,
                        internal_calls_complete=False, issues=sorted(set(issues)))


def usage_diagnostics(usage: Usage | None, capture: UsageCapture | None) -> dict:
    if capture is None:
        capture = UsageCapture(scope="legacy", source="Undeclared historical usage",
            observations=[UsageObservation(id="legacy", reference="model_event.usage", outcome="unknown", usage=usage)])
    observed = capture.observed_usage()
    issues = list(capture.issues)
    if usage is not None:
        mismatch = observed is None or any(getattr(usage, key) != getattr(observed, key)
                                          for key in ("input_tokens", "output_tokens"))
        mismatch |= observed is not None and any(
            getattr(usage, key) is not None and getattr(observed, key) is not None and getattr(usage, key) != getattr(observed, key)
            for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens"))
        if mismatch:
            issues.append("usage_reconciliation_mismatch")
            observed = None
    scope_known = (capture.scope in {"cli_aggregate", "legacy"} and len(capture.observations) == 1
                   or capture.scope == "provider_calls" and capture.expected_calls == len(capture.observations))
    recorded_complete = bool(observed is not None and scope_known and not issues
                             and all(entry.usage is not None for entry in capture.observations))
    complete = capture.provider_complete and not issues
    blockers = []
    if capture.scope != "provider_calls":
        blockers.append("provider_call_usage_unavailable")
    if not complete:
        blockers.append("provider_usage_incomplete")
    if issues:
        blockers.append("usage_reconciliation_issues")
    return {"scope": capture.scope, "source": capture.source,
            "observed_usage": observed.model_dump(mode="json") if observed is not None else None,
            "recorded_scope_complete": recorded_complete, "provider_complete": complete,
            "expected_calls": capture.expected_calls, "observations": len(capture.observations),
            "observation_records": [entry.model_dump(mode="json") for entry in capture.observations],
            "unknown_observations": sum(entry.usage is None for entry in capture.observations),
            "internal_calls_complete": capture.internal_calls_complete,
            "issues": sorted(set(issues)), "blockers": blockers}


def validate_trace_usage(traces: list[Trace]) -> None:
    calls, attempts = set(), set()
    for trace in traces:
        for event in trace.events:
            if not isinstance(event, ModelEvent):
                continue
            if event.transport is not None:
                key = (event.transport.name, event.transport.attempt_id)
                if key in attempts:
                    raise ValueError("transport attempt appears in more than one model event")
                attempts.add(key)
            if event.usage_capture and event.usage_capture.scope == "provider_calls":
                for observation in event.usage_capture.observations:
                    key = (trace.agent.provider, observation.id)
                    if key in calls:
                        raise ValueError("provider call appears in more than one model event")
                    calls.add(key)


class PriceCard(Contract):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    effective_date: date
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    input_per_million: str
    cached_input_per_million: str
    output_per_million: str
    cache_write_policy: Literal["standard_input", "separate_rate"]
    cache_write_per_million: str | None = None

    @model_validator(mode="after")
    def exact_rates(self):
        for key in ("input_per_million", "cached_input_per_million", "output_per_million", "cache_write_per_million"):
            value = getattr(self, key)
            if value is not None and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                raise ValueError("rates must be nonnegative exact decimal strings")
        if (self.cache_write_policy == "separate_rate") != (self.cache_write_per_million is not None):
            raise ValueError("separate cache-write pricing requires exactly one explicit write rate")
        return self


class CostRecord(Contract):
    id: str = Field(min_length=1)
    category: CostCategory
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    outcome: Literal["completed", "failed", "aborted", "limit_exceeded", "unknown"]
    usage: Usage | None = None
    usage_capture: UsageCapture | None = None
    billing_mode: Literal["api", "subscription", "unknown"] = "unknown"
    price_card_id: str | None = None


class CostLedger(Contract):
    schema_version: Literal[1] = 1
    records: list[CostRecord] = Field(default_factory=list)
    prices: list[PriceCard] = Field(default_factory=list)
    coverage: dict[CostCategory, Literal["complete", "partial", "not_recorded", "not_applicable"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_records(self):
        ids = [record.id for record in self.records]
        price_ids = [price.id for price in self.prices]
        if len(ids) != len(set(ids)) or len(price_ids) != len(set(price_ids)):
            raise ValueError("duplicate cost record or price-card IDs")
        if len({price.currency for price in self.prices}) > 1:
            raise ValueError("mixed currencies require separate ledgers; no exchange rate is inferred")
        prices, calls = {price.id: price for price in self.prices}, set()
        for record in self.records:
            if self.coverage.get(record.category) in {"not_recorded", "not_applicable"}:
                raise ValueError("category with records cannot be declared unrecorded/not applicable")
            if record.price_card_id is not None:
                price = prices.get(record.price_card_id)
                if price is None or (price.provider, price.model) != (record.provider, record.model):
                    raise ValueError("price card must match the exact record provider/model")
            if record.usage_capture and record.usage_capture.scope == "provider_calls":
                for observation in record.usage_capture.observations:
                    key = (record.provider, observation.id)
                    if key in calls:
                        raise ValueError("provider call appears in more than one cost record/category")
                    calls.add(key)
        return self


def _price(usage: Usage | None, record: CostRecord, price: PriceCard | None):
    if record.billing_mode != "api":
        return None, "billing_mode_not_token_priced"
    if usage is None:
        return None, "usage_unknown_or_inconsistent"
    if price is None:
        return None, "price_card_missing"
    if usage.cached_input_tokens is None:
        return None, "cache_read_usage_unknown"
    writes = usage.cache_write_input_tokens
    if price.cache_write_policy == "separate_rate" and writes is None:
        return None, "cache_write_usage_unknown"
    rates = [price.input_per_million, price.cached_input_per_million, price.output_per_million,
             price.cache_write_per_million or "0"]
    with localcontext() as context:
        context.prec = 30 + len(str(usage.input_tokens + usage.output_tokens)) + sum(map(len, rates))
        ordinary = usage.input_tokens - usage.cached_input_tokens
        total = Decimal(ordinary) * Decimal(price.input_per_million)
        if price.cache_write_policy == "separate_rate":
            total -= Decimal(writes) * Decimal(price.input_per_million)
            total += Decimal(writes) * Decimal(price.cache_write_per_million)
        total += Decimal(usage.cached_input_tokens) * Decimal(price.cached_input_per_million)
        # Reasoning is already included in output; neither cache nor reasoning
        # detail buckets are added to inclusive model-token totals a second time.
        total += Decimal(usage.output_tokens) * Decimal(price.output_per_million)
        return format(total / Decimal(1000000), "f"), None


def ledger_report(ledger: CostLedger) -> dict:
    prices = {price.id: price for price in ledger.prices}
    records = []
    for record in ledger.records:
        usage = usage_diagnostics(record.usage, record.usage_capture)
        normalized = Usage.model_validate(usage["observed_usage"]) if usage["observed_usage"] is not None else None
        amount, reason = _price(normalized, record, prices.get(record.price_card_id))
        records.append({**record.model_dump(mode="json", exclude={"usage", "usage_capture"}), "usage": usage,
                        "observed_token_rate_estimate": amount, "pricing_unknown_reason": reason,
                        "estimated_total_cost": amount if usage["provider_complete"] else None})
    categories = {}
    for category in CATEGORIES:
        entries = [record for record in records if record["category"] == category]
        declared = ledger.coverage.get(category, "partial" if entries else "not_recorded")
        complete = declared in {"complete", "not_applicable"} and all(entry["usage"]["provider_complete"] for entry in entries)
        known = [entry["usage"]["observed_usage"] for entry in entries if entry["usage"]["observed_usage"] is not None]
        input_tokens = sum(value["input_tokens"] for value in known) if known else 0 if complete else None
        output_tokens = sum(value["output_tokens"] for value in known) if known else 0 if complete else None
        amounts = [Decimal(entry["observed_token_rate_estimate"]) for entry in entries if entry["observed_token_rate_estimate"] is not None]
        # Decimal addition also needs sufficient precision for user-supplied rates.
        with localcontext() as context:
            context.prec = 40 + sum(len(str(value)) for value in amounts)
            subtotal = format(sum(amounts), "f") if amounts else "0" if complete and not entries and ledger.prices else None
        categories[category] = {"declared_coverage": declared,
            "coverage": "complete" if complete else "partial" if entries else declared,
            "records": len(entries), "known_usage_records": len(known),
            "provider_incomplete_records": sum(not entry["usage"]["provider_complete"] for entry in entries),
            "observed_input_tokens": input_tokens, "observed_output_tokens": output_tokens,
            **{f"observed_{key}": sum(value[key] for value in known if key in value)
               if any(key in value for value in known) else None
               for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens")},
            "detail_coverage": {key: {"known_records": sum(key in value for value in known),
                                       "unknown_records": len(entries) - sum(key in value for value in known)}
                                for key in ("cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens")},
            "observed_model_tokens": input_tokens + output_tokens if input_tokens is not None else None,
            "complete_provider_tokens": input_tokens + output_tokens if complete else None,
            "observed_token_rate_subtotal": subtotal,
            "estimated_total_cost": subtotal if complete and all(entry["estimated_total_cost"] is not None for entry in entries) else None}
    return {"schema_version": 1, "currency": ledger.prices[0].currency if ledger.prices else None,
            "pricing_basis": "caller-supplied versioned token rates; estimates, not invoices",
            "cost_scope": "model tokens only; other fees, compute, subscriptions and human effort excluded",
            "prices": [price.model_dump(mode="json") for price in ledger.prices], "categories": categories, "records": records,
            "complete_estimated_cost": _complete_cost(categories) if ledger.prices else None,
            "notes": ["Unknown usage/prices/categories are not zero.",
                      "CLI aggregates do not establish complete provider-call coverage.",
                      "Failed/retried/rejected attempts remain spending; duplicate call IDs are rejected.",
                      "Subscription usage is not automatically priced as API billing.",
                      "No amortization, token-saving claim, or automatic sizing policy is selected."]}


def _complete_cost(categories):
    values = [category["estimated_total_cost"] for category in categories.values()]
    if any(value is None for value in values):
        return None
    with localcontext() as context:
        context.prec = 40 + sum(map(len, values))
        return format(sum(Decimal(value) for value in values), "f")


def ledger_from_traces(traces: list[Trace], *, category: CostCategory, billing_mode="unknown", price: PriceCard | None = None) -> CostLedger:
    validate_trace_usage(traces)
    records = []
    for trace in traces:
        for event in trace.events:
            if isinstance(event, ModelEvent):
                identity = f"{event.transport.name}/{event.transport.attempt_id}" if event.transport else f"{trace.run_id}/model/{event.sequence}"
                records.append(CostRecord(id=identity, category=category, provider=trace.agent.provider, model=trace.agent.model,
                    reference=f"trace:{trace.run_id}/events/{event.sequence}", outcome=trace.outcome, usage=event.usage,
                    usage_capture=event.usage_capture, billing_mode=billing_mode, price_card_id=price.id if price else None))
    return CostLedger(records=records, prices=[price] if price else [])


def merge_ledgers(ledgers: list[CostLedger]) -> CostLedger:
    # A partial collection cannot silently claim that the caller imported every
    # relevant operation. Complete coverage must be reviewed on the final ledger.
    prices = {}
    for ledger in ledgers:
        for price in ledger.prices:
            if price.id in prices and price != prices[price.id]:
                raise ValueError("conflicting versions/content for a price-card ID")
            prices[price.id] = price
    return CostLedger(records=[record for ledger in ledgers for record in ledger.records], prices=list(prices.values()))

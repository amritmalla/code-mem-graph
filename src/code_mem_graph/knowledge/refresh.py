from __future__ import annotations

from .models import KnowledgeLayer, KnowledgeMap, ReviewRecord


def refresh_generated(current: KnowledgeMap, incoming: KnowledgeLayer | None, *, error: str | None = None) -> KnowledgeMap:
    """Validate a new generated snapshot atomically in memory; preserve the curated layer."""
    current = KnowledgeMap.model_validate(current.model_dump(mode="json"))
    data = current.model_dump(mode="json")
    if incoming is None:
        if error is None:
            raise ValueError("a failed refresh requires a reason")
        data["refresh"] = {"status": "failed", "error": error}
        return KnowledgeMap.model_validate(data)
    if error is not None:
        raise ValueError("a successful refresh cannot carry an error")
    generated = incoming.model_dump(mode="json")
    # Review authority never comes from a generator. Reuse only a matching prior review.
    prior_items = {item.id: item for item in current.generated.items}
    for item in generated["items"]:
        item["review"] = ReviewRecord().model_dump(mode="json")
    candidate_items = KnowledgeLayer.model_validate(generated).items
    for item, raw in zip(candidate_items, generated["items"]):
        prior = prior_items.get(item.id)
        if prior is not None and prior.content_digest == item.content_digest:
            raw["review"] = prior.review.model_dump(mode="json")
    prior_conflicts = {entry.id: entry for entry in current.generated.conflicts}
    for conflict in generated["conflicts"]:
        prior = prior_conflicts.get(conflict["id"])
        conflict.update(status="unresolved", resolution=None, resolved_by=None, resolved_versions={})
        if prior is not None and prior.status == "resolved" and set(prior.item_ids) == set(conflict["item_ids"]):
            for field in ("status", "resolution", "resolved_by", "resolved_versions"):
                conflict[field] = prior.model_dump(mode="json")[field]
    for field in ("items", "components", "gaps", "conflicts"):
        ids = {entry["id"] for entry in generated[field]}
        for entry in getattr(current.generated, field):
            if entry.id not in ids:
                retained = entry.model_dump(mode="json")
                if field in {"items", "components"}:
                    retained["lifecycle"] = "retired"
                # Missing gaps/conflicts stay explicit until an incoming entry resolves them.
                generated[field].append(retained)
    data["generated"] = generated
    data["refresh"] = {"status": "ready", "error": None}
    return KnowledgeMap.model_validate(data)

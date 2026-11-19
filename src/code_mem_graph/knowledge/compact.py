"""Lossless shared evidence/assessment representation for selected bundle content."""
from __future__ import annotations

from copy import deepcopy

from code_mem_graph.serialization import fingerprint

DEFAULTS = {"author": None, "facets": [], "component_ids": [], "dependencies": [], "critical": False, "lifecycle": "active"}
ASSESSMENT = ("freshness", "review_required", "reasons", "conflict_ids")


def _content_digest(payload):
    return fingerprint({key:value for key,value in payload.items() if key != "size"})


def compact_bundle(payload: dict) -> dict:
    """Encode without dropping statements, provenance, evidence or safety signals."""
    if payload.get("schema_version") != 1 or "representation" in payload:
        raise ValueError("expected a detailed version-1 bundle")
    result = deepcopy(payload)
    tables = {name:{} for name in ("policies", "sources", "citations", "assessments", "reviews")}
    identities = {name:{} for name in tables}

    def intern(table, value, prefix):
        identity = fingerprint(value)
        if identity not in identities[table]:
            key = prefix + str(len(tables[table]))
            identities[table][identity] = key
            tables[table][key] = value
        return identities[table][identity]

    def citation(evidence):
        source = {key:value for key,value in evidence.items() if key not in {"start_line", "end_line", "policy_digest"}}
        source["policy"] = intern("policies", evidence["policy_digest"], "p")
        reference = {"source":intern("sources", source, "s")}
        if "start_line" in evidence:
            reference.update(start_line=evidence["start_line"], end_line=evidence["end_line"])
        return intern("citations", reference, "c")

    for group in ("items", "cautions"):
        for item in result[group]:
            if not DEFAULTS.keys() <= item.keys():
                raise ValueError("incomplete detailed item metadata")
            item["citations"] = [citation(ref) for ref in item.pop("evidence")]
            item["assessment"] = intern("assessments", {key:item.pop(key) for key in ASSESSMENT}, "a")
            item["review_id"] = intern("reviews", item.pop("review"), "r")
            for key,value in DEFAULTS.items():
                if item[key] == value:
                    del item[key]
    for candidate in result.get("source_candidates", []):
        if "source" in candidate:
            candidate["citation"] = citation(candidate.pop("source"))
    result["schema_version"] = 2
    result["representation"] = {
        "name":"shared-evidence", "version":1, "expanded_schema_version":1,
        "expanded_content_digest":_content_digest(payload), "item_defaults":deepcopy(DEFAULTS),
        "references":"Resolve item citations through citations -> sources -> policies; assessment/review_id through their tables. "
                     "Defaults apply to omitted item fields. Statements are verbatim. knowledge_item(item_id) inspects full records; compare map/content digests.",
    }
    result.update(tables)
    return result


def expand_bundle(payload: dict) -> dict:
    """Reconstruct detailed semantic content, retaining the original wire size metadata.

    This is an offline diagnostic, not another bounded wire response. The size
    belongs to the encoded input, not to this expanded Python object.
    """
    if payload.get("schema_version") == 1 and "representation" not in payload:
        return deepcopy(payload)
    try:
        result = deepcopy(payload)
        representation = result.pop("representation")
        if (result["schema_version"] != 2 or representation["name"] != "shared-evidence" or representation["version"] != 1
                or representation["expanded_schema_version"] != 1 or representation["item_defaults"] != DEFAULTS):
            raise ValueError("unsupported compact representation")
        tables = {key:result.pop(key) for key in ("policies", "sources", "citations", "assessments", "reviews")}

        def evidence(key):
            reference = deepcopy(tables["citations"][key])
            source = deepcopy(tables["sources"][reference.pop("source")])
            source["policy_digest"] = tables["policies"][source.pop("policy")]
            return {**source, **reference}

        for group in ("items", "cautions"):
            for item in result[group]:
                for key,value in DEFAULTS.items():
                    item.setdefault(key,deepcopy(value))
                item["evidence"] = [evidence(key) for key in item.pop("citations")]
                item.update(deepcopy(tables["assessments"][item.pop("assessment")]))
                item["review"] = deepcopy(tables["reviews"][item.pop("review_id")])
        for candidate in result.get("source_candidates", []):
            if "citation" in candidate:
                candidate["source"] = evidence(candidate.pop("citation"))
        result["schema_version"] = 1
        if _content_digest(result) != representation["expanded_content_digest"]:
            raise ValueError("expanded bundle content digest mismatch")
        if _content_digest(compact_bundle(result)) != _content_digest(payload):
            raise ValueError("noncanonical compact tables or representation metadata")
        return result
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid compact bundle references") from exc


def is_inventory(item) -> bool:
    return (item.origin == "generated" and item.generator is not None and item.generator.name == "repository-facts"
            and (item.id == "project/source-inventory" or item.id.endswith("/inventory")))


def is_navigation_fact(item) -> bool:
    return (is_inventory(item) or (item.origin == "generated" and item.generator is not None
            and item.generator.name == "repository-facts" and item.id.startswith("manifest/")
            and ("/script/" in item.id or item.id.endswith("/pytest"))))

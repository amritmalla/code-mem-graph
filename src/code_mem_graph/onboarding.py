"""Repository onboarding without changing client settings or running source commands."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

from .knowledge.build import build_repository, load_repository_map
from .knowledge.sources import EvidenceUnavailable, _safe_path
from .knowledge.workflow import status_report


def agent_configuration(root: Path) -> dict:
    """Use this installation's interpreter and an absolute, pinned repository root."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    return {
        "mcpServers": {
            "code-mem-graph": {
                "command": sys.executable,
                "args": ["-m", "code_mem_graph.cli", "serve", "--root", str(root)],
            }
        }
    }


def inspect_repository(root: Path) -> dict:
    """Check stored knowledge against current sources; do not rebuild or write files."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    result = {
        "repository_root": str(root),
        "status": "not_initialized",
        "ready_for_reads": False,
        "issues": [],
        "repository_id": None,
        "active_claims": 0,
        "source_status": None,
        "review_queue": [],
        "open_gaps": [],
        "unresolved_conflicts": [],
        "reading_view": ".project-knowledge/README.md",
        "reading_view_status": "missing",
        "mcp_config": agent_configuration(root),
        "client_connection_verified": False,
        "model_calls": 0,
        "limits": [
            "Readiness checks stored evidence, not architectural completeness or task correctness.",
            "Tokenizer availability and the MCP client connection are not tested.",
        ],
    }
    try:
        document = load_repository_map(root)
    except EvidenceUnavailable as exc:
        if exc.reason == "missing":
            result["issues"] = ["knowledge_map_missing"]
        else:
            result["status"] = "needs_attention"
            result["issues"] = ["knowledge_map_unreadable"]
        return result
    except (OSError, ValueError, yaml.YAMLError):
        result["status"] = "needs_attention"
        result["issues"] = ["knowledge_map_invalid_or_unreadable"]
        return result
    report = status_report(root, document)
    result.update(
        repository_id=document.repository_id,
        active_claims=report["active_claims"],
        source_status=report["source_status"],
        review_queue=report["review_queue"],
        open_gaps=[
            entry["id"]
            for entry in report["gaps"]
            if entry["effective_status"] == "open"
        ],
        unresolved_conflicts=[
            entry["id"]
            for entry in report["conflicts"]
            if entry["effective_status"] == "unresolved"
        ],
    )
    if report["refresh"]["status"] == "failed":
        result["issues"].append("last_refresh_failed")
    if report["source_status"]["stale"] or report["source_status"]["unknown"]:
        result["issues"].append("source_evidence_needs_refresh")
    if report["review_queue"]:
        result["issues"].append("claims_need_review")
    if result["unresolved_conflicts"]:
        result["issues"].append("unresolved_conflicts")
    try:
        brief = _safe_path(root, result["reading_view"]).read_text(encoding="utf-8")
        result["reading_view_status"] = (
            "current"
            if f"Map content digest: `{report['map_digest']}`." in brief
            else "outdated"
        )
    except (OSError, ValueError, UnicodeError):
        result["reading_view_status"] = "missing_or_unreadable"
    if result["reading_view_status"] != "current":
        result["issues"].append("reading_view_needs_rebuild")
    result["ready_for_reads"] = not result["issues"]
    result["status"] = (
        "ready_for_reads" if result["ready_for_reads"] else "needs_attention"
    )
    return result


def initialize_repository(root: Path, *, repository_id: str | None = None) -> dict:
    """Build facts using the existing preservation/locking rules, then explain setup."""
    build_repository(root, repository_id=repository_id)
    return inspect_repository(root)

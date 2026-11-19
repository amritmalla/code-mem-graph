from __future__ import annotations

from pathlib import Path

import yaml

from .models import KnowledgeMap


def load_map(path: Path) -> KnowledgeMap:
    """Read versionable YAML or JSON, without executing repository content."""
    return KnowledgeMap.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

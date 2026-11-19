"""Stable JSON identities shared by knowledge and evaluation contracts."""
from __future__ import annotations

import hashlib
import json


def serialize(value: object) -> str:
    """Canonical complete JSON payload, including metadata and citations."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(serialize(value).encode("utf-8")).hexdigest()

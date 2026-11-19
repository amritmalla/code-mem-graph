from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from pydantic import Field

from .build import load_repository_map
from .bundle import BudgetTooSmall as BudgetTooSmall, mcp_result as mcp_result, pack_bundle
from .models import Contract, Text


class OrientRequest(Contract):
    budget: int = Field(gt=0)
    encoding: Text
    representation: Literal["detailed", "compact"] = "detailed"
    detail: Literal["full", "focused"] = "full"


def orientation(root: Path, request: OrientRequest, *, transport: Literal["cli", "mcp"] = "cli",
                observe: Callable[[dict], None] | None = None) -> dict:
    return pack_bundle(load_repository_map(root), root, budget=request.budget, encoding=request.encoding,
                       transport=transport, observe=observe, representation=request.representation, detail=request.detail)

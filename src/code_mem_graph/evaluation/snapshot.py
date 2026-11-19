"""Immutable read/search corpus for controlled discovery trials."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import stat

from pydantic import Field, model_validator

from code_mem_graph.knowledge.models import Digest, SourcePath, portable_path
from code_mem_graph.knowledge.sources import _safe_path, discover_snapshot, policy_digest, read_source
from code_mem_graph.serialization import fingerprint

from .models import Contract


class SnapshotFile(Contract):
    path: SourcePath
    digest: Digest
    content: str

    @model_validator(mode="after")
    def valid_content(self):
        if self.path == "." or hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.digest:
            raise ValueError("snapshot content does not match its file digest")
        return self


class Snapshot(Contract):
    schema_version: int = Field(default=1, ge=1, le=1)
    policy_digest: Digest
    files: list[SnapshotFile] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_paths(self):
        paths = [entry.path for entry in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("snapshot paths must be unique and sorted")
        return self

    @property
    def digest(self) -> str:
        return fingerprint({"schema_version": self.schema_version, "policy_digest": self.policy_digest,
                            "files": [{"path": entry.path, "digest": entry.digest} for entry in self.files]})

    @property
    def source_digest(self) -> str:
        """Source-only identity also catches clones with renamed knowledge metadata."""
        return fingerprint({"policy_digest": self.policy_digest,
                            "files": [{"path": entry.path, "digest": entry.digest} for entry in self.files
                                      if not entry.path.startswith(".project-knowledge/")]})

    def read(self, path: str, *, start_line: int, end_line: int) -> dict:
        if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
            raise ValueError("read requires a valid inclusive line range")
        entry = next((entry for entry in self.files if entry.path == path), None)
        if entry is None:
            raise ValueError("path is outside the frozen source catalog")
        lines = entry.content.splitlines()
        if end_line > len(lines):
            raise ValueError("read range extends beyond the frozen file")
        return {"path": path, "digest": entry.digest, "start_line": start_line, "end_line": end_line,
                "text": "\n".join(lines[start_line - 1:end_line])}

    def list_files(self, prefix: str = ".") -> list[dict]:
        portable_path(prefix)
        return [{"path": entry.path, "lines": len(entry.content.splitlines()), "digest": entry.digest}
                for entry in self.files if prefix == "." or entry.path == prefix or entry.path.startswith(prefix + "/")]

    def search(self, query: str) -> list[dict]:
        """Case-insensitive literal search; never executes repo regex or commands."""
        if not query:
            raise ValueError("search query cannot be empty")
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return [{"path": entry.path, "line": index, "text": line}
                for entry in self.files for index, line in enumerate(entry.content.splitlines(), 1) if pattern.search(line)]


def capture_snapshot(root: Path, *, private_paths: list[str] | None = None) -> Snapshot:
    paths, inventory = discover_snapshot(root)
    excluded = set(private_paths or [])
    files = []
    for path in paths:
        if path in excluded:
            continue
        content, evidence = read_source(root, path)
        files.append(SnapshotFile(path=path, digest=evidence.digest, content=content.decode("utf-8")))
    # The native arm gets the same authored/generated documents, through native reads.
    for path in (".project-knowledge/map.yaml", ".project-knowledge/README.md"):
        selected = _safe_path(root, path)
        if not stat.S_ISREG(selected.lstat().st_mode):
            raise ValueError("knowledge snapshot documents must be regular files")
        content = selected.read_bytes()
        files.append(SnapshotFile(path=path, digest=hashlib.sha256(content).hexdigest(), content=content.decode("utf-8")))
    if discover_snapshot(root)[1] != inventory:
        raise ValueError("source membership changed while capturing the trial snapshot")
    # Verify byte changes as well as membership; do not silently mix versions.
    for entry in files:
        if entry.path.startswith(".project-knowledge/"):
            actual = _safe_path(root, entry.path).read_bytes()
        else:
            actual, _ = read_source(root, entry.path)
        if hashlib.sha256(actual).hexdigest() != entry.digest:
            raise ValueError("source bytes changed while capturing the trial snapshot")
    return Snapshot(policy_digest=policy_digest(), files=sorted(files, key=lambda entry: entry.path))

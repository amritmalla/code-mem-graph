"""Conservative, allowlisted fingerprinting of repository evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat

from pathspec import GitIgnoreSpec

from code_mem_graph.serialization import fingerprint

from .models import DirectoryEvidence, FileEvidence, portable_path

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs"}
DOC_EXTENSIONS = {".md", ".markdown"}
MANIFEST_NAMES = {"pyproject.toml", "package.json", "tsconfig.json", "Cargo.toml", "go.mod", "pom.xml"}
EXCLUDED_DIRECTORIES = {
    ".git", ".hg", ".svn", ".claude-repo-mem", ".code-mem-graph", ".project-knowledge",
    ".aws", ".azure", ".ssh", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", "dist", "build", ".next", ".turbo",
}


def evidence_policy() -> dict:
    return {
        "version": "repository-evidence-v2", "file_hash": "sha256-full-bytes",
        "membership": "sorted-eligible-file-paths", "links": "reject-reparse-points",
        "code_extensions": sorted(CODE_EXTENSIONS), "doc_extensions": sorted(DOC_EXTENSIONS),
        "manifest_names": sorted(MANIFEST_NAMES), "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "workflow_paths": ".github/workflows/*.{yml,yaml}",
        "hidden_paths": "exclude-except-workflow-paths", "environment_files": "exclude-dot-env-prefix",
        "repository_rules": "nested-.gitignore-and-root-.knowledgeignore-all-files",
    }


def policy_digest() -> str:
    return fingerprint(evidence_policy())


class EvidenceUnavailable(ValueError):
    def __init__(self, reason: str, path: str):
        self.reason = reason
        super().__init__(f"{reason}: {path}")


def _allowed_directory(path: str) -> bool:
    if path == ".":
        return True
    parts = path.split("/")
    if parts[0] == ".github":
        return path in {".github", ".github/workflows"}
    return all(part not in EXCLUDED_DIRECTORIES and not part.startswith(".") for part in parts)


def _allowed_file(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    if name.startswith(".env") or not _allowed_directory("/".join(parts[:-1]) or "."):
        return False
    if parts[0] == ".github":
        return len(parts) == 3 and parts[1] == "workflows" and Path(name).suffix in {".yml", ".yaml"}
    return (not name.startswith(".")
            and (Path(name).suffix.lower() in CODE_EXTENSIONS | DOC_EXTENSIONS or name in MANIFEST_NAMES))


def _is_link(info) -> bool:
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _safe_path(root: Path, relative: str) -> Path:
    portable_path(relative)
    try:
        selected_root = root.resolve(strict=True)
        if not selected_root.is_dir():
            raise EvidenceUnavailable("unreadable_root", ".")
    except OSError as exc:
        raise EvidenceUnavailable("unreadable_root", ".") from exc
    result = selected_root
    try:
        for part in relative.split("/") if relative != "." else []:
            result = result / part
            if _is_link(result.lstat()):
                raise EvidenceUnavailable("unsafe_link", relative)
    except FileNotFoundError as exc:
        raise EvidenceUnavailable("missing", relative) from exc
    except NotADirectoryError as exc:
        raise EvidenceUnavailable("wrong_kind", relative) from exc
    except OSError as exc:
        raise EvidenceUnavailable("unreadable", relative) from exc
    return result


class IgnoreRules:
    """Git-style repository exclusions, without reading Git metadata or global config."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, GitIgnoreSpec] = {}

    def _load(self, path: str) -> GitIgnoreSpec:
        if path not in self._cache:
            try:
                selected = _safe_path(self.root, path)
            except EvidenceUnavailable as exc:
                if exc.reason == "missing":
                    self._cache[path] = GitIgnoreSpec.from_lines([])
                    return self._cache[path]
                raise
            try:
                if not stat.S_ISREG(selected.lstat().st_mode):
                    raise EvidenceUnavailable("unreadable_ignore", path)
                text = selected.read_text(encoding="utf-8-sig")
                self._cache[path] = GitIgnoreSpec.from_lines(text.splitlines())
            except (OSError, UnicodeError, ValueError) as exc:
                raise EvidenceUnavailable("unreadable_ignore", path) from exc
        return self._cache[path]

    def ignored(self, path: str, *, directory: bool = False) -> bool:
        if path == ".":
            return False
        parts = path.split("/")
        # Check ancestors too: a file negation cannot reopen a pruned directory.
        for length in range(1, len(parts) + 1):
            relative = "/".join(parts[:length])
            is_dir = length < len(parts) or directory
            target = relative + ("/" if is_dir else "")
            ignored = False
            for depth in range(length):
                prefix = "/".join(parts[:depth])
                rule_path = f"{prefix}/.gitignore" if prefix else ".gitignore"
                local = "/".join(parts[depth:length]) + ("/" if is_dir else "")
                match = self._load(rule_path).check_file(local).include
                if match is not None:
                    ignored = match
            if ignored or self._load(".knowledgeignore").match_file(target):
                return True
        return False


def read_source(root: Path, path: str) -> tuple[bytes, FileEvidence]:
    """Read once so parsed facts and their fingerprint describe the same bytes."""
    portable_path(path)
    if not _allowed_file(path):
        raise EvidenceUnavailable("excluded", path)
    if IgnoreRules(root).ignored(path):
        raise EvidenceUnavailable("repository_excluded", path)
    selected = _safe_path(root, path)
    try:
        if not stat.S_ISREG(selected.lstat().st_mode):
            raise EvidenceUnavailable("wrong_kind", path)
        content = selected.read_bytes()
    except OSError as exc:
        raise EvidenceUnavailable("unreadable", path) from exc
    evidence = FileEvidence(path=path, digest=hashlib.sha256(content).hexdigest(), policy_digest=policy_digest())
    return content, evidence


def capture_file(root: Path, path: str, *, start_line: int | None = None,
                 end_line: int | None = None) -> FileEvidence:
    content, captured = read_source(root, path)
    evidence = FileEvidence(path=path, digest=captured.digest, policy_digest=captured.policy_digest,
                            start_line=start_line, end_line=end_line)
    if end_line is not None and end_line > len(content.splitlines()):
        raise EvidenceUnavailable("invalid_line_range", path)
    return evidence


def _members(selected: Path, relative: str, recursive: bool, rules: IgnoreRules) -> list[str]:
    members = []
    pending = [(selected, relative)]
    while pending:
        directory, prefix = pending.pop()
        try:
            for entry in directory.iterdir():
                rel = f"{prefix}/{entry.name}" if prefix != "." else entry.name
                if not (_allowed_file(rel) or _allowed_directory(rel)):
                    continue
                try:
                    portable_path(rel)
                except ValueError as exc:
                    raise EvidenceUnavailable("unsafe_path", rel) from exc
                if rules.ignored(rel, directory=True) and rules.ignored(rel):
                    continue
                # A partial scan is not evidence for a complete directory membership.
                info = entry.lstat()
                if _is_link(info):
                    raise EvidenceUnavailable("unsafe_link", rel)
                if stat.S_ISDIR(info.st_mode):
                    if recursive and _allowed_directory(rel) and not rules.ignored(rel, directory=True):
                        pending.append((entry, rel))
                elif stat.S_ISREG(info.st_mode) and _allowed_file(rel) and not rules.ignored(rel):
                    portable_path(rel)
                    members.append(rel)
                elif not stat.S_ISREG(info.st_mode) and _allowed_file(rel):
                    raise EvidenceUnavailable("wrong_kind", rel)
        except OSError as exc:
            raise EvidenceUnavailable("unreadable", prefix) from exc
    return sorted(members)


def capture_directory(root: Path, path: str, *, recursive: bool = False) -> DirectoryEvidence:
    portable_path(path)
    if not _allowed_directory(path):
        raise EvidenceUnavailable("excluded", path)
    rules = IgnoreRules(root)
    if rules.ignored(path, directory=True):
        raise EvidenceUnavailable("repository_excluded", path)
    selected = _safe_path(root, path)
    try:
        if not stat.S_ISDIR(selected.lstat().st_mode):
            raise EvidenceUnavailable("wrong_kind", path)
    except OSError as exc:
        raise EvidenceUnavailable("unreadable", path) from exc
    # Membership and byte contents are separate dependencies: neither substitutes for the other.
    digest = fingerprint(_members(selected, path, recursive, rules))
    return DirectoryEvidence(path=path, digest=digest, policy_digest=policy_digest(), recursive=recursive)


def discover_sources(root: Path) -> list[str]:
    return discover_snapshot(root)[0]


def discover_snapshot(root: Path) -> tuple[list[str], DirectoryEvidence]:
    paths = _members(_safe_path(root, "."), ".", True, IgnoreRules(root))
    evidence = DirectoryEvidence(path=".", digest=fingerprint(paths), policy_digest=policy_digest(), recursive=True)
    return paths, evidence

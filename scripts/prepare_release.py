"""Prepare a reviewable source ZIP and hash manifest; never commit or publish.

Selection starts with Git's tracked/unignored working files, then applies an
explicit publication allowlist. Local state, client settings and trial reports
are outside that allowlist. The ZIP contains a snapshot, not Git history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

FILES = {".gitattributes", ".gitignore", "pyproject.toml", "README.md", "CONTRIBUTING.md",
         "LICENSE", "CHANGELOG.md", "tests/fixtures/evaluation/repository-onboarding.yaml",
         "docs/README.md", "docs/getting-started.md", "docs/architecture.md", "docs/knowledge-model.md",
         "docs/knowledge-workflow.md", "docs/contributions.md", "docs/tools.md", "docs/usage.md",
         "docs/migration.md", "docs/verification.md"}
DIRECTORIES = ("src/code_mem_graph/", "tests/", "scripts/", ".github/workflows/")
BLOCKED = {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache", ".ruff_cache",
           ".claude", ".codex", ".claude-repo-mem", ".project-knowledge", ".local"}


def included(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if set(parts) & BLOCKED or any(part.startswith(".env") for part in parts):
        return False
    if path.endswith((".pyc", ".sqlite", ".sqlite3", ".db", ".spec")) or ".mcp.json" in parts:
        return False
    return path in FILES or path.startswith(DIRECTORIES)


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=True).stdout


def optional_git_metadata(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True)
    return result.stdout.decode("utf-8").strip() if result.returncode == 0 else None


def prepare(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve()
    # Exclusive creation preserves any previous candidate for review.
    output.mkdir(parents=True, exist_ok=False)
    candidates = sorted(set(git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
                            .decode("utf-8").strip("\0").split("\0")))
    selected, excluded = [], []
    for relative in candidates:
        if not relative:
            continue
        path = root / relative
        if not included(relative) or not path.is_file():
            excluded.append(relative)
            continue
        assert not path.is_symlink() and path.resolve().is_relative_to(root), relative
        content = path.read_bytes()
        selected.append((relative, content))
    names = {name for name, _ in selected}
    assert FILES <= names, sorted(FILES - names)
    assert "src/code_mem_graph/cli.py" in names
    head = optional_git_metadata(root, "rev-parse", "--verify", "HEAD")
    origin = optional_git_metadata(root, "remote", "get-url", "origin")
    foundation = "https://github.com/amritmalla/claude-repo-mem"
    provenance = ("# Source provenance\n\n"
                  f"Original project: {foundation}\n\n"
                  f"Current repository origin: {origin or 'not configured'}\n\n"
                  f"Current repository HEAD: {head or 'no commits yet'}\n\n"
                  "This candidate is a working-tree snapshot; file hashes identify its exact contents.\n"
                  "Current Git metadata does not identify an unchanged original-project revision.\n"
                  "Original MIT license and authorship are retained in LICENSE.\n"
                  "No Git history, local settings, saved state, planning records or private trial reports are included.\n")
    selected.append(("SOURCE_PROVENANCE.md", provenance.encode("utf-8")))
    manifest = [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in selected]
    archive_path = output / "code-mem-graph-source-candidate.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in selected:
            entry = zipfile.ZipInfo("code-mem-graph/" + name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content)
    receipt = {"schema_version": 1, "foundation_repository": foundation,
               "repository_origin": origin, "repository_head": head,
               "identity": "working-tree snapshot; file hashes define the candidate",
               "history_included": False, "published": False,
               "zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
               "files": manifest, "excluded_working_paths": excluded}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    arguments = parser.parse_args()
    receipt = prepare(arguments.source, arguments.output)
    print(json.dumps({"files": len(receipt["files"]), "zip_sha256": receipt["zip_sha256"],
                      "history_included": False, "published": False}, indent=2))


if __name__ == "__main__":
    main()

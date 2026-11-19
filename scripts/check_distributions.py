"""Check release archive contents and source parity using only the standard library."""
from __future__ import annotations

import argparse
import configparser
import email
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def check(packages: Path, source: Path) -> dict:
    wheels, sdists = list(packages.glob("*.whl")), list(packages.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Use an output directory with exactly one wheel and sdist"
    expected = {path.relative_to(source / "src").as_posix(): path.read_bytes()
                for path in (source / "src/code_mem_graph").rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
                and (path.suffix == ".py" or path.name == "py.typed")}
    assert expected, "Canonical source package not found"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        assert all(name.startswith("code_mem_graph/") or name.startswith("code_mem_graph-") and ".dist-info/" in name
                   for name in names), names
        assert {name for name in names if name.startswith("code_mem_graph/")} == set(expected)
        for name, content in expected.items():
            assert archive.read(name) == content, name
        metadata = email.message_from_bytes(archive.read(next(n for n in names if n.endswith("/METADATA"))))
        assert metadata["Name"] == "code-mem-graph"
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(next(n for n in names if n.endswith("/entry_points.txt"))).decode())
        assert dict(entries["console_scripts"]) == {"code-mem-graph": "code_mem_graph.cli:main"}
        assert any(name.endswith("/LICENSE") for name in names), "License missing from wheel"
    with tarfile.open(sdists[0]) as archive:
        members = {member.name.split("/", 1)[1]: member for member in archive.getmembers() if "/" in member.name}
        # Hatch includes the VCS exclusion file so rebuilding the sdist applies it.
        allowed = {"README.md", "LICENSE", "CHANGELOG.md", "pyproject.toml", "PKG-INFO", ".gitignore"}
        allowed.update("src/" + name for name in expected)
        assert set(members) == allowed, sorted(set(members) ^ allowed)
        for name, content in expected.items():
            assert archive.extractfile(members["src/" + name]).read() == content, name
        for name in ("README.md", "LICENSE", "CHANGELOG.md", "pyproject.toml", ".gitignore"):
            assert archive.extractfile(members[name]).read() == (source / name).read_bytes(), name
    return {"distribution": metadata["Name"], "version": metadata["Version"],
            "canonical_python_files": sum(name.endswith(".py") for name in expected),
            "archive_allowlists_and_source_parity": True, "license_included": True,
            "only_canonical_console_entry": True,
            "archives": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in (wheels[0], sdists[0])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    receipt = check(arguments.dist, arguments.source)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

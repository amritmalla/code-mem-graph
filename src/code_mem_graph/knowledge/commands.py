"""Compare literal documented package commands with scoped script keys only."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import posixpath
import re
import shlex

from markdown_it import MarkdownIt

from .models import CommandCheck, portable_path
from .scoping import beneath
from .sources import read_source

_NAME = re.compile(r"[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)*\Z")
_CHECK = re.compile(
    r"(?:test|lint|check|typecheck|type-check|type:check|format|validate|verify)(?:$|[:_-])"
)
_OPTIONS = {
    "npm": {"--prefix"},
    "pnpm": {"-C", "--dir"},
    "yarn": {"--cwd"},
    "bun": {"--cwd"},
}
_SHELLS = {"bash", "sh", "shell", "console", "powershell", "cmd", "zsh"}


def _directory(value, base):
    if base is None or value.startswith(("/", "~")):
        return None
    result = posixpath.normpath(posixpath.join(base, value))
    try:
        return portable_path(result)
    except ValueError:
        return None


def _segments(line):
    line = line.strip().removeprefix("$ ")
    if any(char in line for char in ("$", "`", "\\")):
        return None
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    groups = [[]]
    for token in tokens:
        if token == "&&":
            groups.append([])
        elif token and all(char in ";&|<>()" for char in token):
            return None
        else:
            groups[-1].append(token)
    return groups


@dataclass
class _Scope:
    directory: str | None
    comparable: bool
    basis: str
    start: int | None = None


def _command(tokens, scope):
    if not tokens or tokens[0] not in _OPTIONS:
        return None
    manager = tokens[0]
    args = tokens[1:]
    directory, comparable, basis = scope.directory, scope.comparable, scope.basis
    while args and args[0].startswith("-"):
        if len(args) < 2 or args[0] not in _OPTIONS[manager]:
            return None
        directory = _directory(args[1], directory)
        comparable, basis = True, "explicit_directory_option"
        args = args[2:]
    if not args:
        return None
    explicit = args[0] in {"run", "run-script"} and (
        args[0] == "run" or manager == "npm"
    )
    if explicit:
        args = args[1:]
    if not args or not _NAME.fullmatch(args[0]) or "--if-present" in args:
        return None
    name = args[0]
    # Bun's test/build are native commands. Unsupported dispatch/binary forms
    # are not interpreted as evidence of an absent script.
    if not explicit and (manager == "bun" or (manager == "npm" and name != "test")):
        return None
    if not explicit and not _CHECK.match(name):
        return None
    return (
        manager,
        name,
        directory if comparable else None,
        basis if comparable else "unknown",
    )


def documented_commands(text, *, directory, comparable):
    """Yield supported literal command metadata and its context line range.

    Each shell fence/inline fragment starts independently. No shell, workflow,
    environment, workspace selector or project command is ever executed.
    """
    for token in MarkdownIt().parse(text):
        fragments = []
        if token.type == "fence" and token.info.strip().casefold() in _SHELLS:
            fragments = [(token.content, token.map[0] + 2)]
        elif token.type == "inline":
            cursor = 0
            for child in token.children or []:
                if child.type != "code_inline":
                    continue
                literal = child.markup + child.content + child.markup
                position = token.content.find(literal, cursor)
                if position >= 0:
                    fragments.append(
                        (
                            child.content,
                            token.map[0] + 1 + token.content.count("\n", 0, position),
                        )
                    )
                    cursor = position + len(literal)
        for fragment, first_line in fragments:
            scope = _Scope(
                directory, comparable, "document_directory" if comparable else "unknown"
            )
            for offset, line in enumerate(fragment.splitlines()):
                line_number = first_line + offset
                groups = _segments(line)
                if groups is None:
                    scope = _Scope(None, False, "unknown", scope.start or line_number)
                    continue
                for tokens in groups:
                    if not tokens:
                        continue
                    if tokens[0] == "cd":
                        scope = _Scope(
                            _directory(tokens[1], scope.directory)
                            if len(tokens) == 2
                            else None,
                            True,
                            "literal_cd",
                            scope.start or line_number,
                        )
                        continue
                    if tokens[0] in {
                        "pushd",
                        "popd",
                        "source",
                        ".",
                        "eval",
                        "if",
                        "for",
                        "while",
                        "function",
                        "Set-Location",
                    }:
                        scope = _Scope(
                            None, False, "unknown", scope.start or line_number
                        )
                        continue
                    command = _command(tokens, scope)
                    if command is not None:
                        yield command, scope.start or line_number, line_number


def compare_documented_commands(root, paths, documents, manifests):
    packages = {}
    for manifest in manifests:
        if PurePosixPath(manifest).name != "package.json":
            continue
        content, evidence = read_source(root, manifest)
        scripts = json.loads(content).get("scripts", {})
        packages[str(PurePosixPath(manifest).parent)] = (
            manifest,
            sorted(scripts),
            evidence,
        )
    for path in sorted(documents):
        content, evidence = read_source(root, path)
        directory = max(
            (base for base in packages if beneath(path, base)), key=len, default="."
        )
        # A shared root/container guide cannot assign commands to every child
        # package just because their scripts have similarly named entries.
        comparable = directory in packages and not any(
            base != directory and beneath(base, directory) for base in packages
        )
        seen = set()
        for (manager, name, scoped_directory, basis), start, end in documented_commands(
            content.decode("utf-8-sig"),
            directory=directory,
            comparable=comparable,
        ):
            known = packages.get(scoped_directory)
            status = (
                "scope_unknown"
                if known is None
                else "declared_script"
                if name in known[1]
                else "script_not_declared"
            )
            check = CommandCheck(
                document=path,
                manager=manager,
                script=name,
                directory=scoped_directory,
                manifest=known[0] if known else None,
                status=status,
                scope_basis=basis if known else "unknown",
                similar_names=[
                    candidate
                    for candidate in known[1]
                    if candidate != name
                    and re.sub(r"[:_-]", "", candidate) == re.sub(r"[:_-]", "", name)
                ]
                if known and status == "script_not_declared"
                else [],
            )
            identity = (manager, name, scoped_directory, basis)
            if identity in seen:
                continue
            seen.add(identity)
            reference = evidence.model_copy(
                update={"start_line": start, "end_line": end}
            )
            yield check, [reference, *([known[2]] if known else [])]

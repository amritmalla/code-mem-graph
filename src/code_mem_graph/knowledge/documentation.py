"""Bounded Markdown guidance extraction; statements remain authored excerpts."""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
import re
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from .sources import CODE_EXTENSIONS

SETUP = {
    "quick start",
    "getting started",
    "get started",
    "agent integration",
    "local development",
    "running locally",
    "installation",
    "development setup",
    "setup commands",
}
VALIDATION = {
    "running tests",
    "testing",
    "linting",
    "backend tests",
    "verification",
    "validation",
    "checks",
    "tests",
    "type checking",
    "testing strategy",
    "types of tests",
    "unit tests",
    "integration tests",
    "e2e",
    "end-to-end testing",
    "end-to-end testing with playwright",
}
RESPONSIBILITIES = {
    "project layout",
    "components",
    "component responsibilities",
    "general workflow",
    "making requests to the backend api",
    "project structure",
    "code structure",
    "feature structure",
    "application structure",
    "architecture",
    "architecture rules",
}
PREREQUISITES = {"requirements", "prerequisites"}
CONVENTIONS = {
    "conventions",
    "coding standards",
    "code standards",
    "project standards",
    "file naming conventions",
    "code style",
    "migrations",
    "expanding the oxlint configuration",
    "architecture rules",
}
CLIENT = {"generate client", "client generation", "api client generation"}
DOMAIN = {"application overview", "application domain", "data model"}


def normalize_heading(token):
    # Link labels/emphasis/code are text; URL destinations and decorative emoji
    # are not part of a section's meaning. Keep the original heading for display.
    text = (
        "".join(
            child.content
            for child in token.children or []
            if child.type in {"text", "code_inline"}
        )
        or token.content
    )
    return re.sub(r"^[^\w]+", "", text.casefold()).strip().rstrip(":").strip()


def classification(heading):
    if heading in CONVENTIONS:
        return "conventions", "constraints"
    if heading in SETUP:
        return "workflow", "entrypoints"
    if (
        heading in VALIDATION
        or heading in CLIENT
        or heading in {"release", "release flow"}
    ):
        return "workflow", "workflow"
    if heading in RESPONSIBILITIES:
        return "architecture", "components"
    if heading in PREREQUISITES:
        return "workflow", "constraints"
    if heading in DOMAIN:
        return "domain", None
    return None


@dataclass(frozen=True)
class Section:
    heading: str
    normalized: str
    topic: str
    facet: str | None
    body: str
    start: int
    end: int
    # Heading, verbatim preamble, one-based inclusive evidence range.
    context: tuple[tuple[str, str, int, int], ...]


def sections(text):
    tokens = MarkdownIt().parse(text)
    lines = text.splitlines()
    headings = []
    stack = []
    for index, token in enumerate(tokens[:-1]):
        if token.type != "heading_open":
            continue
        level = int(token.tag[1:])
        while stack and headings[stack[-1]]["level"] >= level:
            stack.pop()
        headings.append(
            {
                "heading": tokens[index + 1].content.strip(),
                "normalized": normalize_heading(tokens[index + 1]),
                "level": level,
                "start": token.map[0],
                "body_start": token.map[1],
                "ancestors": tuple(stack),
            }
        )
        stack.append(len(headings) - 1)
    for index, entry in enumerate(headings):
        kind = classification(entry["normalized"])
        if kind is None:
            continue
        # Keep one chapter per topic/facet, including its nested examples and
        # preconditions. Don't repeat each testing subsection inside Testing.
        if any(
            classification(headings[parent]["normalized"]) == kind
            for parent in entry["ancestors"]
        ):
            continue
        end = next(
            (
                later["start"]
                for later in headings[index + 1 :]
                if later["level"] <= entry["level"]
            ),
            len(lines),
        )
        body = _excerpt(lines[entry["body_start"] : end])
        if not body:
            continue
        context = []
        for ancestor_index in entry["ancestors"]:
            ancestor = headings[ancestor_index]
            preamble_end = headings[ancestor_index + 1]["start"]
            preamble = _excerpt(lines[ancestor["body_start"] : preamble_end])
            context.append(
                (
                    ancestor["heading"],
                    preamble,
                    ancestor["start"] + 1,
                    max(ancestor["body_start"], preamble_end),
                )
            )
        yield Section(
            entry["heading"],
            entry["normalized"],
            kind[0],
            kind[1],
            body,
            entry["start"] + 1,
            end,
            tuple(context),
        )


def _excerpt(lines):
    # Trim surrounding blank lines, preserving indentation in Markdown code.
    first, last = 0, len(lines)
    while first < last and not lines[first].strip():
        first += 1
    while last > first and not lines[last - 1].strip():
        last -= 1
    return "\n".join(lines[first:last])


def has_commands(tokens):
    command = re.compile(
        r"^(?:\$\s*)?(?:uv\s+(?:sync|run|pip)\b|(?:npm|pnpm|yarn|bun)\s+(?:install|ci|run|start|test|dev|build)\b|"
        r"(?:python(?:3)?\s+-m\s+)?(?:pytest|ruff|mypy)\b|docker\s+compose\b)"
    )
    for token in tokens:
        if (
            token.type == "fence"
            and token.content.strip()
            and token.info.casefold().strip()
            in {"bash", "sh", "shell", "powershell", "console", "cmd", "zsh"}
        ):
            lines = token.content.splitlines()
            tree = any(re.search(r"(?:[├└][─-]{2}|\+--)", line) for line in lines)
            if not tree or any(command.match(line.strip()) for line in lines):
                return True
        if any(
            child.type == "code_inline" and command.match(child.content.strip())
            for child in token.children or []
        ):
            return True
    return False


def documented_paths(tokens, paths, *, document, package_root="."):
    """Resolve Markdown links relative to their document; code paths need evidence.

    Inline paths can be package-, document-, or repository-relative. If these
    produce different existing targets, leave the association unknown.
    """
    eligible = set(paths)
    directories = {prefix for path in paths for prefix in _parents(path)}
    known = eligible | directories
    found = set()
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                try:
                    target = urlsplit(child.attrGet("href") or "")
                except ValueError:
                    continue
                if (
                    target.scheme
                    or target.netloc
                    or not target.path
                    or target.path.startswith("/")
                ):
                    continue
                value = posixpath.normpath(
                    posixpath.join(posixpath.dirname(document), unquote(target.path))
                )
                if value in known:
                    found.add(value)
            elif child.type == "code_inline":
                value = child.content.strip().rstrip("/")
                if not value or value.startswith(("/", "~")) or "\\" in value:
                    continue
                candidates = {
                    posixpath.normpath(posixpath.join(base, value))
                    for base in (package_root, posixpath.dirname(document), ".")
                }
                existing = candidates & known
                if len(existing) == 1:
                    found.update(existing)
    return sorted(found)


def _parents(path):
    while "/" in path:
        path = path.rsplit("/", 1)[0]
        yield path


def is_test_example(path):
    if posixpath.splitext(path)[1].casefold() not in CODE_EXTENSIONS:
        return False
    return (
        bool({"tests", "test", "__tests__"}.intersection(path.split("/")))
        or re.search(
            r"(?:^|/)test_[^/]+\.py$|\.(?:test|spec)\.[cm]?[jt]sx?$", path, re.I
        )
        is not None
    )

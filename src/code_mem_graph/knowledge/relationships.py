"""Static local import navigation. Never imports or executes target project code."""

from __future__ import annotations

import ast
from functools import lru_cache
import json
from pathlib import PurePosixPath
import posixpath

from .models import LocationRelationship
from .sources import read_source


def is_import_record(item):
    return (
        item.generator is not None
        and item.generator.name == "repository-facts"
        and item.id.startswith("source/")
        and item.id.endswith("/imports")
    )


@lru_cache(maxsize=3)
def _parser(suffix):
    from tree_sitter import Language, Parser
    import tree_sitter_javascript
    import tree_sitter_typescript

    if suffix == ".ts":
        language = tree_sitter_typescript.language_typescript()
    elif suffix == ".tsx":
        language = tree_sitter_typescript.language_tsx()
    else:
        language = tree_sitter_javascript.language()
    return Parser(Language(language))


def _javascript_declarations(path, content):
    tree = _parser(PurePosixPath(path).suffix).parse(content)
    # Recovery nodes can look like real declarations. Leave a malformed file
    # unknown instead of extracting a plausible edge from a broken parse.
    if tree.root_node.has_error:
        return
    for node in tree.root_node.named_children:
        if node.type not in {"import_statement", "export_statement"}:
            continue
        source = node.child_by_field_name("source")
        if source is None:
            continue
        raw = content[source.start_byte : source.end_byte].decode("utf-8")
        if len(raw) < 2 or raw[0] not in "\"'" or "\\" in raw:
            continue
        yield raw[1:-1], node.start_point[0] + 1, node.end_point[0] + 1


def _js_targets(base, eligible):
    if PurePosixPath(base).suffix:
        candidates = {base}
    else:
        candidates = {base + suffix for suffix in (".ts", ".tsx", ".js", ".jsx")} | {
            posixpath.join(base, "index" + suffix)
            for suffix in (".ts", ".tsx", ".js", ".jsx")
        }
    return candidates & eligible


def _alias_targets(config, specifier):
    options = config.get("compilerOptions", {})
    if not isinstance(options, dict) or "extends" in config:
        return []
    base = options.get("baseUrl", ".")
    paths = options.get("paths", {})
    if not isinstance(base, str) or not isinstance(paths, dict):
        return []
    found = []
    for key, values in paths.items():
        if (
            not isinstance(values, list)
            or not isinstance(key, str)
            or key.count("*") > 1
        ):
            continue
        if "*" in key:
            prefix, suffix = key.split("*")
            if not specifier.startswith(prefix) or not specifier.endswith(suffix):
                continue
            middle = specifier[
                len(prefix) : len(specifier) - len(suffix) if suffix else None
            ]
        elif key == specifier:
            middle = ""
        else:
            continue
        for value in values:
            if isinstance(value, str) and value.count("*") <= 1:
                found.append(posixpath.join(base, value.replace("*", middle)))
    return found


def _json_config(content):
    """Accept tsconfig comments/trailing commas without rewriting string values."""
    text = content.decode("utf-8-sig")
    cleaned = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            end = index + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                elif text[end] == '"':
                    end += 1
                    break
                else:
                    end += 1
            cleaned.append(text[index:end])
            index = end
            continue
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            cleaned.append(" ")
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated tsconfig comment")
            cleaned.append(" ")
            index = end + 2
            continue
        cleaned.append(char)
        index += 1
    text = "".join(cleaned)
    # A second pass sees comments removed, but still skips complete strings.
    result = []
    index = 0
    while index < len(text):
        if text[index] == '"':
            end = index + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                elif text[end] == '"':
                    end += 1
                    break
                else:
                    end += 1
            result.append(text[index:end])
            index = end
            continue
        if text[index] == "," and text[index + 1 :].lstrip().startswith(("}", "]")):
            index += 1
            continue
        result.append(text[index])
        index += 1
    return json.loads("".join(result))


def import_records(root, paths, manifests):
    """Yield importer, declaration range, unique edges and alias-config evidence.

    JS: static imports/re-exports; relative modules and simple explicit local
    tsconfig paths. Python: literal import/from modules under the nearest
    manifest directory or repository root, plus explicit relative imports.
    Ambiguous targets, external packages and unsupported syntax produce no edge.
    """
    eligible = set(paths)
    package_roots = {str(PurePosixPath(p).parent) for p in manifests}
    configs = {}
    for path in paths:
        if PurePosixPath(path).name != "tsconfig.json":
            continue
        content, reference = read_source(root, path)
        try:
            data = _json_config(content)
        except (ValueError, UnicodeError):
            data = None
        configs[path] = data if isinstance(data, dict) else {}, reference
    for path in paths:
        suffix = PurePosixPath(path).suffix
        if suffix not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            continue
        content, reference = read_source(root, path)
        directory = posixpath.dirname(path) or "."
        edges, ranges, used_configs = {}, [], {}
        if suffix == ".py":
            try:
                tree = ast.parse(content)
            except (SyntaxError, ValueError):
                continue
            package = max(
                (p for p in package_roots if p != "." and path.startswith(p + "/")),
                key=len,
                default=".",
            )
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [(alias.name, 0) for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        modules = [(node.module, node.level)]
                    elif node.level:
                        modules = [
                            (alias.name, node.level)
                            for alias in node.names
                            if alias.name != "*"
                        ]
                for module, level in modules:
                    roots = {package, "."}
                    if level:
                        if level > len(PurePosixPath(path).parts) - 1:
                            continue
                        base = directory
                        for _ in range(level - 1):
                            base = posixpath.dirname(base) or "."
                        roots = {base}
                    candidates = {
                        posixpath.normpath(
                            posixpath.join(base, module.replace(".", "/") + tail)
                        )
                        for base in roots
                        for tail in (".py", "/__init__.py")
                    } & eligible
                    if len(candidates) == 1:
                        target = next(iter(candidates))
                        if target != path:
                            edges[target] = LocationRelationship(
                                kind="static_import", source=path, target=target
                            )
                            ranges.append((node.lineno, node.end_lineno))
        else:
            config_path = next(
                (
                    str(parent / "tsconfig.json")
                    for parent in PurePosixPath(path).parents
                    if str(parent / "tsconfig.json") in configs
                ),
                None,
            )
            for specifier, start, end in _javascript_declarations(path, content):
                bases = []
                if specifier.startswith(("./", "../")):
                    bases = [posixpath.normpath(posixpath.join(directory, specifier))]
                elif config_path:
                    bases = [
                        posixpath.normpath(
                            posixpath.join(posixpath.dirname(config_path), value)
                        )
                        for value in _alias_targets(configs[config_path][0], specifier)
                    ]
                targets = {
                    target for base in bases for target in _js_targets(base, eligible)
                }
                if len(targets) != 1:
                    continue
                target = next(iter(targets))
                if target != path:
                    edges[target] = LocationRelationship(
                        kind="static_import", source=path, target=target
                    )
                    ranges.append((start, end))
                    if not specifier.startswith(".") and config_path:
                        used_configs[config_path] = configs[config_path][1]
        if edges:
            declaration = reference.model_copy(
                update={
                    "start_line": min(start for start, _ in ranges),
                    "end_line": max(end for _, end in ranges),
                }
            )
            yield (
                path,
                declaration,
                [edges[key] for key in sorted(edges)],
                list(used_configs.values()),
            )

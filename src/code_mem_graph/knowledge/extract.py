"""Deterministic, cited repository facts. No code execution or model synthesis."""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import posixpath
import re
import tomllib
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import yaml

from code_mem_graph.serialization import fingerprint

from .models import Component, Generator, ItemDependency, KnowledgeConflict, KnowledgeGap, KnowledgeItem, KnowledgeLayer, LocationRelationship, portable_path
from .sources import CODE_EXTENSIONS, capture_directory, discover_snapshot, read_source
from .documentation import (
    CONVENTIONS, PREREQUISITES, RESPONSIBILITIES, SETUP, VALIDATION,
    documented_paths, has_commands, is_test_example, normalize_heading, sections as documented_sections,
)
from .scoping import manifest_paths as _manifest_paths, instruction_paths
from .relationships import import_records
from .commands import compare_documented_commands

FACETS = ("purpose", "stack", "entrypoints", "components", "constraints", "workflow")
ONBOARDING = ("purpose", "connection", "prerequisites", "responsibilities", "conventions", "validation")
GENERATOR = Generator(name="repository-facts", version="8")
_NAME = re.compile(r"(?:@[A-Za-z0-9._-]+/)?[A-Za-z0-9._-]+\Z")
_SCRIPT_NAME = re.compile(r"(?:@[A-Za-z0-9._-]+/)?[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)*\Z")
_RUNTIME = re.compile(r"[0-9vVxX*<>=~^!|., +\-]+\Z")
_PY_ENTRY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")


def _key(value: str) -> str:
    return fingerprint(value)[:20]


def _linked_docs(root: Path, paths: list[str], seeds: set[str]) -> set[str]:
    """Follow only local Markdown links already in the eligible source catalog."""
    eligible = set(paths)
    selected = {path for path in paths if path.lower() in seeds}
    pending = list(sorted(selected))
    while pending:
        path = pending.pop()
        content, _ = read_source(root, path)
        for token in MarkdownIt().parse(content.decode("utf-8-sig")):
            for child in token.children or []:
                if child.type != "link_open":
                    continue
                target = child.attrGet("href") or ""
                try:
                    parts = urlsplit(target)
                except ValueError:
                    continue
                if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
                    continue
                linked = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(parts.path)))
                if linked in eligible and linked not in selected and Path(linked).suffix.lower() in {".md", ".markdown"}:
                    selected.add(linked)
                    pending.append(linked)
    return selected


def extract_layer(root: Path) -> KnowledgeLayer:
    paths, inventory = discover_snapshot(root)
    manifests = _manifest_paths(paths)
    instructions = instruction_paths(paths)
    directories = {path.split("/")[0] for path in paths if "/" in path and not path.startswith(".")}
    directories.update("/".join(path.split("/")[:2]) for path in paths
                       if path.startswith("src/") and len(path.split("/")) > 2)
    # Retain existing directory IDs and add each manifest's containing scope and
    # ancestors, including multiple nested applications under a shared container.
    for manifest in [*manifests,*instructions]:
        for parent in PurePosixPath(manifest).parents:
            if str(parent) != ".":
                directories.add(str(parent))
    components = [Component(id="project", name="Repository", paths=["."])]
    for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
        parent_path = str(PurePosixPath(directory).parent)
        parent = f"directory/{_key(parent_path)}" if parent_path != "." else "project"
        components.append(Component(id=f"directory/{_key(directory)}", name=directory, paths=[directory], parent_id=parent))
    items: list[KnowledgeItem] = []
    conflicts = []
    documented_node = []
    declared_node = []

    def component_for(path):
        return next((component.id for component in reversed(components)
                     if component.paths[0] != "." and path.startswith(component.paths[0] + "/")), "project")

    def add(item_id, topic, title, statement, evidence, facets=(), *, basis="observed", component="project", critical=False,
            onboarding=(), relationships=(), dependencies=(), command_checks=()):
        items.append(KnowledgeItem(id=item_id, topic=topic, title=title, statement=statement,
                                   origin="generated", basis=basis, generator=GENERATOR,
                                   evidence=evidence, facets=list(facets), onboarding=list(onboarding),
                                   component_ids=[component], critical=critical,
                                   relationships=list(relationships), dependencies=list(dependencies),
                                   command_checks=list(command_checks)))

    counts = Counter(Path(path).suffix.lower() for path in paths if Path(path).suffix.lower() in CODE_EXTENSIONS)
    summary = ", ".join(f"{extension}: {count}" for extension, count in sorted(counts.items())) or "no supported source files"
    add("project/source-inventory", "overview", "Source inventory",
        f"Eligible repository files: {len(paths)}. Source extensions: {summary}. "
        "File extensions identify source locations, not runtime architecture.", [inventory],
        ("components", "stack") if counts else ("components",))
    for component in components[1:]:
        directory = component.paths[0]
        count = sum(path.startswith(directory + "/") for path in paths)
        add(f"{component.id}/inventory", "architecture", f"Files in {directory}",
            f"{directory}/ contains {count} eligible files. Its responsibilities are not inferred from the directory name.",
            [capture_directory(root, directory, recursive=True)], ("components",), component=component.id)

    selected_docs = {"readme.md", "readme.markdown", "contributing.md", "agents.md", "claude.md", "docs/architecture.md",
                     "docs/development.md", "docs/testing.md", "docs/release.md"}
    package_roots = {str(PurePosixPath(path).parent) for path in manifests}
    for directory in package_roots - {"."}:
        selected_docs.update(f"{directory}/{name}".casefold() for name in
                             ("README.md", "README.markdown", "CONTRIBUTING.md"))
    selected_docs = _linked_docs(root, paths, selected_docs)
    selected_docs.update(instructions)
    target_references = {}

    def location_reference(path):
        if path not in target_references:
            target_references[path] = read_source(root, path)[1] if path in paths else capture_directory(root,path,recursive=True)
        return target_references[path]

    for path in paths:
        if path not in selected_docs:
            continue
        content, evidence = read_source(root, path)
        text = content.decode("utf-8-sig")
        if path in instructions:
            directory = str(PurePosixPath(path).parent)
            applicability = "" if directory=="." else f" Applies only within {directory}/."
            add(f"docs/{_key(path)}/agent-guidance", "conventions", f"Declared repository guidance: {path}",
                f"Repository guidance documented in {path}.{applicability}\n{text}", [evidence], ("constraints",),
                basis="documented", critical=directory==".", component=component_for(path))
            if directory!='.':
                # Conditional local rules must not establish repository-wide
                # onboarding coverage merely because their file was discovered.
                continue
        tokens = MarkdownIt().parse(text)
        if path.lower() in {"readme.md", "readme.markdown"}:
            for major in sorted(set(re.findall(r"\bNode(?:\.js)?\s+v?(\d+)\b", text))):
                item_id = f"docs/{_key(path)}/node/{major}"
                add(item_id, "conventions", "Documented Node version", f"{path} mentions Node.js {major}.",
                    [evidence], ("constraints",), basis="documented")
                documented_node.append((item_id, int(major), path))
        if "/" not in path and path.lower().startswith("readme.") and not any(item.id == "project/purpose" for item in items):
            introduction = True
            for index, token in enumerate(tokens[:-1]):
                if token.type == "heading_open" and int(token.tag[1:]) >= 2:
                    introduction = normalize_heading(tokens[index + 1]) in {"purpose", "overview", "about"}
                if introduction and token.type == "paragraph_open" and tokens[index + 1].type == "inline":
                    paragraph = tokens[index + 1].content.strip()
                    if not paragraph or paragraph.startswith(("![", "[![")):
                        continue
                    reference = evidence.model_copy(update={"start_line": token.map[0] + 1, "end_line": token.map[1]})
                    add("project/purpose", "overview", "Documented project introduction",
                        f"{path} states:\n{paragraph}", [reference], ("purpose",), basis="documented", onboarding=("purpose",))
                    break
        seen_headings: Counter = Counter()
        package_root = max((directory for directory in package_roots if path.startswith(directory + "/")),key=len,default=".")
        for section in documented_sections(text):
            heading,normalized,excerpt = section.heading,section.normalized,section.body
            seen_headings[normalized] += 1
            onboarding = []
            excerpt_tokens = MarkdownIt().parse(excerpt)
            locations = documented_paths(excerpt_tokens,paths,document=path,package_root=package_root)
            supporting_locations = set()
            if normalized in RESPONSIBILITIES and locations and re.search(
                r"\b(?:owns?|handles?|contains?|models?|endpoints?|utils|renders?|configur\w*|routes?|stores?|holds?|provid\w*|responsib\w*)\b", excerpt, re.I
            ):
                onboarding.append("responsibilities")
                supporting_locations.update(locations)
            # A heading or a configuration declaration alone is not actionable
            # setup/check evidence. Retain the documented command block verbatim.
            commands = has_commands(excerpt_tokens)
            if commands and normalized in SETUP:
                onboarding.append("connection")
            if normalized in PREREQUISITES and re.search(r"\b(?:requires?|python|node|docker|uv|pnpm|npm|yarn|bun)\b", excerpt, re.I):
                onboarding.append("prerequisites")
            if normalized in SETUP and re.search(r"\b(?:requires?|prerequisites?)\b", excerpt, re.I) and re.search(
                r"\b(?:python|node|docker|uv|pnpm|npm|yarn|bun)\b", excerpt, re.I
            ):
                onboarding.append("prerequisites")
            test_locations = {location for location in locations if location in paths and is_test_example(location)}
            if normalized in VALIDATION and (commands or test_locations):
                onboarding.append("validation")
                supporting_locations.update(test_locations)
            if normalized in CONVENTIONS and not re.fullmatch(r"(?:TODO|TBD|To be written)[.!\s]*",excerpt,re.I):
                onboarding.append("conventions")
            reference = evidence.model_copy(update={"start_line": section.start, "end_line": section.end})
            references = [reference]
            context = []
            for parent,preamble,start,end in section.context:
                references.append(evidence.model_copy(update={"start_line": start,"end_line": end}))
                if preamble:
                    context.append(f"Parent section context ({parent}):\n{preamble}")
            # These are locations explicitly named together in a documented
            # chapter, not a causal change-propagation chain.
            locations = [location for location in locations if location != path]
            references.extend(location_reference(location) for location in sorted(set(locations) | supporting_locations) if location != path)
            heading_path = " > ".join([*(entry[0] for entry in section.context),heading])
            body = "\n\n".join([*context,excerpt])
            add(f"docs/{_key(path)}/{_key(normalized)}/{seen_headings[normalized]}", section.topic,
                f"{path}: {heading_path}", f"Documented excerpt from {path} ({heading_path}):\n{body}",
                references, (section.facet,) if section.facet else (), basis="documented", component=component_for(path), onboarding=onboarding,
                relationships=[LocationRelationship(kind="documented_location", source=path, target=location) for location in locations])

    inventory_item = next(item for item in items if item.id == "project/source-inventory")
    for path, declaration, relationships, configs in import_records(root, paths, manifests):
        targets = [edge.target for edge in relationships]
        add(f"source/{_key(path)}/imports", "architecture", f"Static local imports: {path}",
            f"{path} declares static module imports/re-exports with unique eligible local targets: {', '.join(targets)}. "
            "This is source navigation, not runtime binding, a call/data-flow graph or proof of test coverage. "
            "External, ambiguous, dynamic and unsupported imports are not represented.",
            [declaration, *(location_reference(target) for target in targets), *configs],
            component=component_for(path), relationships=relationships,
            dependencies=[ItemDependency(item_id=inventory_item.id, content_digest=inventory_item.content_digest)])

    for path in manifests:
        content, evidence = read_source(root, path)
        prefix, component = f"manifest/{_key(path)}", component_for(path)
        if path.endswith("pyproject.toml"):
            data = tomllib.loads(content.decode("utf-8-sig"))
            project = data.get("project", {})
            if not isinstance(project, dict):
                raise ValueError(f"{path}: project must be a table")
            name, description = project.get("name"), project.get("description")
            if isinstance(name, str) and _NAME.fullmatch(name):
                add(prefix + "/name", "overview", "Declared Python package", f"{path} declares Python package {name}.",
                    [evidence], component=component)
            if path == "pyproject.toml" and isinstance(description, str) and description.strip() and not any(item.id == "project/purpose" for item in items):
                add("project/purpose", "overview", "Declared project description",
                    f"{path} describes the project as:\n{description}", [evidence], ("purpose",), basis="documented", onboarding=("purpose",))
            runtime = project.get("requires-python")
            if isinstance(runtime, str) and _RUNTIME.fullmatch(runtime):
                add(prefix + "/runtime", "overview", "Declared Python runtime",
                    f"{path} declares requires-python = {runtime}.", [evidence], ("stack", "constraints"), component=component)
            scripts = project.get("scripts", {})
            if not isinstance(scripts, dict):
                raise ValueError(f"{path}: project.scripts must be a table")
            for name, target in sorted(scripts.items()):
                if _NAME.fullmatch(name) and isinstance(target, str) and _PY_ENTRY.fullmatch(target):
                    add(prefix + "/entry/" + _key(name), "architecture", f"Console entry: {name}",
                        f"{path} declares console entry {name} -> {target}.", [evidence], ("entrypoints",), component=component)
            tool = data.get("tool", {})
            if isinstance(tool, dict) and "pytest" in tool:
                add(prefix + "/pytest", "workflow", "Declared pytest configuration",
                    f"{path} declares pytest configuration; inspect its options and project docs before running checks.",
                    [evidence], ("workflow",), component=component)
            fastapi = tool.get("fastapi", {}) if isinstance(tool, dict) else {}
            target = fastapi.get("entrypoint") if isinstance(fastapi, dict) else None
            if isinstance(target, str) and _PY_ENTRY.fullmatch(target):
                add(prefix + "/entry/fastapi", "architecture", "Declared FastAPI entry point",
                    f"{path} declares tool.fastapi.entrypoint = {target}; declaration does not prove the target is runnable.",
                    [evidence], ("entrypoints",), component=component)
            dependencies = project.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError(f"{path}: project.dependencies must be an array")
            names = sorted({match.group(1) for value in dependencies if isinstance(value, str)
                            and (match := re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:[\s\[<>=!~@;]|$)", value))})
        else:
            import json
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: package manifest must be an object")
            runtime = data.get("engines", {}).get("node") if isinstance(data.get("engines", {}), dict) else None
            if isinstance(runtime, str) and _RUNTIME.fullmatch(runtime):
                add(prefix + "/runtime", "overview", "Declared Node runtime",
                    f"{path} declares Node.js engine {runtime}.", [evidence], ("stack", "constraints"), component=component)
                if path == "package.json":
                    declared_node.append((prefix + "/runtime", runtime))
            scripts = data.get("scripts", {})
            if not isinstance(scripts, dict):
                raise ValueError(f"{path}: scripts must be an object")
            for name in sorted(scripts):
                if _SCRIPT_NAME.fullmatch(name):
                    add(prefix + "/script/" + _key(name), "workflow", f"Declared script: {name}",
                        f"{path} declares script {name}; inspect the manifest before executing it.", [evidence], ("workflow",), component=component)
            entries = {key: data[key] for key in ("main", "module") if isinstance(data.get(key), str)}
            binaries = data.get("bin", {})
            if isinstance(binaries, str):
                entries["bin"] = binaries
            elif isinstance(binaries, dict):
                entries.update({f"bin/{name}": target for name, target in binaries.items()
                                if _NAME.fullmatch(name) and isinstance(target, str)})
            for name, target in sorted(entries.items()):
                target = target.removeprefix("./")
                try:
                    portable_path(target)
                except ValueError:
                    continue
                add(prefix + "/entry/" + _key(name), "architecture", f"Declared entry: {name}",
                    f"{path} declares {name} -> {target}; declaration does not prove the target is runnable.",
                    [evidence], ("entrypoints",), component=component)
            names = set()
            for field in ("dependencies", "devDependencies", "peerDependencies"):
                values = data.get(field, {})
                if not isinstance(values, dict):
                    raise ValueError(f"{path}: {field} must be an object")
                names.update(name for name in values if _NAME.fullmatch(name))
            names = sorted(names)
        if names:
            add(prefix + "/dependencies", "architecture", "Declared dependency names",
                f"{path} declares dependency names: {', '.join(names)}. Versions, URLs, and credentials are not extracted.",
                [evidence], ("stack",), component=component)

    for check, references in compare_documented_commands(root, paths, selected_docs, manifests):
        invocation = f"{check.manager} {check.script}"
        if check.status == "scope_unknown":
            description = f"{check.document} mentions {invocation}, but an eligible application manifest scope is not established. No script mismatch is asserted."
        else:
            description = (
                f"{check.document} mentions {invocation} in directory {check.directory} "
                f"(scope basis: {check.scope_basis}). {check.manifest} "
                + (f"declares script key {check.script}." if check.status == "declared_script"
                   else f"does not declare script key {check.script}.")
            )
            if check.similar_names:
                description += f" Similarly spelled script keys: {', '.join(check.similar_names)}; spelling does not establish equivalence or a replacement."
        description += " This compares documented names with manifest keys only; command resolution, external binaries/workspaces, prerequisites and execution are not verified."
        identity = _key(f"{check.document}|{check.manager}|{check.script}|{check.directory}|{check.scope_basis}")
        add(f"command/{identity}", "workflow", f"Documented command: {invocation} ({check.status})",
            description, references, ("workflow",), component=component_for(
                (check.directory + "/package.json") if check.directory and check.directory != "." else "package.json"
            ), critical=check.status == "script_not_declared", command_checks=[check],
            dependencies=[ItemDependency(item_id=inventory_item.id, content_digest=inventory_item.content_digest)])

    for path in paths:
        if not path.startswith(".github/workflows/"):
            continue
        content, evidence = read_source(root, path)
        data = yaml.load(content.decode("utf-8-sig"), Loader=yaml.BaseLoader)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: workflow must be a mapping")
        trigger = data.get("on", {})
        triggers = sorted(trigger) if isinstance(trigger, (dict, list)) else [trigger] if isinstance(trigger, str) else []
        triggers = [name for name in triggers if isinstance(name, str) and re.fullmatch(r"[a-z_]+", name)]
        add(f"workflow/{_key(path)}", "workflow", f"Workflow file: {path}",
            f"{path} declares workflow triggers: {', '.join(triggers) or 'not established'}. "
            "Job behavior, approvals, and external environment settings require further evidence.", [evidence], ("workflow",))

    for manifest_id, runtime in declared_node:
        simple = re.fullmatch(r"\s*(>=|<=|>|<|\^|~|=|==)?\s*(\d+)\s*", runtime)
        if simple is None:
            continue
        operator, required = simple.group(1) or "=", int(simple.group(2))
        for doc_id, major, path in documented_node:
            incompatible = (major < required if operator in {">=", ">"}
                            else major > required if operator == "<="
                            else major >= required if operator == "<" else major != required)
            if incompatible:
                competing = [doc_id, manifest_id]
                purpose = next((item for item in items if item.id == "project/purpose"), None)
                if purpose and any(ref.path == path for ref in purpose.evidence) and re.search(r"\bNode(?:\.js)?\s+v?" + str(major) + r"\b", purpose.statement):
                    competing.append(purpose.id)
                conflicts.append(KnowledgeConflict(id=f"runtime/node/{_key(doc_id)}", item_ids=competing,
                    description=f"{path} mentions Node.js {major}, incompatible with package.json engine {runtime}."))
    topics = {"purpose": "overview", "stack": "overview", "entrypoints": "architecture", "components": "architecture",
              "constraints": "conventions", "workflow": "workflow"}
    gaps = []
    for facet in FACETS:
        supporting = [item.id for item in items if facet in item.facets]
        gaps.append(KnowledgeGap(id=f"orientation/{facet}", topic=topics[facet],
                                 description=f"Automatic extraction has not established {facet}; consult cited sources or a maintainer.",
                                 status="resolved" if supporting else "open", resolved_by=supporting))
    for question in ONBOARDING:
        supporting = [item.id for item in items if question in item.onboarding]
        gaps.append(KnowledgeGap(id=f"onboarding/{question}", topic="overview",
                                 description=f"No extracted onboarding evidence for {question}; inspect project docs or ask a maintainer.",
                                 status="resolved" if supporting else "open", resolved_by=supporting))
    for topic, label in (("architecture", "architecture-rationale"), ("domain", "domain-meaning"),
                         ("decision", "decisions"), ("workflow", "release-approvers")):
        gaps.append(KnowledgeGap(id=f"knowledge/{label}", topic=topic,
                                 description=f"Automatic extraction has not established {label}; maintainer knowledge may supply it."))
    return KnowledgeLayer(components=components, items=items, gaps=gaps, conflicts=conflicts)

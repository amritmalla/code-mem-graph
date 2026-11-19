"""Explicit directory boundaries and directory-scoped instruction discovery."""

from pathlib import PurePosixPath

NON_PROJECT_MANIFEST_DIRECTORIES = {
    "test",
    "tests",
    "__tests__",
    "fixture",
    "fixtures",
    "__fixtures__",
    "example",
    "examples",
    "sample",
    "samples",
    "mock",
    "mocks",
    "__mocks__",
    "vendor",
    "third_party",
    "third-party",
}


def beneath(path, directory):
    return directory == "." or path == directory or path.startswith(directory + "/")


def manifest_paths(paths):
    return [
        path
        for path in paths
        if PurePosixPath(path).name in {"pyproject.toml", "package.json"}
        and not any(
            part.casefold() in NON_PROJECT_MANIFEST_DIRECTORIES
            for part in PurePosixPath(path).parts[:-1]
        )
    ]


def instruction_paths(paths):
    # Discovery is already constrained by the evidence catalog. An instruction
    # in tests/examples is conditional on that directory, never global policy.
    return [
        path
        for path in paths
        if PurePosixPath(path).name.casefold() in {"agents.md", "claude.md"}
    ]


def instruction_directory(item):
    if (
        item.generator is not None
        and item.generator.name == "repository-facts"
        and item.id.endswith("/agent-guidance")
        and item.evidence
        and item.evidence[0].kind == "file"
        and instruction_paths([item.evidence[0].path])
    ):
        return str(PurePosixPath(item.evidence[0].path).parent)
    return None


class InstructionIndexIncomplete(ValueError):
    pass


class ApplicationBoundaryError(ValueError):
    pass


def applicable_instructions(paths, targets, broad_targets=()):
    return [
        path
        for path in paths
        if any(beneath(target, str(PurePosixPath(path).parent)) for target in targets)
        or any(
            beneath(str(PurePosixPath(path).parent), target) for target in broad_targets
        )
    ]


def peer_item(item, components, application, package_roots):
    """Containment metadata, not a guessed runtime dependency or responsibility."""
    return peer_paths(
        [path for key in item.component_ids for path in components[key].paths],
        application,
        package_roots,
    )


def peer_paths(scopes, application, package_roots):
    if application is None or application == ".":
        return False
    if (
        not scopes
        or "." in scopes
        or any(beneath(path, application) for path in scopes)
    ):
        return False
    if any(beneath(application, path) for path in scopes):
        return False
    return any(
        beneath(path, boundary)
        for path in scopes
        for boundary in package_roots
        if boundary != "."
        and not beneath(application, boundary)
        and not beneath(boundary, application)
    )

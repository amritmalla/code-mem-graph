"""A source bundle works before the first commit and keeps local records private."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


def test_unborn_repository_bundle_excludes_plans_even_when_force_added(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("source bundling requires Git")
    source = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("prepare_release", source / "scripts/prepare_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repository = tmp_path / "new repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    for name in module.FILES:
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    code = repository / "src/code_mem_graph/cli.py"
    code.parent.mkdir(parents=True)
    code.write_text("# Uncommitted source fixture\n", encoding="utf-8")
    excluded = [".local/private-plan.md", ".mcp.json", "docs/product/roadmap.md",
                "docs/implementation-progress.md", "evaluation/studies/draft.json"]
    for name in excluded:
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("local-record-only", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "--force", *excluded], check=True)
    output = tmp_path / "bundle"
    receipt = module.prepare(repository, output)
    assert receipt["repository_head"] is None
    assert receipt["repository_origin"] is None
    with zipfile.ZipFile(output / "code-mem-graph-source-candidate.zip") as archive:
        assert archive.read("code-mem-graph/src/code_mem_graph/cli.py") == code.read_bytes()
        assert archive.read("code-mem-graph/LICENSE") == (source / "LICENSE").read_bytes()
        assert all("code-mem-graph/" + name not in archive.namelist() for name in excluded)
        assert b"no commits yet" in archive.read("code-mem-graph/SOURCE_PROVENANCE.md")
    saved = (output / "manifest.json").read_bytes()
    assert json.loads(saved)["published"] is False
    with pytest.raises(FileExistsError):
        module.prepare(repository, output)
    assert (output / "manifest.json").read_bytes() == saved

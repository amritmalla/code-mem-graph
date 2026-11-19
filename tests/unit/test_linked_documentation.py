import json

from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.knowledge.freshness import assess_map
from code_mem_graph.knowledge.orient import OrientRequest, orientation
from code_mem_graph.knowledge.sources import capture_file


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_configuration_link_is_not_a_test_source_example(tmp_path):
    write(tmp_path, "README.md", "A project.\n\n[Testing](docs/testing.md)\n")
    write(
        tmp_path,
        "docs/testing.md",
        "# Testing\n\n[Configuration](../tests/pyproject.toml)\n",
    )
    write(
        tmp_path,
        "tests/pyproject.toml",
        '[tool.pytest.ini_options]\ntestpaths=["tests"]\n',
    )
    document = build_repository(tmp_path, repository_id="test-configuration")
    assert "validation" not in records(document, "docs/testing.md")[0].onboarding


def records(document, path):
    return [
        item
        for item in document.generated.items
        if item.id.startswith("docs/") and item.evidence[0].path == path
    ]


def test_linked_project_standards_preserve_practices_and_indented_examples(tmp_path):
    write(tmp_path, "README.md", "An application.\n\n[Standards](docs/standards.md)\n")
    write(
        tmp_path,
        "docs/standards.md",
        "# ⚙️ Project Standards\n\nUse feature-local types.\n\n"
        "## File Naming Conventions\n\nKeep related modules together.\n\n"
        "    feature-name.ts\n    feature-name.test.ts\n",
    )
    document = build_repository(tmp_path, repository_id="linked-standards")
    guidance = records(document, "docs/standards.md")
    assert len(guidance) == 1 and "conventions" in guidance[0].onboarding
    assert "Use feature-local types" in guidance[0].statement
    assert "\n    feature-name.ts\n    feature-name.test.ts" in guidance[0].statement


def test_manifest_local_readme_and_linked_decorated_architecture_are_cited_and_scoped(
    tmp_path,
):
    write(
        tmp_path,
        "apps/site/package.json",
        json.dumps({"scripts": {"dev": "do not execute"}}),
    )
    write(
        tmp_path,
        "apps/site/README.md",
        "# Site\n\n## Get Started\n\nPrerequisites: Node 20 and Yarn.\n"
        "```bash\nyarn install\nyarn dev\n```\n\n[Structure](docs/structure.md)\n",
    )
    write(
        tmp_path,
        "apps/site/docs/structure.md",
        "# 🗄️ **Project Structure**:\n\n`src/handler.ts` handles page requests.\n",
    )
    write(tmp_path, "apps/site/src/handler.ts", "export const handler = () => true;\n")
    document = build_repository(tmp_path, repository_id="nested-docs")
    setup = next(
        item
        for item in records(document, "apps/site/README.md")
        if "connection" in item.onboarding
    )
    assert "prerequisites" in setup.onboarding and "yarn dev" in setup.statement
    architecture = records(document, "apps/site/docs/structure.md")[0]
    component = next(
        component.id
        for component in document.generated.components
        if component.paths == ["apps/site"]
    )
    assert (
        architecture.component_ids == [component]
        and architecture.topic == "architecture"
    )
    assert "responsibilities" in architecture.onboarding
    assert {reference.path for reference in architecture.evidence} == {
        "apps/site/docs/structure.md",
        "apps/site/src/handler.ts",
    }
    for reference in architecture.evidence:
        assert (
            capture_file(
                tmp_path,
                reference.path,
                start_line=reference.start_line,
                end_line=reference.end_line,
            )
            == reference
        )


def test_directory_diagrams_are_architecture_evidence_but_not_setup_commands(tmp_path):
    write(tmp_path, "README.md", "# Example\n\n[Guide](docs/guide.md)\n")
    write(
        tmp_path,
        "docs/guide.md",
        "# Project Structure\n\n```sh\nsrc\n|\n+-- features # feature modules\n```\n"
        "\n## Quick Start\n\n```sh\nsrc\n├── app/\n└── tests/\n```\n",
    )
    document = build_repository(tmp_path, repository_id="diagrams")
    assert any(
        item.topic == "architecture" and "+-- features" in item.statement
        for item in document.items
    )
    assert not any(
        "connection" in item.onboarding or "validation" in item.onboarding
        for item in document.items
    )


def test_parent_conditions_and_working_directory_travel_with_nested_check(tmp_path):
    write(tmp_path, "README.md", "# Example project\n\n[Guide](docs/checks.md)\n")
    write(
        tmp_path,
        "docs/checks.md",
        "# Development\n\nFor the PostgreSQL variant only. Start the database first.\n"
        "\n## Backend commands\n\nRun from `services/api`.\n\n### Verification\n\n"
        "```console\npytest tests\n```\n",
    )
    document = build_repository(tmp_path, repository_id="preconditions")
    check = next(item for item in document.items if "validation" in item.onboarding)
    assert (
        "PostgreSQL variant only" in check.statement
        and "Start the database first" in check.statement
    )
    assert (
        "Run from `services/api`" in check.statement
        and "pytest tests" in check.statement
    )
    assert "Development > Backend commands > Verification" in check.title
    assert len(check.evidence) == 3
    for reference in check.evidence:
        assert (
            capture_file(
                tmp_path,
                reference.path,
                start_line=reference.start_line,
                end_line=reference.end_line,
            )
            == reference
        )


def test_linked_testing_chapter_uses_real_example_links_without_inventing_commands(
    tmp_path,
):
    write(
        tmp_path, "README.md", "A sample application.\n\n[Testing](docs/testing.md)\n"
    )
    write(
        tmp_path,
        "docs/testing.md",
        "# 🧪 [Testing](https://example.invalid):\n\nPrefer behavior-focused testing.\n"
        "\n## Unit Tests\n\n[Unit test example](../apps/site/src/widget.test.ts)\n"
        "\n## Integration Tests\n\n[Integration test example](../apps/site/tests/workflow.ts)\n",
    )
    write(tmp_path, "apps/site/src/widget.test.ts", "export const testCase = true;\n")
    write(
        tmp_path,
        "apps/site/tests/workflow.ts",
        "export const integrationCase = true;\n",
    )
    document = build_repository(tmp_path, repository_id="test-guide")
    guidance = records(document, "docs/testing.md")
    assert len(guidance) == 1 and "validation" in guidance[0].onboarding
    assert (
        "Unit Tests" in guidance[0].statement
        and "Integration Tests" in guidance[0].statement
    )
    assert (
        "yarn test" not in guidance[0].statement
        and "pytest" not in guidance[0].statement
    )
    assert {reference.path for reference in guidance[0].evidence} == {
        "docs/testing.md",
        "apps/site/src/widget.test.ts",
        "apps/site/tests/workflow.ts",
    }


def test_example_link_removal_invalidates_validation_knowledge_before_rebuild(tmp_path):
    write(tmp_path, "README.md", "A project.\n\n[Testing](docs/testing.md)\n")
    write(
        tmp_path,
        "docs/testing.md",
        "# Testing\n\n[Behavior test example](../tests/test_behavior.py)\n",
    )
    write(tmp_path, "tests/test_behavior.py", "def test_behavior(): pass\n")
    document = build_repository(tmp_path, repository_id="stale-example")
    guidance = records(document, "docs/testing.md")[0]
    (tmp_path / "tests/test_behavior.py").unlink()
    assessment = next(
        entry
        for entry in assess_map(document, tmp_path)
        if entry.item_id == guidance.id
    )
    assert assessment.freshness == "stale"
    response = orientation(
        tmp_path, OrientRequest(budget=20000, encoding="cl100k_base")
    )
    assert response["coverage"]["onboarding"]["validation"]["status"] == "unavailable"
    updated = build_repository(tmp_path)
    assert "validation" not in records(updated, "docs/testing.md")[0].onboarding


def test_ambiguous_inline_source_path_does_not_guess_between_root_and_app(tmp_path):
    write(tmp_path, "apps/site/package.json", "{}")
    write(
        tmp_path, "apps/site/README.md", "A site.\n\n[Structure](docs/structure.md)\n"
    )
    write(
        tmp_path,
        "apps/site/docs/structure.md",
        "# Code Structure\n\n`src/handler.ts` handles requests.\n",
    )
    for path in ("src/handler.ts", "apps/site/src/handler.ts"):
        write(tmp_path, path, "export const handler = true;\n")
    document = build_repository(tmp_path, repository_id="ambiguous-path")
    item = records(document, "apps/site/docs/structure.md")[0]
    assert "responsibilities" not in item.onboarding
    assert all(reference.path.endswith("structure.md") for reference in item.evidence)


def test_client_generation_preserves_alternatives_conditions_and_source_ranges(
    tmp_path,
):
    write(tmp_path, "README.md", "An application.\n\n[Frontend](frontend/README.md)\n")
    write(
        tmp_path,
        "frontend/README.md",
        "# Frontend\n\n## Generate Client\n\n"
        "Regenerate whenever the API schema changes.\n\n### Automatically\n\n"
        "From the repository root: `bun run generate-client`.\n\n### Manually\n\n"
        "Start the backend and export its schema before running the generator.\n",
    )
    document = build_repository(tmp_path, repository_id="client-workflow")
    item = records(document, "frontend/README.md")[0]
    assert item.topic == "workflow" and "workflow" in item.facets
    for text in (
        "whenever the API schema changes",
        "Automatically",
        "Manually",
        "Start the backend",
    ):
        assert text in item.statement
    assert "validation" not in item.onboarding


def test_agent_instructions_remain_whole_and_mandatory_at_the_selected_root(
    tmp_path,
):
    root_text = "# Agent instructions\n\nAlways inspect local policies.\n\n## Conventions\n\nUse shared conventions.\n"
    local_text = "# Local instructions\n\nDo not run deployment commands.\n\n## Testing\n\n```bash\ncustom-check\n```\n"
    write(tmp_path, "AGENTS.md", root_text)
    write(tmp_path, "apps/site/package.json", "{}")
    write(tmp_path, "apps/site/CLAUDE.md", local_text)
    document = build_repository(tmp_path, repository_id="mandatory-docs")
    guides = [item for item in document.items if item.id.endswith("/agent-guidance")]
    assert len(guides) == 2 and sum(item.critical for item in guides) == 1
    assert any(
        root_text.rstrip() in item.statement.replace("\r\n", "\n")
        and item.component_ids == ["project"]
        for item in guides
    )
    # Local guidance is stored conditionally; the selected context promotes
    # applicable records to required groups. An application root is mandatory.
    assert any(
        not item.critical and item.evidence[0].path == "apps/site/CLAUDE.md"
        for item in guides
    )
    local_document = build_repository(tmp_path / "apps/site", repository_id="site-root")
    local_guides = [
        item for item in local_document.items if item.id.endswith("/agent-guidance")
    ]
    assert len(local_guides) == 1 and local_guides[0].critical
    assert any(
        local_text.rstrip() in item.statement.replace("\r\n", "\n")
        and item.component_ids == ["project"]
        for item in local_guides
    )


def test_linked_example_agent_file_is_not_promoted_to_mandatory_repository_policy(
    tmp_path,
):
    write(
        tmp_path, "README.md", "A repository.\n\n[Example](examples/demo/AGENTS.md)\n"
    )
    write(tmp_path, "AGENTS.md", "Follow the real project policy.\n")
    write(
        tmp_path,
        "examples/demo/AGENTS.md",
        "# Example only\n\nThis is an illustrative configuration.\n"
        "\n## Conventions\n\nUse this example convention.\n",
    )
    document = build_repository(tmp_path, repository_id="example-instructions")
    assert not any(
        item.critical and item.evidence[0].path == "examples/demo/AGENTS.md"
        for item in document.items
    )
    example = records(document, "examples/demo/AGENTS.md")[0]
    assert "illustrative configuration" in example.statement


def test_manifest_local_readme_cannot_follow_remote_outside_or_ignored_docs(tmp_path):
    write(tmp_path, "apps/site/package.json", "{}")
    write(tmp_path, ".knowledgeignore", "private/\n")
    write(
        tmp_path,
        "apps/site/README.md",
        "# Site\n\n[Private](../../private/guide.md)\n"
        "[Remote](https://example.invalid/guide.md)\n[Outside](../../../outside.md)\n",
    )
    write(
        tmp_path, "private/guide.md", "# Get Started\n\n```bash\nPRIVATE_COMMAND\n```\n"
    )
    document = build_repository(tmp_path, repository_id="link-boundaries")
    assert "PRIVATE_COMMAND" not in document.model_dump_json()
    assert not any(
        item.evidence[0].path == "private/guide.md" for item in document.items
    )

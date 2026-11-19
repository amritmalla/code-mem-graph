import pytest
from pydantic import ValidationError

from code_mem_graph.knowledge.build import build_repository, load_repository_map
from code_mem_graph.knowledge.compact import expand_bundle
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.freshness import assess_map
from code_mem_graph.knowledge.models import CommandCheck
from code_mem_graph.knowledge.orient import mcp_result
from code_mem_graph.tokens import PayloadCounter


def write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    for path, content in {
        "package.json": '{"scripts":{"prepare":"never-run"}}',
        "AGENTS.md": "# Setup Commands\n\n```bash\ncd apps/web # or apps/peer\nyarn test\nyarn test:e2e\n```\n",
        "apps/web/package.json": '{"scripts":{"test":"never-run", "test-e2e":"never-run", "lint":"never-run"}}',
        "apps/peer/package.json": '{"scripts":{"test:e2e":"never-run"}}',
        "apps/web/widget.ts": "export const widget = true;\n",
        "apps/peer/widget.ts": "export const widget = true;\n",
    }.items():
        write(tmp_path, path, content)
    build_repository(tmp_path, repository_id="commands")
    return tmp_path


def comparisons(document):
    return [
        item
        for item in document.items
        if item.command_checks and item.lifecycle == "active"
    ]


def context(project, **changes):
    return task_context(
        project,
        ContextRequest(
            **{
                "task": "widget checks",
                "application": "apps/web",
                "budget": 100000,
                "encoding": "cl100k_base",
                **changes,
            }
        ),
    )


def test_literal_cd_assigns_only_the_named_application_and_keeps_both_sources(project):
    items = comparisons(load_repository_map(project))
    missing = next(
        item for item in items if item.command_checks[0].status == "script_not_declared"
    )
    check = missing.command_checks[0]
    assert check.directory == "apps/web" and check.scope_basis == "literal_cd"
    assert check.similar_names == ["test-e2e"] and missing.critical
    assert [ref.path for ref in missing.evidence] == [
        "AGENTS.md",
        "apps/web/package.json",
    ]
    assert missing.evidence[0].start_line == 4 and missing.evidence[0].end_line == 6
    assert all(item.command_checks[0].directory != "apps/peer" for item in items)
    payload = context(project)
    assert (
        missing.id
        in payload["coverage"]["documented_commands"]["script_not_declared_ids"]
    )
    assert missing.id in {item["id"] for item in payload["items"]}
    assert payload["conflicts"] == []


def test_shared_document_scope_is_unknown_instead_of_comparing_all_children(project):
    write(project, "AGENTS.md", "Use `yarn test:e2e` for checks.\n")
    items = comparisons(build_repository(project))
    assert len(items) == 1
    assert items[0].command_checks[0].status == "scope_unknown"
    assert items[0].command_checks[0].manifest is None and not items[0].critical


def test_app_document_directory_defaults_are_explicit_and_peer_checks_stay_out(project):
    write(
        project,
        "apps/web/README.md",
        "# Testing\n\nUse `npm run lint` and `pnpm run test:e2e`.\n",
    )
    build_repository(project)
    items = comparisons(load_repository_map(project))
    local = [
        i.command_checks[0]
        for i in items
        if i.command_checks[0].document.endswith("README.md")
    ]
    assert {check.scope_basis for check in local} == {"document_directory"}
    peer = context(project, application="apps/peer")
    assert not any(
        i.get("command_checks") and i["command_checks"][0]["directory"] == "apps/web"
        for i in peer["items"]
    )


@pytest.mark.parametrize(
    "command",
    [
        "npm --prefix apps/web run test:e2e",
        "yarn --cwd apps/web test:e2e",
        "pnpm -C apps/web run test:e2e",
        "bun --cwd apps/web run test:e2e",
    ],
)
def test_explicit_directory_options_are_compared_without_execution(project, command):
    write(project, "AGENTS.md", f"```sh\n{command}\n```\n")
    item = comparisons(build_repository(project))[0]
    assert item.command_checks[0].scope_basis == "explicit_directory_option"
    assert item.command_checks[0].status == "script_not_declared"


def test_complex_shell_native_bun_and_binary_commands_do_not_invent_mismatches(project):
    write(
        project,
        "AGENTS.md",
        """```sh
cd apps/web
bun test
yarn playwright test
pnpm exec vitest
npm run --if-present absent
echo 'npm run absent'
```

```typescript
const example = "yarn test:missing";
```
""",
    )
    assert comparisons(build_repository(project)) == []


def test_unsupported_scope_changes_and_fence_boundaries_stay_unknown(project):
    write(
        project,
        "AGENTS.md",
        """```sh
cd apps/web || exit 1
yarn test:e2e
```

```sh
yarn test:e2e
```
""",
    )
    assert all(
        i.command_checks[0].status == "scope_unknown"
        for i in comparisons(build_repository(project))
    )


def test_inline_source_range_cites_the_command_line(project):
    write(
        project,
        "apps/web/README.md",
        "# Testing\n\nA paragraph spanning\nseveral lines with `yarn test:e2e`.\n",
    )
    item = next(
        i
        for i in comparisons(build_repository(project))
        if i.command_checks[0].document.endswith("README.md")
    )
    assert item.evidence[0].start_line == item.evidence[0].end_line == 4


def test_changed_manifest_is_stale_until_refresh_and_fix_changes_comparison(project):
    old = load_repository_map(project)
    missing = next(
        i
        for i in comparisons(old)
        if i.command_checks[0].status == "script_not_declared"
    )
    write(
        project,
        "apps/web/package.json",
        '{"scripts":{"test":"never-run","test:e2e":"never-run"}}',
    )
    assert (
        next(a for a in assess_map(old, project) if a.item_id == missing.id).freshness
        == "stale"
    )
    payload = context(project)
    assert missing.id in payload["coverage"]["documented_commands"]["unavailable_ids"]
    refreshed = build_repository(project)
    check = next(
        i.command_checks[0]
        for i in comparisons(refreshed)
        if i.command_checks[0].script == "test:e2e"
    )
    assert check.status == "declared_script" and check.similar_names == []


@pytest.mark.parametrize("transport", ["cli", "mcp"])
def test_warning_is_mandatory_and_measured_omissions_remain_visible(project, transport):
    seen = []
    request = ContextRequest(
        task="unmatched", application="apps/web", budget=100000, encoding="cl100k_base"
    )
    task_context(project, request, transport=transport, observe=seen.append)
    request = request.model_copy(update={"budget": seen[0]["size"]["tokens"]})
    payload = task_context(project, request, transport=transport)
    assert payload["omitted"]["mandatory_claims"] > 0
    assert payload["coverage"]["documented_commands"]["omitted_ids"]
    wire = (
        mcp_result(payload).model_dump(mode="json", by_alias=True, exclude_none=True)
        if transport == "mcp"
        else payload
    )
    assert (
        PayloadCounter("cl100k_base")(wire)
        == payload["size"]["tokens"]
        <= request.budget
    )


def test_compact_output_roundtrip_and_comparison_evidence_validation(project):
    payload = expand_bundle(
        context(project, representation="compact", detail="focused")
    )
    assert any(i.get("command_checks") for i in payload["items"])
    missing = next(i for i in comparisons(load_repository_map(project)) if i.critical)
    data = missing.model_dump(mode="json")
    data["evidence"] = data["evidence"][:1]
    with pytest.raises(ValidationError, match="manifest evidence"):
        type(missing).model_validate(data)
    with pytest.raises(ValidationError, match="scoped package"):
        CommandCheck(
            document="README.md",
            manager="npm",
            script="test",
            directory="apps/web",
            manifest="package.json",
            status="declared_script",
            scope_basis="literal_cd",
        )
    plain = next(i for i in load_repository_map(project).items if not i.command_checks)
    assert "command_checks" not in plain.model_dump(mode="json")


def test_excluded_manifests_do_not_establish_scope_and_source_commands_are_not_run(
    project,
):
    write(project, ".knowledgeignore", "apps/web/package.json\n")
    items = comparisons(build_repository(project))
    assert all(i.command_checks[0].status == "scope_unknown" for i in items)

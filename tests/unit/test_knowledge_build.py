import json
from pathlib import Path
import shutil

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import BuildError, build_repository, load_repository_map
from code_mem_graph.knowledge.extract import extract_layer
from code_mem_graph.knowledge.freshness import assess_map
from code_mem_graph.knowledge.models import KnowledgeItem, review_item
from code_mem_graph.knowledge.orient import BudgetTooSmall, OrientRequest, mcp_result, orientation
from code_mem_graph.knowledge.sources import EvidenceUnavailable, capture_directory, capture_file, discover_sources
from code_mem_graph.serialization import serialize
from code_mem_graph.tokens import PayloadCounter


@pytest.fixture
def service(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/knowledge/python_service'
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    return tmp_path


def save_map(root, document):
    (root / '.project-knowledge/map.yaml').write_text(
        yaml.safe_dump(document.model_dump(mode='json'), sort_keys=False), encoding='utf-8')


def test_build_is_repeatable_grounded_and_readable(service):
    first = build_repository(service, repository_id='fixture-service')
    first_bytes = (service / '.project-knowledge/map.yaml').read_bytes()
    second = build_repository(service)
    assert first == second
    assert (service / '.project-knowledge/map.yaml').read_bytes() == first_bytes
    assert load_repository_map(service) == first
    assert all(entry.freshness == 'fresh' for entry in assess_map(first, service))
    assert all(item.evidence and item.basis in {'observed', 'documented'} for item in first.items)
    brief = (service / '.project-knowledge/README.md').read_text(encoding='utf-8')
    assert 'Source:' in brief and 'Gaps and conflicts' in brief
    assert any(gap.id == 'orientation/entrypoints' and gap.status == 'open' for gap in first.generated.gaps)


def test_repository_identity_must_be_explicit_and_stable(service):
    with pytest.raises(BuildError, match='explicit repository ID'):
        build_repository(service)
    build_repository(service, repository_id='fixture-service')
    with pytest.raises(BuildError, match='differs'):
        build_repository(service, repository_id='another-project')


def test_build_preserves_maintainer_claims_reviews_and_retires_removed_facts(service):
    document = build_repository(service, repository_id='fixture-service')
    data = document.model_dump(mode='json')
    curated = review_item(KnowledgeItem(id='decision/explanation', topic='decision', title='Rationale',
        statement='A maintainer explanation that generation cannot replace.', origin='maintainer',
        basis='maintainer', author='fixture-maintainer'), reviewer='fixture-reviewer', status='accepted')
    data['curated']['items'] = [curated.model_dump(mode='json')]
    from code_mem_graph.knowledge.models import KnowledgeMap
    save_map(service, KnowledgeMap.model_validate(data))
    (service / 'pyproject.toml').unlink()
    rebuilt = build_repository(service)
    assert rebuilt.curated.items == [curated]
    assert any(item.lifecycle == 'retired' for item in rebuilt.generated.items)


def test_failed_extraction_retains_prior_map_and_marks_unknown(service):
    previous = build_repository(service, repository_id='fixture-service')
    (service / 'pyproject.toml').write_text('[project\ninvalid', encoding='utf-8')
    with pytest.raises(BuildError, match='prior knowledge was retained'):
        build_repository(service)
    failed = load_repository_map(service)
    assert failed.generated == previous.generated
    assert failed.refresh.status == 'failed'
    assert all(entry.freshness != 'fresh' for entry in assess_map(failed, service))


def test_concurrent_edit_never_overwritten(service, monkeypatch):
    build_repository(service, repository_id='fixture-service')
    original = extract_layer
    edited_bytes = None

    def editing(root):
        nonlocal edited_bytes
        path = root / '.project-knowledge/map.yaml'
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        data['curated']['items'].append(KnowledgeItem(id='decision/concurrent', topic='decision', title='Concurrent',
            statement='An edit made during generation.', origin='maintainer', basis='maintainer', author='fixture').model_dump(mode='json'))
        edited_bytes = yaml.safe_dump(data).encode('utf-8')
        path.write_bytes(edited_bytes)
        return original(root)

    monkeypatch.setattr('code_mem_graph.knowledge.build.extract_layer', editing)
    with pytest.raises(BuildError, match='changed during'):
        build_repository(service)
    assert (service / '.project-knowledge/map.yaml').read_bytes() == edited_bytes


def test_lock_and_failed_atomic_replace_preserve_map(service, monkeypatch):
    build_repository(service, repository_id='fixture-service')
    lock = service / '.project-knowledge/.build.lock'
    lock.write_text('held', encoding='utf-8')
    with pytest.raises(BuildError, match='another build'):
        build_repository(service)
    lock.unlink()
    before = (service / '.project-knowledge/map.yaml').read_bytes()

    def denied(*args):
        raise PermissionError('simulated publication failure')

    monkeypatch.setattr('code_mem_graph.knowledge.build.os.replace', denied)
    with pytest.raises(PermissionError):
        build_repository(service)
    assert (service / '.project-knowledge/map.yaml').read_bytes() == before
    assert not lock.exists()
    assert not list((service / '.project-knowledge').glob('.write-*'))


def test_exclusions_nested_negation_custom_rules_and_new_ignores(service):
    (service / '.gitignore').write_text('private/\n*.py\n!src/\n', encoding='utf-8')
    (service / 'src/.gitignore').write_text('!parcel/app.py\n', encoding='utf-8')
    (service / 'private').mkdir()
    (service / 'private/passwords.md').write_text('private content', encoding='utf-8')
    paths = discover_sources(service)
    assert 'src/parcel/app.py' in paths
    assert 'private/passwords.md' not in paths
    before = capture_file(service, 'src/parcel/app.py')
    directory = capture_directory(service, '.', recursive=True)
    (service / '.knowledgeignore').write_text('src/parcel/app.py\n', encoding='utf-8')
    assert 'src/parcel/app.py' not in discover_sources(service)
    assert capture_directory(service, '.', recursive=True).digest != directory.digest
    with pytest.raises(EvidenceUnavailable, match='repository_excluded'):
        capture_file(service, before.path)


def test_cannot_reinclude_pruned_parent_or_read_linked_ignore(service):
    (service / '.gitignore').write_text('src/\n!src/parcel/app.py\n', encoding='utf-8')
    assert 'src/parcel/app.py' not in discover_sources(service)
    (service / '.gitignore').unlink()
    secret = service / 'private-ignore'
    secret.write_text('*', encoding='utf-8')
    try:
        (service / '.gitignore').symlink_to(secret)
    except OSError:
        pytest.skip('symlink privilege unavailable')
    with pytest.raises(EvidenceUnavailable, match='unsafe_link'):
        discover_sources(service)


def test_extraction_keeps_config_secrets_and_command_bodies_out(service):
    (service / 'package.json').write_text(json.dumps({
        'name': 'fixture', 'engines': {'node': '>=22'}, 'main': 'src/main.ts',
        'scripts': {'test': 'SECRET=do-not-copy node test.js'},
        'dependencies': {'library': 'https://name:secret@private.example/repo'},
        'privateToken': 'do-not-copy-private-token',
    }), encoding='utf-8')
    layer = extract_layer(service)
    text = layer.model_dump_json()
    assert 'do-not-copy' not in text and 'name:secret' not in text
    assert any('script test' in item.statement for item in layer.items)
    assert any('library' in item.statement for item in layer.items)
    assert any('src/main.ts' in item.statement for item in layer.items)


def test_readme_manifest_conflict_is_visible_and_not_guidance(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/knowledge/typescript_app'
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    document = build_repository(tmp_path, repository_id='web-fixture')
    assert document.generated.conflicts
    payload = orientation(tmp_path, OrientRequest(budget=20000, encoding='cl100k_base'))
    assert payload['conflicts']
    assert not any(item['conflict_ids'] for item in payload['items'])
    assert any(item['conflict_ids'] for item in payload['cautions'])
    assert payload['coverage']['missing_evidence']  # No declared entry point in this fixture.


@pytest.mark.parametrize('transport', ['cli', 'mcp'])
@pytest.mark.parametrize('budget', [1000, 2000, 5000, 20000])
def test_complete_payload_and_envelope_always_fit(service, transport, budget):
    build_repository(service, repository_id='fixture-service')
    request = OrientRequest(budget=budget, encoding='cl100k_base')
    result = orientation(service, request, transport=transport)
    measured = (mcp_result(result).model_dump(mode='json', by_alias=True, exclude_none=True)
                if transport == 'mcp' else result)
    assert PayloadCounter(request.encoding)(measured) == result['size']['tokens'] <= budget
    assert result['policy']['calibrated'] is False
    assert all(item['freshness'] == 'fresh' and not item['review_required'] for item in result['items'])


def test_tiny_budget_staleness_and_unknown_knowledge(service):
    build_repository(service, repository_id='fixture-service')
    with pytest.raises(BudgetTooSmall):
        orientation(service, OrientRequest(budget=1, encoding='cl100k_base'))
    (service / 'README.md').write_text('Changed introduction.', encoding='utf-8')
    payload = orientation(service, OrientRequest(budget=20000, encoding='cl100k_base'))
    assert payload['source_status']['stale'] >= 1
    assert not any(item['id'] == 'project/purpose' for item in payload['items'])
    assert any(item['id'] == 'project/purpose' for item in payload['cautions'])


def test_mandatory_claims_cannot_be_displaced_by_optional_facts(service):
    document = build_repository(service, repository_id='fixture-service')
    data = document.model_dump(mode='json')
    data['curated']['items'] = [KnowledgeItem(id='decision/critical', topic='decision', title='Critical guidance',
        statement='critical maintainer context ' * 3000, origin='maintainer', basis='maintainer', author='fixture',
        critical=True).model_dump(mode='json')]
    from code_mem_graph.knowledge.models import KnowledgeMap
    save_map(service, KnowledgeMap.model_validate(data))
    result = orientation(service, OrientRequest(budget=1000, encoding='cl100k_base'))
    assert result['omitted']['mandatory_claims'] == 1
    assert result['items'] == [] and result['status'] == 'partial'
    assert result['knowledge_ref']['path'] == '.project-knowledge/map.yaml'


def test_cli_build_and_orientation_require_explicit_sizing(service):
    runner = CliRunner()
    built = runner.invoke(main, ['knowledge', 'build', '--root', str(service), '--repository-id', 'fixture-service'])
    assert built.exit_code == 0, built.output
    args = ['orient', '--root', str(service)]
    assert runner.invoke(main, args).exit_code != 0
    result = runner.invoke(main, args + ['--budget', '2000', '--encoding', 'cl100k_base'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert result.output == serialize(payload)
    assert payload['size']['tokens'] == PayloadCounter('cl100k_base')(payload)


def test_source_added_during_discovery_aborts_inconsistent_counts(service, monkeypatch):
    from code_mem_graph.knowledge.sources import discover_snapshot

    def changing(root):
        snapshot = discover_snapshot(root)
        (root / 'new.py').write_text('# added during extraction', encoding='utf-8')
        return snapshot

    monkeypatch.setattr('code_mem_graph.knowledge.extract.discover_snapshot', changing)
    with pytest.raises(BuildError, match='source evidence changed'):
        build_repository(service, repository_id='fixture-service')
    assert not (service / '.project-knowledge/map.yaml').exists()


def test_agent_instructions_are_cited_mandatory_repository_evidence(service):
    (service / 'AGENTS.md').write_text('Use the existing test conventions.\n', encoding='utf-8')
    document = build_repository(service, repository_id='fixture-service')
    guidance = next(item for item in document.items if item.title.endswith('AGENTS.md'))
    assert guidance.critical and guidance.basis == 'documented'
    payload = orientation(service, OrientRequest(budget=20000, encoding='cl100k_base'))
    assert any(item['id'] == guidance.id for item in payload['items'])


def test_reading_view_rebases_links_without_changing_source_or_code(service):
    (service / 'README.md').write_text('A fictional [service](docs/guide.md).\n', encoding='utf-8')
    (service / 'docs').mkdir()
    (service / 'docs/guide.md').write_text('A guide.', encoding='utf-8')
    (service / 'CONTRIBUTING.md').write_text(
        '# Conventions\n\nSee [the guide](docs/guide.md).\n\n```text\n[literal](unchanged.md)\n```\n', encoding='utf-8')
    document = build_repository(service, repository_id='fixture-service')
    brief = (service / '.project-knowledge/README.md').read_text(encoding='utf-8')
    assert '[service](../docs/guide.md)' in brief
    assert '[the guide](../docs/guide.md)' in brief
    assert '[literal](unchanged.md)' in brief
    assert any('[service](docs/guide.md)' in item.statement for item in document.items)


def test_reading_view_failure_does_not_undo_committed_map(service, monkeypatch):
    prior = build_repository(service, repository_id='fixture-service')
    view = (service / '.project-knowledge/README.md').read_bytes()
    (service / 'README.md').write_text('A revised project introduction.', encoding='utf-8')
    from code_mem_graph.knowledge.build import _atomic_write

    def fail_view(root, name, text, *, expected):
        if name == 'README.md':
            raise PermissionError('reading view could not be written')
        return _atomic_write(root, name, text, expected=expected)

    monkeypatch.setattr('code_mem_graph.knowledge.build._atomic_write', fail_view)
    with pytest.raises(BuildError, match='map saved'):
        build_repository(service)
    current = load_repository_map(service)
    assert current != prior and current.refresh.status == 'ready'
    assert (service / '.project-knowledge/README.md').read_bytes() == view
    assert all(result.freshness == 'fresh' for result in assess_map(current, service))

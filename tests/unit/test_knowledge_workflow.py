import json
from pathlib import Path
import shutil

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import BuildError, build_repository, load_repository_map
from code_mem_graph.knowledge.models import ItemDependency, KnowledgeItem, KnowledgeMap, review_item
from code_mem_graph.knowledge.sources import capture_file
from code_mem_graph.knowledge.workflow import record_review, refresh_repository, resolve_conflict, show_item, status_report


def persist(root, data):
    document = KnowledgeMap.model_validate(data)
    (root / '.project-knowledge/map.yaml').write_text(yaml.safe_dump(document.model_dump(mode='json')), encoding='utf-8')
    return document


@pytest.fixture
def repository(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/knowledge/python_service'
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    document = build_repository(tmp_path, repository_id='workflow-fixture')
    runtime = next(item for item in document.items if item.id.endswith('/runtime'))
    authored = review_item(KnowledgeItem(id='decision/compatibility', topic='decision', title='Compatibility rationale',
        statement='The fixture supports its declared runtime.', origin='maintainer', basis='inferred', author='fixture',
        evidence=[capture_file(tmp_path, 'src/parcel/app.py')],
        dependencies=[ItemDependency(item_id=runtime.id, content_digest=runtime.content_digest)]),
        reviewer='fixture-reviewer', status='accepted')
    data = document.model_dump(mode='json')
    data['curated']['items'] = [authored.model_dump(mode='json')]
    persist(tmp_path, data)
    return tmp_path


def test_status_and_show_report_versioned_sources_and_dependents(repository):
    report = status_report(repository)
    assert report['source_status']['stale'] == 0
    item = show_item(repository, 'decision/compatibility')
    assert item['assessment']['freshness'] == 'fresh'
    assert item['current_evidence'][0]['current'] == item['current_evidence'][0]['stored']
    dependency = item['item']['dependencies'][0]['item_id']
    assert 'decision/compatibility' in show_item(repository, dependency)['dependents']
    with pytest.raises(ValueError, match='unknown knowledge'):
        show_item(repository, 'missing')


def test_refresh_updates_generated_facts_but_keeps_curated_explanation_and_review(repository):
    prior = load_repository_map(repository)
    path = repository / 'pyproject.toml'
    path.write_text(path.read_text(encoding='utf-8').replace('>=3.11', '>=3.12'), encoding='utf-8')
    assert status_report(repository)['source_status']['stale'] > 0
    result = refresh_repository(repository)
    assert result['changes']['changed'] and result['changes']['curated_preserved']
    current = load_repository_map(repository)
    assert current.curated == prior.curated
    assert 'decision/compatibility' in result['review_queue']
    assert show_item(repository, 'decision/compatibility')['assessment']['freshness'] == 'stale'


@pytest.mark.parametrize('operation', ['add', 'delete', 'rename'])
def test_membership_changes_and_retirement_are_reported(repository, operation):
    path = repository / 'src/parcel/app.py'
    if operation == 'add':
        (path.parent / 'worker.py').write_text('def work(): pass\n', encoding='utf-8')
    elif operation == 'delete':
        path.unlink()
    else:
        path.rename(path.parent / 'renamed.py')
    assert status_report(repository)['source_status']['stale'] > 0
    refreshed = refresh_repository(repository)
    assert refreshed['changes']['changed']
    assert refreshed['changes']['curated_preserved']
    if operation != 'add':
        assert 'decision/compatibility' in refreshed['review_queue']
    manifest = repository / 'pyproject.toml'
    manifest.unlink()
    assert refresh_repository(repository)['changes']['retired']


def test_failed_refresh_retains_previous_material_and_reports_unknown(repository):
    before = load_repository_map(repository)
    (repository / 'pyproject.toml').write_text('invalid !!!', encoding='utf-8')
    with pytest.raises(BuildError):
        refresh_repository(repository)
    failed = load_repository_map(repository)
    assert failed.generated == before.generated and failed.curated == before.curated
    assert status_report(repository)['refresh']['status'] == 'failed'
    assert status_report(repository)['source_status']['unknown'] > 0


def test_version_guard_and_stale_acceptance_never_modify_map(repository):
    inspected = show_item(repository, 'decision/compatibility')
    path = repository / '.project-knowledge/map.yaml'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='version changed'):
        record_review(repository, 'decision/compatibility', expected_digest='0' * 64,
                      reviewer='fixture', status='accepted')
    assert path.read_bytes() == before
    (repository / 'src/parcel/app.py').write_text('# changed', encoding='utf-8')
    with pytest.raises(ValueError, match='current evidence'):
        record_review(repository, 'decision/compatibility', expected_digest=inspected['content_digest'],
                      reviewer='fixture', status='accepted')
    assert path.read_bytes() == before


def test_explicit_curated_recapture_rebinds_review_and_dependencies(repository):
    inspected = show_item(repository, 'decision/compatibility')
    path = repository / 'pyproject.toml'
    path.write_text(path.read_text(encoding='utf-8').replace('>=3.11', '>=3.12'), encoding='utf-8')
    refresh_repository(repository)
    updated = record_review(repository, 'decision/compatibility', expected_digest=inspected['content_digest'],
                            reviewer='fixture-reviewer', status='accepted', refresh_evidence=True)
    current = show_item(repository, 'decision/compatibility')
    assert current['assessment']['freshness'] == 'fresh'
    assert not current['assessment']['review_required']
    assert current['content_digest'] != inspected['content_digest']
    assert updated.curated.items[0].statement == inspected['item']['statement']
    assert updated.curated.items[0].review.content_digest == current['content_digest']


def test_generated_evidence_cannot_be_rebound_by_review_command(repository):
    generated = load_repository_map(repository).generated.items[0]
    with pytest.raises(ValueError, match='rebuild generated facts'):
        record_review(repository, generated.id, expected_digest=generated.content_digest,
                      reviewer='fixture', status='accepted', refresh_evidence=True)
    record_review(repository, generated.id, expected_digest=generated.content_digest, reviewer='fixture', status='accepted')
    assert show_item(repository, generated.id)['item']['review']['status'] == 'accepted'


def test_rejection_is_allowed_for_stale_claims_and_sourcefree_attestation_stays_unknown(repository):
    inspected = show_item(repository, 'decision/compatibility')
    (repository / 'src/parcel/app.py').unlink()
    record_review(repository, 'decision/compatibility', expected_digest=inspected['content_digest'],
                  reviewer='fixture', status='rejected')
    assert show_item(repository, 'decision/compatibility')['item']['review']['status'] == 'rejected'
    document = load_repository_map(repository)
    assertion = KnowledgeItem(id='decision/intent', topic='decision', title='Intent', statement='Fixture intent.',
                              origin='maintainer', basis='maintainer', author='fixture')
    data = document.model_dump(mode='json')
    data['curated']['items'].append(assertion.model_dump(mode='json'))
    persist(repository, data)
    record_review(repository, assertion.id, expected_digest=assertion.content_digest, reviewer='fixture', status='accepted')
    assert show_item(repository, assertion.id)['assessment']['freshness'] == 'unknown'


def test_conflict_resolution_is_versioned_durable_and_invalidated_by_changed_claims(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/knowledge/typescript_app'
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    build_repository(tmp_path, repository_id='web-fixture')
    report = status_report(tmp_path)
    conflict = report['conflicts'][0]
    with pytest.raises(ValueError, match='version changed'):
        resolve_conflict(tmp_path, conflict['id'], expected_map_digest='0' * 64, reviewer='fixture', explanation='Use declared engine.')
    resolved = resolve_conflict(tmp_path, conflict['id'], expected_map_digest=report['map_digest'], reviewer='fixture',
                                explanation='The README is outdated; follow the declared engine and update the docs.')
    assert status_report(tmp_path)['conflicts'][0]['effective_status'] == 'resolved'
    assert resolved.curated.items[0].statement.startswith('The README is outdated')
    refreshed = refresh_repository(tmp_path)
    assert refreshed['changes']['curated_preserved']
    assert status_report(tmp_path)['conflicts'][0]['effective_status'] == 'resolved'
    package = tmp_path / 'package.json'
    package.write_text(package.read_text(encoding='utf-8').replace('>=22', '>=24'), encoding='utf-8')
    assert status_report(tmp_path)['conflicts'][0]['effective_status'] == 'unresolved'
    refreshed = refresh_repository(tmp_path)
    assert refreshed['conflicts'][0]['effective_status'] == 'unresolved'
    assert load_repository_map(tmp_path).curated == resolved.curated


def test_refresh_guard_rejects_a_newer_map(repository, monkeypatch):
    original = build_repository

    def changed(root, **kwargs):
        document = load_repository_map(root)
        data = document.model_dump(mode='json')
        data['refresh'] = {'status': 'failed', 'error': 'concurrent version'}
        persist(root, data)
        return original(root, **kwargs)

    monkeypatch.setattr('code_mem_graph.knowledge.workflow.build_repository', changed)
    with pytest.raises(BuildError, match='changed before refresh'):
        refresh_repository(repository)
    assert load_repository_map(repository).refresh.error == 'concurrent version'


def test_cli_status_refresh_show_and_version_bound_review(repository):
    runner = CliRunner()
    args = ['--root', str(repository)]
    status = runner.invoke(main, ['knowledge', 'status', *args, '--check'])
    assert status.exit_code == 0, status.output
    inspected = runner.invoke(main, ['knowledge', 'show', 'decision/compatibility', *args])
    digest = json.loads(inspected.output)['content_digest']
    reviewed = runner.invoke(main, ['knowledge', 'review', 'decision/compatibility', *args, '--expected-digest', digest,
                                   '--reviewer', 'fixture', '--status', 'accepted'])
    assert reviewed.exit_code == 0, reviewed.output
    (repository / 'src/parcel/app.py').write_text('# changed', encoding='utf-8')
    assert runner.invoke(main, ['knowledge', 'status', *args, '--check']).exit_code == 1
    refreshed = runner.invoke(main, ['knowledge', 'refresh', *args])
    assert refreshed.exit_code == 0, refreshed.output
    assert json.loads(refreshed.output)['changes']['curated_preserved']


def test_generated_input_cannot_assign_conflict_resolution_authority(repository):
    from code_mem_graph.knowledge.models import KnowledgeConflict, KnowledgeLayer
    from code_mem_graph.knowledge.refresh import refresh_generated
    current = load_repository_map(repository)
    first, second = current.generated.items[:2]
    forged = KnowledgeConflict(id='fixture/forged', item_ids=[first.id, second.id], description='Fixture conflict',
        status='resolved', resolution='Generator trying to approve itself.', resolved_by='generator',
        resolved_versions={first.id: first.content_digest, second.id: second.content_digest})
    data = current.generated.model_dump(mode='json')
    data['conflicts'] = [forged.model_dump(mode='json')]
    merged = refresh_generated(KnowledgeMap(schema_version=1, repository_id=current.repository_id), KnowledgeLayer.model_validate(data))
    assert merged.generated.conflicts[0].status == 'unresolved'
    assert merged.generated.conflicts[0].resolved_by is None


def test_rejected_dependencies_block_accepting_a_derived_explanation(repository):
    inspected = show_item(repository, 'decision/compatibility')
    parent = show_item(repository, inspected['item']['dependencies'][0]['item_id'])
    record_review(repository, parent['item']['id'], expected_digest=parent['content_digest'], reviewer='fixture', status='rejected')
    with pytest.raises(ValueError, match='dependencies first'):
        record_review(repository, 'decision/compatibility', expected_digest=inspected['content_digest'],
                      reviewer='fixture', status='accepted')

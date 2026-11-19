import json

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import build_repository, load_repository_map
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.models import (
    ItemDependency, KnowledgeConflict, KnowledgeItem, KnowledgeMap, review_item,
)
from code_mem_graph.knowledge.orient import BudgetTooSmall, OrientRequest, mcp_result, orientation
from code_mem_graph.knowledge.sources import capture_file
from code_mem_graph.tokens import PayloadCounter


def persist(root, data):
    document = KnowledgeMap.model_validate(data)
    (root / '.project-knowledge/map.yaml').write_text(yaml.safe_dump(document.model_dump(mode='json')), encoding='utf-8')
    return document


@pytest.fixture
def repository(tmp_path):
    for path, content in {
        'src/api/health.py': 'def health():\n    return "ok"\n',
        'src/ui/view.ts': 'export const view = "dashboard";\n',
        'tests/test_health.py': 'def test_health():\n    assert True\n',
        'README.md': 'A fictional project.\n',
        'pyproject.toml': '[project]\nname="fixture"\nrequires-python=">=3.11"\n[tool.pytest.ini_options]\ntestpaths=["tests"]\n',
    }.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    document = build_repository(tmp_path, repository_id='context-fixture')
    components = {entry.paths[0]: entry.id for entry in document.generated.components}
    handler = KnowledgeItem(id='api/handler', topic='architecture', title='Health handler',
        statement='The health handler provides the health status.', origin='maintainer', basis='observed',
        author='fixture', component_ids=[components['src/api']], evidence=[capture_file(tmp_path, 'src/api/health.py')])
    test = KnowledgeItem(id='api/regression', topic='workflow', facets=['workflow'], title='Health regression',
        statement='Health regression checks are recorded in tests/test_health.py.', origin='maintainer', basis='observed',
        author='fixture', component_ids=[components['src/api']], evidence=[capture_file(tmp_path, 'tests/test_health.py')])
    rationale = review_item(KnowledgeItem(id='api/rationale', topic='decision', title='Isolation rationale',
        statement='The fixture endpoint should remain dependency-free.', origin='maintainer', basis='inferred', author='fixture',
        dependencies=[ItemDependency(item_id=handler.id, content_digest=handler.content_digest)],
        component_ids=[components['src/api']]), reviewer='fixture-reviewer', status='accepted')
    unrelated = KnowledgeItem(id='ui/view', topic='architecture', title='Dashboard view',
        statement='The dashboard view displays parcel information.', origin='maintainer', basis='observed',
        author='fixture', component_ids=[components['src/ui']], evidence=[capture_file(tmp_path, 'src/ui/view.ts')])
    data = document.model_dump(mode='json')
    data['curated']['items'] = [item.model_dump(mode='json') for item in [handler, test, rationale, unrelated]]
    persist(tmp_path, data)
    return tmp_path


def request(**kwargs):
    return ContextRequest(task='Update health checks', budget=30000, encoding='cl100k_base', **kwargs)


def entries(payload):
    return {entry['id']: entry for key in ('items', 'cautions') for entry in payload[key]}


def test_task_scope_includes_checks_decisions_and_dependency_closure(repository):
    payload = task_context(repository, request())
    included = entries(payload)
    assert {'api/handler', 'api/regression', 'api/rationale'} <= included.keys()
    assert 'ui/view' not in included
    assert payload['scope']['dependency_kind'] == 'knowledge_evidence_not_runtime_call_graph'
    assert any('requires-python' in item['statement'] for item in payload['items'])
    assert any(item['role'] == 'test_candidate' for item in payload['source_candidates'])
    assert payload['coverage']['task_scope']['usable_claims'] > 0


@pytest.mark.parametrize('path', ['src/api/health.py', 'src/api/proposed.py', 'src/api'])
def test_explicit_existing_and_new_paths_resolve_component_scope(repository, path):
    payload = task_context(repository, request(paths=[path]))
    assert 'api/handler' in entries(payload)
    assert 'ui/view' not in entries(payload)
    assert payload['scope']['matched_components'] >= 1
    if path.endswith('proposed.py'):
        assert payload['scope']['unavailable_path_hints'] == 1


def test_unknown_task_does_not_fabricate_semantic_matches(repository):
    payload = task_context(repository, ContextRequest(task='quuxxyz', budget=10000, encoding='cl100k_base'))
    assert payload['scope']['matched_claims'] == 0
    assert payload['coverage']['task_scope']['usable_claims'] == 0
    assert payload['status'] == 'partial'
    assert payload['source_candidates'] == []


def test_explicit_item_and_component_ids(repository):
    document = load_repository_map(repository)
    component_id = next(item.component_ids[0] for item in document.curated.items if item.id == 'api/handler')
    result = task_context(repository, ContextRequest(task='inspect', component_ids=[component_id], budget=30000, encoding='cl100k_base'))
    assert 'api/handler' in entries(result) and 'ui/view' not in entries(result)
    result = task_context(repository, ContextRequest(task='inspect', item_ids=['api/rationale'], budget=30000, encoding='cl100k_base'))
    assert {'api/rationale', 'api/handler'} <= entries(result).keys()
    with pytest.raises(ValueError, match='unknown requested knowledge'):
        task_context(repository, request(item_ids=['missing']))
    with pytest.raises(ValueError, match='unknown requested component'):
        task_context(repository, request(component_ids=['missing']))


@pytest.mark.parametrize('transport', ['cli', 'mcp'])
@pytest.mark.parametrize('budget', [1100, 2500, 5000, 30000])
def test_dependency_groups_and_complete_payload_limits(repository, transport, budget):
    payload = task_context(repository, ContextRequest(task='Isolation rationale', item_ids=['api/rationale'],
        budget=budget, encoding='cl100k_base'), transport=transport)
    included = entries(payload)
    for item in included.values():
        assert all(dep['item_id'] in included for dep in item['dependencies'])
    counted = mcp_result(payload).model_dump(mode='json', by_alias=True, exclude_none=True) if transport == 'mcp' else payload
    assert PayloadCounter('cl100k_base')(counted) == payload['size']['tokens'] <= budget


def test_stale_and_rejected_dependencies_remain_cautions(repository):
    (repository / 'src/api/health.py').write_text('# changed source', encoding='utf-8')
    result = task_context(repository, request())
    assert all(entry['id'] not in {'api/handler', 'api/rationale'} for entry in result['items'])
    assert entries(result)['api/rationale']['freshness'] == 'stale'
    assert entries(result)['api/handler']['review_required'] is False  # Source staleness is independent of review.


def test_unresolved_conflicts_and_resolution_explanations_are_never_separated(repository):
    document = load_repository_map(repository)
    data = document.model_dump(mode='json')
    conflict = KnowledgeConflict(id='fixture/conflict', item_ids=['api/handler', 'ui/view'], description='Competing fixture claims')
    data['curated']['conflicts'] = [conflict.model_dump(mode='json')]
    persist(repository, data)
    result = task_context(repository, request(item_ids=['api/handler']))
    assert {'api/handler', 'ui/view'} <= entries(result).keys()
    assert result['conflicts'][0]['effective_status'] == 'unresolved'
    data['curated']['conflicts'][0].update(status='resolved', resolution='Use each claim in its own component.', resolved_by='fixture',
        resolved_versions={item.id: item.content_digest for item in document.items if item.id in conflict.item_ids})
    persist(repository, data)
    result = task_context(repository, request(item_ids=['api/handler']))
    assert result['conflicts'][0]['resolution'] == 'Use each claim in its own component.'
    assert result['conflicts'][0]['effective_status'] == 'resolved'
    oriented = orientation(repository, OrientRequest(budget=30000, encoding='cl100k_base'))
    assert oriented['conflicts'][0]['resolution'] == result['conflicts'][0]['resolution']


def test_excluded_sources_and_invalid_paths_are_not_read(repository):
    (repository / '.knowledgeignore').write_text('private/\n', encoding='utf-8')
    (repository / 'private').mkdir()
    (repository / 'private/health.py').write_text('SECRET=not-for-context', encoding='utf-8')
    payload = task_context(repository, request(paths=['private/health.py']))
    assert 'not-for-context' not in json.dumps(payload)
    assert not any(entry.get('source', {}).get('path') == 'private/health.py' for entry in payload['source_candidates'])
    with pytest.raises(ValueError):
        ContextRequest(task='inspect', paths=['../outside.py'], budget=5000, encoding='cl100k_base')
    with pytest.raises(BudgetTooSmall):
        task_context(repository, ContextRequest(task='health', budget=1, encoding='cl100k_base'))


def test_cli_context_is_read_only_and_requires_cap(repository):
    path = repository / '.project-knowledge/map.yaml'
    before = path.read_bytes()
    args = ['context', '--root', str(repository), '--task', 'health', '--path', 'src/api/health.py']
    runner = CliRunner()
    assert runner.invoke(main, args).exit_code != 0
    result = runner.invoke(main, args + ['--budget', '5000', '--encoding', 'cl100k_base'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['scope']['matched_claims'] > 0
    assert path.read_bytes() == before


def test_unknown_inventory_keeps_available_saved_knowledge_explicitly_partial(repository, monkeypatch):
    def denied(root):
        from code_mem_graph.knowledge.sources import EvidenceUnavailable
        raise EvidenceUnavailable('unreadable', 'unrelated')
    monkeypatch.setattr('code_mem_graph.knowledge.context.discover_sources', denied)
    result = task_context(repository, request(item_ids=['api/handler']))
    assert 'api/handler' in entries(result)
    assert result['status'] == 'partial'
    assert result['scope']['inventory_status'] == 'unknown'


def test_overlapping_component_constraints_are_not_lost_to_a_path_tie(repository):
    document = load_repository_map(repository)
    data = document.model_dump(mode='json')
    data['curated']['components'] = [{'id': 'crosscutting/security', 'name': 'Security policy', 'paths': ['src/api'],
                                      'parent_id': 'project', 'lifecycle': 'active'}]
    constraint = KnowledgeItem(id='crosscutting/rule', topic='conventions', facets=['constraints'],
        title='Crosscutting fixture rule', statement='Inspect the cited policy before changing this source area.',
        origin='maintainer', basis='documented', author='fixture', critical=True,
        component_ids=['crosscutting/security'], evidence=[capture_file(repository, 'README.md')])
    data['curated']['items'].append(constraint.model_dump(mode='json'))
    persist(repository, data)
    result = task_context(repository, request(paths=['src/api/health.py']))
    assert 'crosscutting/rule' in entries(result)

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.models import KnowledgeItem, KnowledgeMap
from code_mem_graph.knowledge.orient import OrientRequest, mcp_result, orientation
from code_mem_graph.knowledge.sources import capture_file
from code_mem_graph.knowledge.workflow import record_review
from code_mem_graph.tokens import PayloadCounter

FIXTURES = Path(__file__).parents[1]/'fixtures'/'onboarding'


def repository(tmp_path,name):
    root = tmp_path/name
    shutil.copytree(FIXTURES/name,root)
    build_repository(root,repository_id=name)
    return root


def orient(root,transport='cli'):
    return orientation(root,OrientRequest(budget=30000,encoding='cl100k_base'),transport=transport)


def statements(payload):
    return '\n'.join(item['statement'] for item in payload['items'])


def test_upstream_bytes_and_attribution_are_pinned():
    manifest = json.loads((FIXTURES/'UPSTREAM.json').read_text(encoding='utf-8'))
    assert len({entry['repository'] for entry in manifest['repositories']})==2
    for repo in manifest['repositories']:
        assert len(repo['revision'])==40
        assert 'MIT License' in (FIXTURES/repo['license_path']).read_text(encoding='utf-8')
        for entry in repo['files']:
            data = (FIXTURES/repo['fixture']/entry['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==entry['sha256']
            assert len(data)==entry['bytes'] and repo['revision'] in entry['url']


@pytest.mark.parametrize('transport',['cli','mcp'])
def test_service_linked_docs_retain_setup_roles_migration_and_check_prerequisites(tmp_path,transport):
    root = repository(tmp_path,'fastapi-service')
    result = orient(root,transport)
    coverage = result['coverage']['onboarding']
    assert all(coverage[key]['status']=='included' for key in
               ('connection','prerequisites','responsibilities','conventions','validation'))
    assert coverage['purpose']['status']=='missing_evidence'  # Navigation is not an introduction.
    text = statements(result)
    for expected in ('docker compose up -d db mailpit','uv sync','uv run fastapi dev','uv run bash scripts/test.sh',
                     './backend/app/models.py','./backend/app/api/','./backend/tests/',
                     'uv run alembic revision','uv run alembic upgrade head'):
        assert expected in text
    checks = [item for item in result['items'] if 'validation' in item.get('onboarding',[])]
    assert all(item['basis']=='documented' for item in checks)
    assert any('backend/README.md'==ref['path'] and ref['start_line'] for item in checks for ref in item['evidence'])
    assert any(gap['id']=='knowledge/release-approvers' for gap in result['gaps'])


def test_real_frontend_script_declarations_do_not_invent_setup_or_tests(tmp_path):
    root = repository(tmp_path,'vite-frontend')
    result = orient(root)
    coverage = result['coverage']['onboarding']
    assert coverage['purpose']['status']==coverage['conventions']['status']=='included'
    assert all(coverage[key]['status']=='missing_evidence' for key in
               ('connection','prerequisites','responsibilities','validation'))
    assert result['status']=='partial'
    assert 'type-aware' in statements(result)
    assert 'pnpm install' not in statements(result)


@pytest.mark.parametrize('transport',['cli','mcp'])
def test_linked_ui_inline_commands_tables_and_full_payload_bounds(tmp_path,transport):
    root = repository(tmp_path,'documented-ui')
    result = orient(root,transport)
    assert all(entry['status']=='included' for entry in result['coverage']['onboarding'].values())
    text = statements(result)
    assert all(expected in text for expected in ('pnpm install','pnpm dev','Node.js 22','pnpm 10',
        'src/pricing/PriceCard.tsx','PriceCard.spec.tsx','pnpm test -- PriceCard.spec.tsx','pnpm typecheck','pnpm lint'))
    bounded = orientation(root,OrientRequest(budget=result['size']['tokens'],encoding='cl100k_base'),transport=transport)
    wire = mcp_result(bounded).model_dump(mode='json',by_alias=True,exclude_none=True) if transport=='mcp' else bounded
    assert PayloadCounter('cl100k_base')(wire)==bounded['size']['tokens']<=result['size']['tokens']


def test_sparse_service_leaves_practical_questions_unknown(tmp_path):
    root = repository(tmp_path,'sparse-service')
    result = orient(root)
    assert result['coverage']['onboarding']['purpose']['status']=='included'
    assert all(entry['status']=='missing_evidence' for key,entry in result['coverage']['onboarding'].items() if key!='purpose')
    assert result['status']=='partial'
    assert 'Declared Python runtime' in [item['title'] for item in result['items']]


def test_frontend_task_finds_colocated_regression_and_documented_checks(tmp_path):
    root = repository(tmp_path,'documented-ui')
    result = task_context(root,ContextRequest(task='Change how PriceCard displays quoted totals',budget=30000,encoding='cl100k_base'))
    candidates = {entry['source']['path']:entry for entry in result['source_candidates']}
    assert candidates['src/pricing/PriceCard.tsx']['role']=='source_candidate'
    assert candidates['src/pricing/PriceCard.spec.tsx']['role']=='test_candidate'
    assert 'pnpm test -- PriceCard.spec.tsx' in statements(result)
    assert 'does not calculate discounts' in statements(result)
    assert result['scope']['calibrated'] is False


def test_service_schema_task_carries_documented_migration_and_test_checks(tmp_path):
    root = repository(tmp_path,'fastapi-service')
    result = task_context(root,ContextRequest(task='Add a field to SQLModel models and SQL tables',budget=30000,encoding='cl100k_base'))
    assert 'backend/app/models.py' in {entry['source']['path'] for entry in result['source_candidates']}
    assert 'uv run alembic upgrade head' in statements(result)
    assert 'uv run bash scripts/test.sh' in statements(result)
    assert 'General Workflow' in statements(result)


def test_linked_guidance_obeys_exclusions_and_staleness(tmp_path):
    root = repository(tmp_path,'documented-ui')
    guide = root/'docs/working-on-ui.md'
    guide.write_text('Changed guidance.\n',encoding='utf-8')
    stale = orient(root)
    assert all(stale['coverage']['onboarding'][key]['status']=='unavailable' for key in
               ('responsibilities','conventions','validation'))
    (root/'.knowledgeignore').write_text('docs/working-on-ui.md\n',encoding='utf-8')
    build_repository(root)
    ignored = orient(root)
    assert ignored['coverage']['onboarding']['validation']['status']=='missing_evidence'
    assert 'pnpm typecheck' not in statements(ignored)


def test_unlinked_and_escaping_docs_do_not_become_knowledge(tmp_path):
    root = tmp_path/'links'
    root.mkdir()
    (tmp_path/'outside.md').write_text('## Testing\n\n```sh\npytest external\n```\n',encoding='utf-8')
    (root/'README.md').write_text('A link boundary fixture.\n\n'
        '[Outside](../outside.md) [Encoded outside](%2e%2e/outside.md) '
        '[Remote](https://example.test/remote.md)\n',encoding='utf-8')
    (root/'unlinked.md').write_text('## Testing\n\n```sh\npytest unlinked\n```\n',encoding='utf-8')
    build_repository(root,repository_id='links')
    result = orient(root)
    assert result['coverage']['onboarding']['validation']['status']=='missing_evidence'
    assert 'pytest external' not in statements(result) and 'pytest unlinked' not in statements(result)


def test_inline_tool_name_prefix_is_not_a_setup_command(tmp_path):
    root = tmp_path/'prose'
    root.mkdir()
    (root/'README.md').write_text('A prose fixture.\n\n## Getting started\n\n'
        'This placeholder only mentions `npm installation` and `uv runner`.\n',encoding='utf-8')
    build_repository(root,repository_id='prose')
    assert orient(root)['coverage']['onboarding']['connection']['status']=='missing_evidence'


def test_maintainer_explanation_is_review_bound_and_not_invented_for_upstream(tmp_path):
    root = repository(tmp_path,'documented-ui')
    path = root/'.project-knowledge/map.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    item = KnowledgeItem(id='decision/ui-display-only',topic='decision',title='Keep PriceCard display-only',
        statement='This fictional UI displays server-provided totals; pricing policy stays on the server to keep it in one place.',
        origin='maintainer',basis='maintainer',author='fictional-fixture-author',
        evidence=[capture_file(root,'docs/working-on-ui.md',start_line=12,end_line=13)])
    data['curated']['items']=[item.model_dump(mode='json')]
    document = KnowledgeMap.model_validate(data)
    path.write_text(yaml.safe_dump(document.model_dump(mode='json')),encoding='utf-8')
    before = task_context(root,ContextRequest(task='Change PriceCard pricing policy',budget=30000,encoding='cl100k_base'))
    assert item.id not in {entry['id'] for entry in before['items']}
    record_review(root,item.id,expected_digest=item.content_digest,reviewer='fictional-fixture-reviewer',status='accepted',
                  note='Synthetic review fixture, not an upstream maintainer or human review.')
    after = task_context(root,ContextRequest(task='Change PriceCard pricing policy',budget=30000,encoding='cl100k_base'))
    assert item.id in {entry['id'] for entry in after['items']}
    (root/'docs/working-on-ui.md').write_text('Changed rationale.\n',encoding='utf-8')
    changed = task_context(root,ContextRequest(task='Change PriceCard pricing policy',budget=30000,encoding='cl100k_base'))
    assert item.id not in {entry['id'] for entry in changed['items']}

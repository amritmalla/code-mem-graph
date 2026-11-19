import json

import pytest

from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.knowledge.models import review_item
from code_mem_graph.knowledge.orient import OrientRequest, mcp_result, orientation
from code_mem_graph.serialization import fingerprint
from code_mem_graph.tokens import PayloadCounter


@pytest.fixture
def project(tmp_path):
    (tmp_path / 'README.md').write_text('A fictional delivery service.\n', encoding='utf-8')
    (tmp_path / 'app.py').write_text('def main(): pass\n', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text(
        '[project]\nname="delivery"\nrequires-python=">=3.11"\n'
        '[project.scripts]\ndelivery="app:main"\n[tool.pytest.ini_options]\ntestpaths=["tests"]\n', encoding='utf-8')
    return tmp_path


def enrich(project):
    (project / 'README.md').write_text('A fictional delivery service.\n\n## Quick start\n\n'
        'Requires Python 3.11. Start the service:\n```bash\ndelivery serve --root .\n```\n', encoding='utf-8')
    (project / 'CONTRIBUTING.md').write_text('## Project layout\n\n'
        '- `app.py` owns request handling; `tests/` contains request regressions.\n\n'
        '## Running tests\n\nInstall development dependencies, then:\n```bash\npytest tests\n```\n\n'
        '## Linting\n\n```bash\nruff check app.py tests\n```\n\n'
        '## Conventions\n\nKeep request handlers small.\n', encoding='utf-8')


def test_configuration_and_inventory_do_not_establish_practical_onboarding(project):
    build_repository(project, repository_id='fictional-delivery')
    result = orientation(project, OrientRequest(budget=20000, encoding='cl100k_base'))
    assert result['coverage']['missing_evidence'] == []  # All old category labels are present.
    coverage = result['coverage']['onboarding']
    assert coverage['purpose']['status'] == 'included'
    assert all(coverage[key]['status'] == 'missing_evidence' for key in ('connection', 'responsibilities', 'validation'))
    assert result['status'] == 'partial'


@pytest.mark.parametrize('transport', ['cli', 'mcp'])
def test_practical_evidence_is_prioritized_and_complete_checks_travel_together(project, transport):
    enrich(project)
    document = build_repository(project, repository_id='fictional-delivery')
    observed = []
    result = orientation(project, OrientRequest(budget=20000, encoding='cl100k_base'), transport=transport,
                         observe=observed.append)
    assert all(value['status'] == 'included' for value in result['coverage']['onboarding'].values())
    included = next(candidate for candidate in observed if all(
        value['status'] == 'included' for value in candidate['coverage']['onboarding'].values()))
    checks = [entry for entry in included['items'] if 'validation' in entry.get('onboarding', [])]
    assert len(checks) == 2
    assert any('pytest tests' in entry['statement'] for entry in checks)
    assert any('ruff check app.py tests' in entry['statement'] for entry in checks)
    assert all(entry['basis'] == 'documented' and entry['evidence'][0]['start_line'] for entry in checks)
    bounded = orientation(project, OrientRequest(budget=included['size']['tokens'], encoding='cl100k_base'), transport=transport)
    counted = mcp_result(bounded).model_dump(mode='json', by_alias=True, exclude_none=True) if transport == 'mcp' else bounded
    assert PayloadCounter('cl100k_base')(counted) == bounded['size']['tokens'] <= included['size']['tokens']
    (project / 'CONTRIBUTING.md').write_text('Changed instructions.\n', encoding='utf-8')
    changed = orientation(project, OrientRequest(budget=20000, encoding='cl100k_base'))
    assert changed['coverage']['onboarding']['validation']['status'] == 'unavailable'
    assert not any('validation' in item.get('onboarding', []) for item in changed['items'])
    assert document.generated.items


def test_empty_annotations_preserve_legacy_reviews_and_new_annotations_invalidate_them(project):
    document = build_repository(project, repository_id='fictional-delivery')
    item = next(item for item in document.items if item.id == 'project/source-inventory')
    legacy = item.model_dump(mode='json', exclude={'review', 'lifecycle'})
    assert 'onboarding' not in legacy and item.content_digest == fingerprint(legacy)
    reviewed = review_item(item, reviewer='fixture-reviewer', status='accepted')
    from code_mem_graph.knowledge.models import KnowledgeItem
    assert KnowledgeItem.model_validate_json(reviewed.model_dump_json()) == reviewed
    changed = json.loads(reviewed.model_dump_json())
    changed['onboarding'] = ['responsibilities']
    with pytest.raises(ValueError, match='review does not match'):
        KnowledgeItem.model_validate(changed)


def test_placeholders_and_noncommand_fences_do_not_count_as_guidance(project):
    (project / 'CONTRIBUTING.md').write_text('## Components\n\nTo be written.\n\n'
        '## Testing\n\n```json\n{"tests": "unknown"}\n```\n', encoding='utf-8')
    (project / 'README.md').write_text('Fictional service.\n\n## Quick start\n\nSee a maintainer.\n', encoding='utf-8')
    build_repository(project, repository_id='fictional-delivery')
    result = orientation(project, OrientRequest(budget=20000, encoding='cl100k_base'))
    assert all(result['coverage']['onboarding'][key]['status'] == 'missing_evidence'
               for key in ('connection', 'responsibilities', 'validation'))

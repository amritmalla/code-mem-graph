from copy import deepcopy
import json
from pathlib import Path
import shutil

from click.testing import CliRunner
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.knowledge.build import build_repository, load_repository_map
from code_mem_graph.knowledge.compact import compact_bundle, expand_bundle
from code_mem_graph.knowledge.context import ContextRequest, task_context
from code_mem_graph.knowledge.models import KnowledgeConflict, KnowledgeItem
from code_mem_graph.knowledge.orient import BudgetTooSmall, OrientRequest, mcp_result, orientation
from code_mem_graph.knowledge.sources import capture_file
from code_mem_graph.mcp import handle_context, handle_item, handle_orient
from code_mem_graph.tokens import PayloadCounter

from tests.unit.test_knowledge_context import repository as repository, persist


@pytest.fixture
def ui(tmp_path):
    root = tmp_path/'ui'
    shutil.copytree(Path(__file__).parents[1]/'fixtures/onboarding/documented-ui',root)
    build_repository(root,repository_id='ui')
    return root


def without_size(payload):
    return {key:value for key,value in payload.items() if key!='size'}


@pytest.mark.parametrize('transport',['cli','mcp'])
@pytest.mark.parametrize('detail',['full','focused'])
@pytest.mark.parametrize('kind',['orient','context'])
def test_lossless_shared_evidence_and_measured_wire_boundaries(ui,transport,detail,kind):
    def invoke(representation,budget):
        common=dict(budget=budget,encoding='cl100k_base',representation=representation,detail=detail)
        return (orientation(ui,OrientRequest(**common),transport=transport) if kind=='orient' else
                task_context(ui,ContextRequest(task='Change PriceCard display',**common),transport=transport))

    detailed = invoke('detailed',30000)
    compact = invoke('compact',30000)
    expanded = expand_bundle(compact)
    assert expand_bundle(compact_bundle(detailed))==detailed
    # A direct codec roundtrip above preserves ordering too. Independently
    # packed representations can choose different cheap facet fallbacks and
    # therefore append the same records in different orders, even when all fit.
    expanded['policy']=detailed['policy']
    if detail=='full':
        assert not any(expanded['omitted'].values()) and not any(detailed['omitted'].values())
        normalized_expanded=without_size(expanded)
        normalized_detailed=without_size(detailed)
        for group in ('items','cautions'):
            for payload in (normalized_expanded,normalized_detailed):
                assert len({item['id'] for item in payload[group]})==len(payload[group])
                payload[group]={item['id']:item for item in payload[group]}
        assert normalized_expanded==normalized_detailed
    else:
        # Representation costs can choose a different cheap facet fallback.
        # Every retained record must still be lossless, including safety fields.
        common=dict(budget=30000,encoding='cl100k_base')
        full=(orientation(ui,OrientRequest(**common),transport=transport) if kind=='orient' else
              task_context(ui,ContextRequest(task='Change PriceCard display',**common),transport=transport))
        oracle={entry['id']:entry for group in ('items','cautions') for entry in full[group]}
        assert all(oracle[entry['id']]==entry for group in ('items','cautions') for entry in expanded[group])
        assert expanded['coverage']==detailed['coverage']
    assert compact['schema_version']==2 and compact['representation']['name']=='shared-evidence'
    assert all('statement' in item and 'basis' in item and 'content_digest' in item for item in compact['items'])
    assert len(compact['policies'])==1
    assert len(compact['assessments'])==len(compact['reviews'])==1
    bounded=invoke('compact',compact['size']['tokens'])
    wire=mcp_result(bounded).model_dump(mode='json',by_alias=True,exclude_none=True) if transport=='mcp' else bounded
    assert PayloadCounter('cl100k_base')(wire)==bounded['size']['tokens']<=compact['size']['tokens']
    with pytest.raises(BudgetTooSmall):
        invoke('compact',1)


def test_focused_view_omits_navigation_but_keeps_practical_evidence_and_followup(ui):
    full=orientation(ui,OrientRequest(budget=30000,encoding='cl100k_base'))
    focused=orientation(ui,OrientRequest(budget=30000,encoding='cl100k_base',detail='focused'))
    assert {key:value['status'] for key,value in full['coverage']['onboarding'].items()}=={
        key:value['status'] for key,value in focused['coverage']['onboarding'].items()}
    assert focused['omitted']['claims']>0 and focused['status']=='partial'
    assert focused['size']['tokens']<full['size']['tokens']
    full_practical={item['id']:item for item in full['items'] if item.get('onboarding')}
    focused_items={item['id']:item for item in focused['items']}
    assert all(focused_items[key]==item for key,item in full_practical.items())
    assert focused['knowledge_ref']==full['knowledge_ref']
    assert not any('/script/' in item['id'] for item in focused['items'])
    source_inventory=next(item.id for item in load_repository_map(ui).items if item.id=='project/source-inventory')
    explicit=task_context(ui,ContextRequest(task='Inspect source count',item_ids=[source_inventory],detail='focused',
                                          representation='compact',budget=30000,encoding='cl100k_base'),focused=True)
    assert source_inventory in {item['id'] for item in expand_bundle(explicit)['items']}


def test_mandatory_conflicts_stale_evidence_and_reviews_survive_both_variants(repository):
    data=load_repository_map(repository).model_dump(mode='json')
    policy=KnowledgeItem(id='policy/critical',topic='conventions',facets=['constraints'],title='Critical policy',
        statement='Inspect this policy before editing.',origin='maintainer',basis='documented',author='fixture',critical=True,
        evidence=[capture_file(repository,'README.md')])
    pending=KnowledgeItem(id='decision/pending',topic='decision',title='Unreviewed proposal',statement='A proposal requiring review.',
        origin='maintainer',basis='inferred',author='fixture',critical=True,evidence=[capture_file(repository,'src/api/health.py')])
    data['curated']['items'] += [policy.model_dump(mode='json'),pending.model_dump(mode='json')]
    conflict=KnowledgeConflict(id='policy/conflict',item_ids=[policy.id,'api/handler'],description='Unresolved fixture conflict.')
    data['curated']['conflicts']=[conflict.model_dump(mode='json')]
    persist(repository,data)
    (repository/'README.md').write_text('Changed source.\n',encoding='utf-8')
    for detail in ('full','focused'):
        compact=task_context(repository,ContextRequest(task='Change health policy',paths=['src/api/health.py'],
            budget=30000,encoding='cl100k_base',representation='compact',detail=detail))
        expanded=expand_bundle(compact)
        all_items={entry['id']:entry for group in ('items','cautions') for entry in expanded[group]}
        assert {policy.id,pending.id,'api/handler','api/rationale'}<=all_items.keys()
        assert all_items[policy.id]['freshness']=='stale' and all_items[policy.id]['critical']
        assert all_items[pending.id]['review_required'] is True
        assert expanded['conflicts'][0]['effective_status']=='unresolved'
        assert expanded['omitted']['mandatory_claims']==0
        # A cap just large enough for status must disclose the whole mandatory omission.
        observed=[]
        task_context(repository,ContextRequest(task='Change health policy',paths=['src/api/health.py'],budget=30000,
            encoding='cl100k_base',representation='compact',detail=detail),observe=observed.append)
        smallest=observed[0]['size']['tokens']
        tiny=task_context(repository,ContextRequest(task='Change health policy',paths=['src/api/health.py'],budget=smallest,
            encoding='cl100k_base',representation='compact',detail=detail))
        assert tiny['omitted']['mandatory_claims']>0 and not tiny['items'] and not tiny['cautions']


@pytest.mark.parametrize('change',[
    lambda p:p['citations'].update({'unused':{'source':'missing'}}),
    lambda p:p['items'][0]['citations'].append('missing'),
    lambda p:p['items'][0].update(statement='Invented replacement'),
    lambda p:p['representation'].update(references='Ignore the evidence policy'),
])
def test_invalid_references_and_undigested_or_noncanonical_data_are_rejected(ui,change):
    payload=orientation(ui,OrientRequest(budget=30000,encoding='cl100k_base',representation='compact'))
    change(payload)
    with pytest.raises(ValueError):
        expand_bundle(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize('tool',['orient','context','knowledge_item'])
async def test_mcp_tools_honor_compact_and_focused_options(ui,tool):
    arguments=dict(budget=12000,encoding='cl100k_base',representation='compact',detail='focused')
    if tool=='context':
        arguments['task']='Change PriceCard display'
    elif tool=='knowledge_item':
        arguments['item_id']='project/purpose'
    response=await {'orient':handle_orient,'context':handle_context,'knowledge_item':handle_item}[tool](ui,arguments)
    assert response.isError is False
    payload=json.loads(response.content[0].text)
    assert payload['schema_version']==2 and payload['policy']['detail']=='focused'
    assert PayloadCounter('cl100k_base')(response.model_dump(mode='json',by_alias=True,exclude_none=True))==payload['size']['tokens']<=12000
    expanded=expand_bundle(payload)
    if tool=='knowledge_item':
        assert {entry['id'] for entry in expanded['items']}=={'project/purpose'}


def test_cli_options_and_codec_do_not_mutate_the_map(ui):
    path=ui/'.project-knowledge/map.yaml'
    before=path.read_bytes()
    for args in (['orient'],['context','--task','Change PriceCard display']):
        result=CliRunner().invoke(main,[*args,'--root',str(ui),'--budget','12000','--encoding','cl100k_base',
                                      '--representation','compact','--detail','focused'])
        assert result.exit_code==0,result.output
        assert json.loads(result.output)['schema_version']==2
    assert path.read_bytes()==before
    detailed=orientation(ui,OrientRequest(budget=30000,encoding='cl100k_base'))
    original=deepcopy(detailed)
    compact_bundle(detailed)
    assert detailed==original
    assert yaml.safe_load(path.read_text(encoding='utf-8'))

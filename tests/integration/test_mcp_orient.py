import json
from pathlib import Path
import shutil

from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest
import pytest

from code_mem_graph.engine.config import Settings
from code_mem_graph.engine.server import build_server as legacy_server
from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.mcp import build_server, handle_orient
from code_mem_graph.tokens import PayloadCounter


@pytest.fixture
def repository(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/knowledge/python_service'
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    build_repository(tmp_path, repository_id='mcp-fixture')
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True])
async def test_registered_orientation_counts_actual_sdk_envelope(repository, legacy, monkeypatch):
    def no_model(*args, **kwargs):
        pytest.fail('orientation must not initialize embeddings or an LLM')

    monkeypatch.setattr('code_mem_graph.engine.embeddings.factory.make_embedder', no_model)
    monkeypatch.setattr('code_mem_graph.engine.llm.factory.make_llm_client', no_model)
    server = legacy_server(settings=Settings.for_repo(repository)) if legacy else build_server(repository)
    listing = await server.request_handlers[ListToolsRequest](ListToolsRequest(method='tools/list'))
    schemas = {tool.name: tool for tool in listing.root.tools}
    assert 'orient' in schemas
    assert set(schemas['orient'].inputSchema['required']) == {'budget', 'encoding'}
    response = await server.request_handlers[CallToolRequest](CallToolRequest(
        method='tools/call', params=CallToolRequestParams(name='orient', arguments={'budget': 3000, 'encoding': 'cl100k_base'})))
    result = response.root
    assert not result.isError
    payload = json.loads(result.content[0].text)
    actual_envelope = result.model_dump(mode='json', by_alias=True, exclude_none=True)
    assert PayloadCounter('cl100k_base')(actual_envelope) == payload['size']['tokens'] <= 3000
    assert payload['repository_id'] == 'mcp-fixture'


@pytest.mark.asyncio
async def test_invalid_arguments_and_tiny_budget_are_errors(repository):
    server = build_server(repository)
    for arguments in ({}, {'budget': True, 'encoding': 'cl100k_base'},
                      {'budget': 1, 'encoding': 'cl100k_base'},
                      {'budget': 1000, 'encoding': 'unknown-encoding'},
                      {'budget': 1000, 'encoding': 'cl100k_base', 'path': '../private'}):
        response = await server.request_handlers[CallToolRequest](CallToolRequest(
            method='tools/call', params=CallToolRequestParams(name='orient', arguments=arguments)))
        assert response.root.isError


@pytest.mark.asyncio
async def test_missing_map_and_pinned_root(repository, tmp_path, monkeypatch):
    empty = repository / 'empty'
    empty.mkdir()
    result = await handle_orient(empty, {'budget': 2000, 'encoding': 'cl100k_base'})
    assert result.isError and 'knowledge' in result.content[0].text.lower()
    server = build_server(repository)
    monkeypatch.chdir(empty)
    response = await server.request_handlers[CallToolRequest](CallToolRequest(
        method='tools/call', params=CallToolRequestParams(name='orient', arguments={'budget': 2000, 'encoding': 'cl100k_base'})))
    assert not response.root.isError
    assert json.loads(response.root.content[0].text)['repository_id'] == 'mcp-fixture'


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('name', ['context', 'knowledge_item'])
async def test_task_and_item_tools_use_complete_envelope_caps(repository, legacy, name, monkeypatch):
    def no_model(*args, **kwargs):
        pytest.fail('knowledge reads must not initialize an LLM or embedding model')
    monkeypatch.setattr('code_mem_graph.engine.embeddings.factory.make_embedder', no_model)
    monkeypatch.setattr('code_mem_graph.engine.llm.factory.make_llm_client', no_model)
    server = legacy_server(settings=Settings.for_repo(repository)) if legacy else build_server(repository)
    arguments = {'budget': 4000, 'encoding': 'cl100k_base'}
    if name == 'context':
        arguments.update(task='Inspect the health implementation', paths=['src/parcel/app.py'])
    else:
        arguments.update(item_id='project/purpose')
    response = await server.request_handlers[CallToolRequest](CallToolRequest(
        method='tools/call', params=CallToolRequestParams(name=name, arguments=arguments)))
    result = response.root
    assert not result.isError
    payload = json.loads(result.content[0].text)
    assert PayloadCounter('cl100k_base')(result.model_dump(mode='json', by_alias=True, exclude_none=True)) == payload['size']['tokens'] <= 4000
    if name == 'knowledge_item':
        assert {item['id'] for key in ('items', 'cautions') for item in payload[key]} == {'project/purpose'}
        assert payload['source_candidates'] == []
    else:
        assert any(entry.get('source', {}).get('path') == 'src/parcel/app.py' for entry in payload['source_candidates'])
    assert not (repository / '.project-knowledge/.build.lock').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [
    {'budget': 4000, 'encoding': 'cl100k_base'},
    {'task': 'health', 'budget': 'auto', 'encoding': 'cl100k_base'},
    {'task': 'health', 'budget': 4000, 'encoding': 'cl100k_base', 'paths': ['../private.py']},
    {'task': 'health', 'budget': 4000, 'encoding': 'cl100k_base', 'item_ids': ['absent']},
])
async def test_task_arguments_are_validated(repository, arguments):
    server = build_server(repository)
    response = await server.request_handlers[CallToolRequest](CallToolRequest(
        method='tools/call', params=CallToolRequestParams(name='context', arguments=arguments)))
    assert response.root.isError

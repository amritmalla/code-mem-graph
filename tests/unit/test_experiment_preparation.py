import json
import shutil

from click.testing import CliRunner
from pydantic import ValidationError
import pytest
import yaml

from code_mem_graph.cli import main
from code_mem_graph.evaluation.analysis import analyze_experiment
from code_mem_graph.evaluation.experiment import (
    Experiment, bundle_identity, check_partitions, count_bundle, load_experiment, preparation_summary,
    prepare_experiment, save_experiment,
)
from code_mem_graph.evaluation.models import AgentProfile, CaptureInfo, ModelEvent, Review, Suite, Trace, Usage, UsageCapture, UsageObservation
from code_mem_graph.evaluation.snapshot import Snapshot, capture_snapshot
from code_mem_graph.knowledge.build import build_repository
from code_mem_graph.knowledge.models import KnowledgeItem, KnowledgeMap
from code_mem_graph.knowledge.orient import OrientRequest, orientation
from code_mem_graph.knowledge.compact import compact_bundle, expand_bundle
from code_mem_graph.tokens import PayloadCounter
from tests.review_fixtures import supported_audit


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'README.md').write_text('A fictional experiment fixture.\n', encoding='utf-8')
    (root / 'app.py').write_text('def health():\n    return "ok"\n', encoding='utf-8')
    (root / 'pyproject.toml').write_text('[project]\nname="experiment-fixture"\nrequires-python=">=3.11"\n'
        '[project.scripts]\nfixture="app:health"\n[tool.pytest.ini_options]\ntestpaths=["tests"]\n', encoding='utf-8')
    build_repository(root, repository_id='experiment-fixture')
    return root


@pytest.fixture
def suite():
    return Suite.model_validate({
        'schema_version': 1, 'id': 'fixture-suite',
        'protocol': {'id': 'fixture-protocol', 'max_tool_calls': 5, 'max_model_turns': 3,
                     'max_elapsed_ms': 10000.0, 'stopping_rule': 'Stop on answer or ceilings.'},
        'repositories': [{'id': 'fixture-repo', 'revision': 'old-foundation'}],
        'tasks': [{'id': 'overview', 'repository_id': 'fixture-repo', 'category': 'orientation',
                   'prompt': 'Explain the fixture and its runtime.',
                   'criteria': [{'id': 'runtime', 'description': 'PRIVATE_RUBRIC_SENTINEL', 'critical': True,
                                 'source_refs': ['pyproject.toml']}],
                   'acceptance_checks': {'answer': 'PRIVATE_ACCEPTANCE_SENTINEL'}}],
    })


def prepare(root, suite, **kwargs):
    settings = dict(task_id='overview', mode='orient', encoding='cl100k_base', ceiling=10000,
                    split='calibration', replicates=2, cache_state='cold')
    settings.update(kwargs)
    return prepare_experiment(root, suite, **settings)


def profile(kind='agent'):
    return AgentProfile(kind=kind, provider='synthetic-unit-fixture', model='fixture', model_version='pinned-fixture',
                        instructions_digest='fixture-instructions', settings={'tools': ['list', 'read', 'search']})


def trace_for(experiment, scheduled, *, accepted=True, tokens=100, kind='agent', usage=True):
    counters = Usage(input_tokens=tokens, output_tokens=0, cached_input_tokens=0) if usage else None
    trace = Trace(schema_version=1, run_id=scheduled['run_id'], suite_digest=experiment.suite.digest,
        task_id=experiment.task_id, repository_revision='snapshot:' + experiment.snapshot.digest,
        replicate=scheduled['replicate'], cache_state=scheduled['cache_state'], agent=profile(kind),
        policy=scheduled['policy'], elapsed_ms=25.0, outcome='completed' if accepted else 'failed',
        coordination={'registration_digest':'fictional-shared-registration','deviations':[]},
        events=[ModelEvent(sequence=0, request={'messages': ['synthetic request']},
                           response={'tool_calls':[],'final':'The package declares Python >=3.11 [pyproject.toml:3].'},
                           usage=counters, usage_capture=UsageCapture(scope='provider_calls',source='fictional fixture provider',
                               expected_calls=1,internal_calls_complete=True,observations=[UsageObservation(id=scheduled['run_id'],
                                   reference='fictional response',outcome='completed',usage=counters)]))],
        capture=CaptureInfo(request_capture='complete',provider_usage_available=True,model_version_pinned=True,
                            host_tools_enforced=True,provider_token_budget_enforced=True),
        review=Review(verdict='accepted' if accepted else 'rejected', reviewer='independent-fixture',
                      criteria={'runtime': 'met' if accepted else 'not_met'},
                      acceptance_checks={'answer': 'passed' if accepted else 'failed'}))
    trace.review.validation=supported_audit(experiment.suite,trace,experiment.snapshot).model_dump(mode='json')
    return trace


def small_experiment(root, suite, **kwargs):
    value = prepare(root, suite, **kwargs)
    data = value.model_dump(mode='json')
    data['candidates'] = data['candidates'][-1:]  # Analysis fixtures test one predeclared treatment, without executing models.
    return Experiment.model_validate(data)


@pytest.mark.parametrize('mode', ['orient', 'context'])
def test_candidate_states_are_measured_complete_bundle_boundaries(repository, suite, mode):
    experiment = prepare(repository, suite, mode=mode)
    assert len(experiment.candidates) > 1
    counter = PayloadCounter(experiment.encoding)
    identities = set()
    for candidate in experiment.candidates:
        assert count_bundle(candidate.payload, counter) == candidate.tokens <= experiment.ceiling
        assert candidate.id == bundle_identity(candidate.payload)
        assert candidate.id not in identities
        identities.add(candidate.id)
        assert not candidate.payload['omitted']['mandatory_claims']
        assert not candidate.payload['omitted']['conflicts']
    assert experiment.suite.repositories[0].revision == 'snapshot:' + experiment.snapshot.digest
    assert experiment.source_suite_digest == suite.digest
    assert prepare(repository, suite, mode=mode).digest == experiment.digest
    summary = preparation_summary(experiment)
    assert summary['model_calls'] == 0 and summary['default_selected'] is None
    assert summary['calibration_status'] == 'not_calibrated'


@pytest.mark.parametrize('mode',['orient','context'])
def test_all_bundle_variants_enter_frozen_evaluation_and_keep_exact_records(repository,suite,mode,tmp_path):
    experiment=prepare(repository,suite,mode=mode,representations=('detailed','compact'),details=('full','focused'))
    variants={(candidate.payload['policy'].get('representation','detailed'),candidate.payload['policy'].get('detail','full'))
              for candidate in experiment.candidates}
    assert variants=={('detailed','full'),('compact','full'),('detailed','focused'),('compact','focused')}
    counter=PayloadCounter(experiment.encoding)
    for candidate in experiment.candidates:
        assert count_bundle(candidate.payload,counter)==candidate.tokens<=experiment.ceiling
        assert expand_bundle(candidate.payload)['knowledge_ref']['digest']==candidate.payload['knowledge_ref']['digest']
    path=tmp_path/'variants.json'
    save_experiment(experiment,path)
    restored=load_experiment(path)
    assert restored.digest==experiment.digest
    for candidate in restored.candidates:
        run=next(entry for entry in restored.schedule() if entry['policy']['parameters'].get('candidate_id')==candidate.id)
        assert restored.agent_packet(run['run_id'])['initial_knowledge']==candidate.payload


def test_even_self_consistent_compact_claim_edits_cannot_rebind_to_frozen_map(repository,suite):
    experiment=prepare(repository,suite,representations=('compact',))
    data=experiment.model_dump(mode='json')
    candidate=data['candidates'][-1]
    expanded=expand_bundle(candidate['payload'])
    expanded['items'][0]['statement']='Invented, but encoded with a valid compact content checksum.'
    candidate['payload']=compact_bundle(expanded)
    candidate['id']=bundle_identity(candidate['payload'])
    with pytest.raises(ValueError,match='frozen knowledge record'):
        Experiment.model_validate(data)


def test_cli_registers_explicit_bundle_alternatives(repository,suite,tmp_path):
    suite_path=tmp_path/'variants-suite.yaml'
    suite_path.write_text(yaml.safe_dump(suite.model_dump(mode='json')),encoding='utf-8')
    output=tmp_path/'variants-cli.json'
    args=['experiment','prepare','--root',str(repository),'--suite',str(suite_path),'--task-id','overview',
          '--mode','orient','--encoding','cl100k_base','--ceiling','10000','--split','calibration',
          '--replicates','1','--cache-state','cold','--output',str(output),
          '--representation','detailed','--representation','compact','--detail','full','--detail','focused']
    result=CliRunner().invoke(main,args)
    assert result.exit_code==0,result.output
    summary=json.loads(result.output)
    assert summary['model_calls']==0 and summary['default_selected'] is None
    assert len({(entry['representation'],entry['detail']) for entry in summary['candidates']})==4


def test_snapshot_is_immutable_and_native_access_matches_knowledge_documents(repository):
    frozen = capture_snapshot(repository)
    assert '.project-knowledge/map.yaml' in {entry.path for entry in frozen.files}
    assert '.project-knowledge/README.md' in {entry.path for entry in frozen.files}
    before = frozen.read('README.md', start_line=1, end_line=1)
    assert frozen.list_files('README.md')[0]['lines'] == 1
    (repository / 'README.md').write_text('Modified after capture.', encoding='utf-8')
    assert frozen.read('README.md', start_line=1, end_line=1) == before
    assert any(entry['path'] == 'README.md' for entry in frozen.search('fictional'))
    assert Snapshot.model_validate_json(frozen.model_dump_json()) == frozen
    with pytest.raises(ValueError, match='outside'):
        frozen.read('../private', start_line=1, end_line=1)
    for start, end in [(0, 1), (True, 1), (1.5, 2), (2, 1), (1, 500)]:
        with pytest.raises(ValueError):
            frozen.read('README.md', start_line=start, end_line=end)


def test_snapshot_excludes_ignored_files_and_local_settings(repository):
    (repository / '.gitignore').write_text('private/\n', encoding='utf-8')
    (repository / 'private').mkdir()
    (repository / 'private/secrets.py').write_text('do-not-ingest-private', encoding='utf-8')
    (repository / '.mcp.json').write_text('{"private":"do-not-ingest-settings"}', encoding='utf-8')
    frozen = capture_snapshot(repository)
    assert 'do-not-ingest-' not in frozen.model_dump_json()


def test_packets_hide_review_rubrics_and_schedule_every_arm_before_execution(repository, suite):
    experiment = prepare(repository, suite)
    schedule = experiment.schedule()
    assert len(schedule) == (len(experiment.candidates) + 1) * experiment.replicates
    assert len({entry['run_id'] for entry in schedule}) == len(schedule)
    for run in schedule:
        packet = experiment.agent_packet(run['run_id'])
        assert 'PRIVATE_RUBRIC_SENTINEL' not in json.dumps(packet)
        assert 'PRIVATE_ACCEPTANCE_SENTINEL' not in json.dumps(packet)
        assert packet['prompt'] == suite.task('overview').prompt
        assert packet['native_tools'] == ['list', 'read', 'search']
        assert (packet['initial_knowledge'] is None) == (run['policy']['name'] == 'native')
    with pytest.raises(ValueError, match='outside'):
        experiment.agent_packet('not-scheduled')


def test_artifact_hash_and_size_tampering_is_rejected(repository, suite, tmp_path):
    experiment = prepare(repository, suite)
    path = tmp_path / 'experiment.json'
    save_experiment(experiment, path)
    assert load_experiment(path) == experiment
    with pytest.raises(FileExistsError):
        save_experiment(experiment, path)
    assert not list(tmp_path.glob('.experiment-*'))
    data = experiment.model_dump(mode='json')
    data['snapshot']['files'][0]['content'] += 'tampered'
    with pytest.raises(ValidationError, match='digest'):
        Experiment.model_validate(data)
    data = experiment.model_dump(mode='json')
    data['candidates'][0]['tokens'] += 1
    with pytest.raises(ValidationError, match='size metadata'):
        Experiment.model_validate(data)
    data = experiment.model_dump(mode='json')
    candidate = data['candidates'][0]
    candidate['tokens'] += 1
    candidate['payload']['size']['tokens'] += 1
    # Identity excludes accounting metadata, so loading additionally verifies the actual tokenizer measurement.
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='tokenizer measurement'):
        load_experiment(path)
    with pytest.raises(ValueError, match='non-manifest'):
        save_experiment(experiment, tmp_path / 'package.json')


def test_changed_snapshot_and_outside_rubric_sources_reject_preparation(repository, suite, monkeypatch):
    import code_mem_graph.evaluation.experiment as module
    original = module.orientation

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        (repository / 'README.md').write_text('changed mid-collection', encoding='utf-8')
        return result

    monkeypatch.setattr(module, 'orientation', changing)
    with pytest.raises(ValueError, match='snapshot changed'):
        prepare(repository, suite)
    monkeypatch.setattr(module, 'orientation', original)
    data = suite.model_dump()
    data['tasks'][0]['criteria'][0]['source_refs'] = ['outside/unknown.py']
    with pytest.raises(ValueError, match='outside the frozen'):
        prepare(repository, Suite.model_validate(data))


def test_observer_cannot_mutate_the_returned_orientation(repository):
    def observe(payload):
        payload['items'].clear()
        payload['cautions'].clear()
    result = orientation(repository, OrientRequest(budget=10000, encoding='cl100k_base'), observe=observe)
    assert result['items']


def test_mandatory_context_cannot_be_turned_into_an_unsafe_trial_candidate(repository, suite):
    path = repository / '.project-knowledge/map.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    critical = KnowledgeItem(id='decision/critical', topic='decision', title='Critical fixture guidance',
        statement='Critical fixture guidance ' * 5000, origin='maintainer', basis='maintainer', author='fixture', critical=True)
    data['curated']['items'] = [critical.model_dump(mode='json')]
    document = KnowledgeMap.model_validate(data)
    path.write_text(yaml.safe_dump(document.model_dump(mode='json')), encoding='utf-8')
    with pytest.raises(ValueError, match='no complete mandatory'):
        prepare(repository, suite, ceiling=2000)


def test_failure_inclusive_analysis_cannot_reward_successful_runs_only(repository, suite):
    experiment = small_experiment(repository, suite, agent=profile())
    traces = []
    for run in experiment.schedule():
        candidate = run['policy']['name'] != 'native'
        # Native costs 100 on each of two successes. Candidate spends 10 on success + 1000 on failure.
        failed = candidate and run['replicate'] == '2'
        traces.append(trace_for(experiment, run, accepted=not failed, tokens=1000 if failed else 10 if candidate else 100))
    report = analyze_experiment(experiment, traces)
    native, candidate = report['policies']
    assert native['tokens_per_accepted_result'] == 100
    assert candidate['observed_tokens_spent'] == 1010
    assert candidate['tokens_per_accepted_result'] == 1010
    assert candidate['acceptance_rate'] == .5
    assert report['comparisons'][0]['observed_quality_regressions'] == 1
    assert report['comparisons'][0]['token_saving_per_accepted_result'] < 0
    assert report['readiness'] == 'observed_tradeoffs_only'
    assert report['default_selected'] is None and report['calibration_status'] == 'not_calibrated'


@pytest.mark.parametrize('coordination,blocker', [
    (None,'coordination_undeclared'),
    ({'registration_digest':'different','deviations':[]},'cohort_registrations_differ'),
    ({'registration_digest':'fictional-shared-registration','deviations':['delivery_recovered_after_interruption']},'delivery_recovered_after_interruption'),
])
def test_analysis_exposes_coordination_deviations(repository,suite,coordination,blocker):
    experiment = small_experiment(repository,suite,agent=profile())
    traces = [trace_for(experiment,entry) for entry in experiment.schedule()]
    traces[0].coordination = coordination
    report = analyze_experiment(experiment,traces)
    assert blocker in report['readiness_reasons']
    assert all(not value['eligible_for_agent_comparison'] for value in report['comparisons'])


@pytest.mark.parametrize('mode,blocker',[
    ('missing','answer_validation_undeclared'),('edited','answer_validation_invalid'),('disagreement','review_disagreement'),
])
def test_analysis_cannot_bypass_answer_evidence_with_complete_capture(repository,suite,mode,blocker):
    experiment=small_experiment(repository,suite,agent=profile())
    traces=[trace_for(experiment,entry) for entry in experiment.schedule()]
    target=traces[0]
    if mode=='missing':
        target.review.validation=None
    elif mode=='edited':
        target.events[0].response['final']+=' Changed after review.'
    else:
        from copy import deepcopy
        dissent=deepcopy(target.review.validation['reviews'][0])
        dissent['reviewer']['reference']='independent-second-opinion'
        dissent['units'][0]['verdict']='unsupported'
        target.review.validation['reviews'].append(dissent)
    report=analyze_experiment(experiment,traces)
    assert blocker in report['readiness_reasons']
    assert all(not entry['eligible_for_agent_comparison'] for entry in report['comparisons'])


def test_missing_reviews_usage_and_models_stay_unknown(repository, suite):
    experiment = small_experiment(repository, suite)
    missing = analyze_experiment(experiment, [])
    assert missing['missing_run_ids'] and missing['readiness'] == 'incomplete'
    assert all(policy['observed_tokens_spent'] is None for policy in missing['policies'])
    run = experiment.schedule()[0]
    trace = trace_for(experiment, run, usage=False)
    trace.review = Review()
    report = analyze_experiment(experiment, [trace])
    assert 'reviews_incomplete' in report['readiness_reasons']
    assert 'agent_profile_not_predeclared' in report['readiness_reasons']
    assert report['policies'][0]['token_basis'] == 'model_total_tokens_estimate'
    assert report['comparisons'][0]['token_saving_per_accepted_result'] is None
    trace.events = []
    report = analyze_experiment(experiment, [trace])
    assert 'model_usage_unavailable' in report['readiness_reasons']


@pytest.mark.parametrize('change', ['run_id', 'policy', 'replicate', 'cache', 'profile'])
def test_unplanned_and_incompatible_trials_are_rejected(repository, suite, change):
    experiment = small_experiment(repository, suite, agent=profile())
    trace = trace_for(experiment, experiment.schedule()[0])
    if change == 'run_id':
        trace.run_id = 'unplanned'
    elif change == 'policy':
        trace.policy.parameters['candidate_id'] = 'unplanned'
    elif change == 'replicate':
        trace.replicate = 'another-block'
    elif change == 'cache':
        trace.cache_state = 'warm'
    else:
        trace.agent.model_version = 'changed-model'
    with pytest.raises(ValueError):
        analyze_experiment(experiment, [trace])


def test_scripted_runs_and_duplicate_imports_never_become_calibration_evidence(repository, suite):
    experiment = small_experiment(repository, suite, agent=profile('scripted'))
    traces = [trace_for(experiment, run, kind='scripted') for run in experiment.schedule()]
    report = analyze_experiment(experiment, traces)
    assert 'scripted_runs_do_not_measure_agent_performance' in report['readiness_reasons']
    assert not report['comparisons'][0]['eligible_for_agent_comparison']
    with pytest.raises(ValueError, match='duplicate'):
        analyze_experiment(experiment, [traces[0], traces[0]])


def test_cli_prepares_inspects_packets_and_reports_missing_trials(repository, suite, tmp_path):
    suite_path = repository / 'rubric.yaml'
    suite_path.write_text(yaml.safe_dump(suite.model_dump()), encoding='utf-8')
    path = tmp_path / 'prepared.json'
    args = ['experiment', 'prepare', '--root', str(repository), '--suite', str(suite_path), '--task-id', 'overview',
            '--mode', 'context', '--encoding', 'cl100k_base', '--ceiling', '10000', '--split', 'validation',
            '--replicates', '2', '--cache-state', 'cold', '--output', str(path)]
    runner = CliRunner()
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary['model_calls'] == 0 and summary['default_selected'] is None
    inspected = runner.invoke(main, ['experiment', 'inspect', str(path)])
    assert inspected.exit_code == 0, inspected.output
    schedule = json.loads(inspected.output)['schedule']
    packet = runner.invoke(main, ['experiment', 'packet', str(path), '--run-id', schedule[0]['run_id']])
    assert 'PRIVATE_RUBRIC_SENTINEL' not in packet.output
    empty = tmp_path / 'traces'
    empty.mkdir()
    assessed = runner.invoke(main, ['experiment', 'assess', str(path), '--traces', str(empty)])
    assert assessed.exit_code == 0, assessed.output
    assert json.loads(assessed.output)['readiness'] == 'incomplete'
    assert runner.invoke(main, args).exit_code != 0


def test_private_inputs_in_public_catalog_are_rejected_instead_of_leaking(repository, suite):
    path = repository / 'private-rubric.md'
    path.write_text('PRIVATE_RUBRIC_SENTINEL', encoding='utf-8')
    with pytest.raises(ValueError, match='outside the public source catalog'):
        prepare(repository, suite, private_paths=['private-rubric.md'])


def test_partition_checks_reject_relabelled_tasks_and_repositories(repository, suite, tmp_path):
    calibration = prepare(repository, suite)
    heldout = prepare(repository, suite, split='heldout')
    with pytest.raises(ValueError, match='held-out repositories overlap'):
        check_partitions([calibration, heldout])
    with pytest.raises(ValueError, match='duplicate'):
        check_partitions([calibration, calibration])
    validation = prepare(repository, suite, split='validation')
    with pytest.raises(ValueError, match='task evidence overlaps'):
        check_partitions([calibration, validation])
    clone = tmp_path / 'renamed-clone'
    clone.mkdir()
    for path in ['README.md', 'app.py', 'pyproject.toml']:
        shutil.copyfile(repository / path, clone / path)
    build_repository(clone, repository_id='renamed-fixture')
    cloned = prepare(clone, suite, split='heldout')
    assert cloned.snapshot.digest != calibration.snapshot.digest
    assert cloned.snapshot.source_digest == calibration.snapshot.source_digest
    with pytest.raises(ValueError, match='held-out repositories overlap'):
        check_partitions([calibration, cloned])


def test_partitioned_drafts_still_do_not_select_a_default(repository, suite, tmp_path):
    calibration = prepare(repository, suite)
    data = suite.model_dump()
    data['tasks'][0].update(id='entrypoints', prompt='Identify declared entry points.')
    validation = prepare(repository, Suite.model_validate(data), task_id='entrypoints', split='validation')
    independent = tmp_path / 'independent'
    independent.mkdir()
    (independent / 'README.md').write_text('Another fictional project.\n', encoding='utf-8')
    (independent / 'pyproject.toml').write_text('[project]\nname="different-fixture"\nrequires-python=">=3.12"\n', encoding='utf-8')
    build_repository(independent, repository_id='independent-fixture')
    heldout = prepare(independent, suite, split='heldout')
    report = check_partitions([calibration, validation, heldout])
    assert report['status'] == 'partitioned'
    assert report['default_selected'] is None and report['calibration_status'] == 'not_calibrated'
    assert check_partitions([calibration])['status'] == 'incomplete'


def test_edit_acceptance_tasks_are_not_misrepresented_as_read_only_trials(repository, suite):
    data = suite.model_dump()
    data['tasks'][0]['category'] = 'change'
    with pytest.raises(ValueError, match='read-only'):
        prepare(repository, Suite.model_validate(data))

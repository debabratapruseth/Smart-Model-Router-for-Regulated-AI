"""Final harness verification: frozen inputs, in-memory provider mocks, no network."""
import copy
import json
import socket
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.core.config_loader import ROOT, fingerprint
from app.core.router import ModelRouter
from app.strategies.ml import MLRouter
from app.strategies.llm import OpenAILiveRouter
from app.strategies.live_jev import JevLiveRouter
from evaluation.frozen_baseline import FrozenBaseline
from evaluation.experiment_sampling import (KINDS, experiment_config, build_sample, validate_sample,
    prepare_experiment, experiment_items, largest_remainder, pilot_exclusions, item_for, stratified_sample)
from evaluation.experiment_preflight import preflight
from evaluation.experiment_statistics import (bootstrap_ci, mcnemar, holm_bonferroni, paired_wilcoxon,
    primary_analysis, generalization_deltas, paired_live_analysis, write_primary_analysis)
from evaluation.experiment_stability import stability_analysis
from evaluation.live_benchmark import run_live


@pytest.fixture(autouse=True)
def prohibit_live_and_training(monkeypatch):
    def forbidden(*args,**kwargs): pytest.fail('External network and ML fitting are prohibited')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(MLRouter,'fit',forbidden)


@pytest.fixture(scope='module')
def sources(tmp_path_factory):
    router = ModelRouter(runtime_dir=tmp_path_factory.mktemp('final-router'))
    bases = {k:FrozenBaseline(k,router) for k in KINDS}
    cfg = experiment_config()
    document = build_sample(bases,cfg)
    return router,bases,cfg,document


def rehash(document):
    document['manifest']['sample_hash'] = fingerprint({k:document[k] for k in
        ['configuration','primary','stability_request_ids','exclusions']})


def test_exact_sizes_unique_ids_frozen_content_and_replay(sources):
    _,bases,cfg,document = sources
    ids = []
    for kind,part in document['primary'].items():
        assert len(part['rows'])==200
        ids.extend(r['request']['request_id'] for r in part['rows'])
        originals = {r['request']['request_id']:r for r in bases[kind].rows}
        for row,label,item in zip(part['rows'],part['reference_labels'],part['items']):
            rid = row['request']['request_id']
            assert row==originals[rid] and label==bases[kind].labels[rid]
            canonical = bases[kind].prepared[rid].canonical()
            assert item['canonical_input_hash']==fingerprint(canonical.model_dump(mode='json'))
            assert item['eligible_models']==[m.model_id for m in canonical.eligible_models]
            for strategy in ['rules','weighted','ml']:
                result = bases[kind].replay(rid,strategy)
                assert result['research_eligible'] and result['execution_mode']=='BASELINE_REPLAY'
    assert len(ids)==len(set(ids))==400
    validate_sample(document,bases,cfg)


def test_deterministic_seeds_hashes_and_proportions(sources):
    _,bases,cfg,document = sources
    same = build_sample(bases,cfg)
    assert same['manifest']['sample_hash']==document['manifest']['sample_hash']
    different = build_sample(bases,{**cfg,'sampling_seed':cfg['sampling_seed']+1})
    assert different['manifest']['sample_hash']!=document['manifest']['sample_hash']
    assert different['manifest']['benchmarks']['IID_SYNTHETIC']['sampled_request_ids']!=document['manifest']['benchmarks']['IID_SYNTHETIC']['sampled_request_ids']
    for kind,summary in document['manifest']['benchmarks'].items():
        source_count = summary['sampling_pool_size']
        for field in cfg['stratification_fields']:
            before,after = summary['distributions_before'][field],summary['distributions_after'][field]
            # Secondary margins are approximate, not tiny full-cross-product cells.
            assert max(abs(after.get(k,0)/200-count/source_count) for k,count in before.items())<.12
        assert summary['expected_no_route_count']==(51 if kind=='IID_SYNTHETIC' else 45)
        assert set(summary['template_ids'])<=set(summary['source_test_template_ids'])


def test_largest_remainder_known_fixture():
    assert largest_remainder({'a':4,'b':3,'c':3},5)=={'a':2,'b':2,'c':1}


def test_pilot_exclusions_are_complete_and_precede_sampling(sources):
    _,bases,cfg,doc = sources
    exclusion = pilot_exclusions(cfg)
    assert exclusion == doc['manifest']['exclusions'] == doc['exclusions']
    assert exclusion['exclusion_reason'] == 'PRIOR_LIVE_PILOT'
    assert len(exclusion['source_pilot_run_ids']) == 7
    assert exclusion['exclusion_set_hash'] == fingerprint({k:v for k,v in exclusion.items() if k!='exclusion_set_hash'})
    for kind,base in bases.items():
        entry = exclusion['benchmarks'][kind]
        excluded = set(entry['excluded_request_ids'])
        assert len(excluded) == entry['excluded_request_count'] == 15
        assert entry['source_pilot_run_ids'] == exclusion['source_pilot_run_ids']
        # Includes the policy NO_ROUTE cases too, regardless of API success.
        assert sum(base.labels[rid]['expected_no_route'] for rid in excluded) == (4 if kind=='IID_SYNTHETIC' else 3)
        pool = [item_for(base,r) for r in base.rows if r['request']['request_id'] not in excluded]
        expected,_ = stratified_sample(pool,200,4042,cfg['stratification_fields'],cfg['minimum_secondary_allocation'])
        assert doc['primary'][kind]['items'] == expected
        assert not excluded & {i['request_id'] for i in expected}
        assert not excluded & set(doc['stability_request_ids'][kind])
        assert doc['manifest']['benchmarks'][kind]['sampling_pool_size'] == len(base.rows)-15


def test_pilot_request_injected_into_sample_fails(sources):
    _,bases,cfg,original = sources
    doc = copy.deepcopy(original)
    kind = 'IID_SYNTHETIC'
    rid = doc['exclusions']['benchmarks'][kind]['excluded_request_ids'][0]
    doc['primary'][kind]['rows'][0] = copy.deepcopy(next(r for r in bases[kind].rows if r['request']['request_id']==rid))
    rehash(doc)
    with pytest.raises(ValueError,match='Prior live pilot request leaked'):
        validate_sample(doc,bases,cfg)


def test_exclusion_provenance_tampering_fails(sources):
    _,bases,cfg,original = sources
    doc = copy.deepcopy(original)
    doc['exclusions']['benchmarks']['IID_SYNTHETIC']['excluded_request_ids'].pop()
    rehash(doc)
    with pytest.raises(ValueError,match='exclusion provenance'):
        validate_sample(doc,bases,cfg)


@pytest.mark.parametrize('mutation', ['count','duplicate','reference','canonical','candidates','template','provenance','unknown_request'])
def test_sample_tampering_fails(sources,mutation):
    _,bases,cfg,original = sources
    doc = copy.deepcopy(original)
    part = doc['primary']['TEMPLATE_HELD_OUT']
    if mutation=='count': part['rows'].pop()
    elif mutation=='duplicate': part['rows'][1]=copy.deepcopy(part['rows'][0])
    elif mutation=='reference': part['reference_labels'][0]['preferred_model']='invented'
    elif mutation=='canonical': part['items'][0]['canonical_input_hash']='wrong'
    elif mutation=='candidates': part['items'][0]['eligible_models']=['invented']
    elif mutation=='template': part['rows'][0].pop('template_id')
    elif mutation=='provenance': doc['manifest']['benchmarks']['TEMPLATE_HELD_OUT'].pop('source_test_template_ids')
    else: part['rows'][0]['request']['request_id']='not-in-source'
    rehash(doc)  # Even a recomputed container hash cannot authorize altered source data.
    with pytest.raises(ValueError): validate_sample(doc,bases,cfg)


@pytest.mark.parametrize('field',['dataset_hash','results_hash','source_run_id'])
def test_pinned_source_checks(sources,field):
    _,bases,cfg,_ = sources
    bad = copy.deepcopy(cfg); bad['sources']['IID_SYNTHETIC'][field]='changed'
    with pytest.raises(ValueError,match='Pinned'): build_sample(bases,bad)


def test_missing_replay_fails(sources):
    _,bases,cfg,document = sources
    kind='IID_SYNTHETIC'; rid=document['primary'][kind]['items'][0]['request_id']
    changed = copy.copy(bases[kind]); changed.results=dict(changed.results)
    changed.results[(rid,'ml')]=[]
    with pytest.raises(ValueError,match='replay'):
        validate_sample(document,{**bases,kind:changed},cfg)


def test_stability_subset_and_500_planned_calls(sources):
    _,_,cfg,doc=sources
    items=experiment_items(doc,'stability')
    assert cfg['repetitions']==5 and len(items)*2==500
    for kind in KINDS:
        stable=doc['stability_request_ids'][kind]
        assert len(stable)==len(set(stable))==25
        routable={i['request_id'] for i in doc['primary'][kind]['items'] if not i['expected_no_route']}
        assert set(stable)<=routable
    assert not any(i['expected_no_route'] for i in items)
    assert set(i['repetition_number'] for i in items)=={1,2,3,4,5}


def test_sample_persistence_never_overwrites(sources,tmp_path):
    router,bases,cfg,_=sources
    _,doc,path=prepare_experiment(router,cfg,tmp_path,bases)
    initial=path.read_bytes()
    _,again,_=prepare_experiment(router,cfg,tmp_path,bases)
    assert again==doc and path.read_bytes()==initial
    saved=json.loads(initial); saved['manifest']['benchmarks']['IID_SYNTHETIC']['routable_count']=0
    path.write_text(json.dumps(saved))  # Only the test's temporary sample, never frozen input.
    with pytest.raises(ValueError): prepare_experiment(router,cfg,tmp_path,bases)


@pytest.mark.parametrize('mode',['final','stability'])
def test_preflight_zero_dispatch_and_presence_only(sources,tmp_path,monkeypatch,mode):
    router,bases,_,_=sources
    def forbidden(*args,**kwargs): pytest.fail('Preflight must not dispatch')
    monkeypatch.setattr(OpenAILiveRouter,'execute',forbidden)
    monkeypatch.setattr(JevLiveRouter,'execute',forbidden)
    monkeypatch.setenv('OPENROUTER_API_KEY','secret-preflight')
    result=preflight(mode,sample_root=tmp_path,router=router,baselines=bases)
    assert result['external_api_calls_made']==0
    assert result['selected_planned_external_calls']==(608 if mode=='final' else 500)
    assert result['providers']['jev']['key_present']
    assert 'secret-preflight' not in json.dumps(result)
    assert result['providers']['jev']['execution_path']=='JEV_VIA_OPENROUTER'
    assert result['providers']['jev']['execution_provider']=='openrouter'
    assert result['providers']['jev']['strategy']=='jev'
    assert result['prior_live_pilot_overlap']=={k:0 for k in KINDS}
    assert result['exclusions']['exclusion_reason']=='PRIOR_LIVE_PILOT'
    assert result['providers']['openai']['execution_path']=='OPENAI_DIRECT'
    assert result['pricing_config_hash']


@pytest.fixture
def execution(sources,tmp_path,monkeypatch):
    """Tiny dispatch fixture; full 400/50 selection is independently validated above."""
    router,bases,cfg,document=sources
    monkeypatch.setenv('OPENAI_API_KEY','secret-openai-final')
    monkeypatch.setenv('OPENROUTER_API_KEY','secret-openrouter-final')
    monkeypatch.setenv('OPENAI_ROUTER_ENABLED','true')
    monkeypatch.setenv('OPENAI_ROUTER_MODEL','gpt-4.1-mini-2025-04-14')
    monkeypatch.setenv('JEV_ROUTER_ENABLED','true'); monkeypatch.setenv('JEV_MOCK_MODE','false')
    adapters={'openai':OpenAILiveRouter(),'jev':JevLiveRouter()}
    calls=[]
    for strategy,adapter in adapters.items():
        def once(value,strategy=strategy):
            calls.append(strategy)
            return {'selected_model':value.eligible_models[0].model_id,'confidence':.8,'reason_codes':[]}, {
                'api_status':200,'returned_model':'gpt-4.1-mini-2025-04-14' if strategy=='openai' else 'typesafe/jev-test',
                'input_tokens':100,'output_tokens':10,'total_tokens':110,'cached_input_tokens':0,
                'provider_reported_cost_usd':.001 if strategy=='jev' else None,'service_tier':'default'}
        monkeypatch.setattr(adapter,'call_once',once)
    from evaluation import experiment_sampling,experiment_statistics
    selected=[next(i for i in document['primary'][k]['items'] if i['eligible_models']) for k in KINDS]
    monkeypatch.setattr(experiment_sampling,'experiment_items',lambda doc,mode:selected if mode=='final' else
        [{**i,'repetition_number':r} for r in range(1,6) for i in selected])
    monkeypatch.setattr(experiment_statistics,'write_primary_analysis',lambda *a:None)
    def run(mode='final',**kwargs):
        return run_live(router=router,baselines=bases,adapters=adapters,experiment_mode=mode,
                        experiment_cfg=cfg,sample_root=tmp_path/'samples',output_root=tmp_path/'reports',**kwargs)
    return run,calls,adapters


def test_permission_and_pre_execution_cap(execution):
    run,calls,_=execution
    with pytest.raises(ValueError,match='permission|allow-live-api'): run()
    with pytest.raises(ValueError,match='cap exceeded'): run(allow_live_api=True,max_live_api_calls=1)
    assert calls==[]


@pytest.mark.parametrize('mode',['final','stability'])
def test_resume_force_and_observation_identity(execution,mode):
    run,calls,_=execution
    directory,_=run(mode,allow_live_api=True,max_live_api_calls=100)
    initial=len(calls)
    assert initial==(4 if mode=='final' else 20)
    run(mode,allow_live_api=True,max_live_api_calls=100,resume=directory)
    assert len(calls)==initial
    run(mode,allow_live_api=True,max_live_api_calls=100,resume=directory,force_rerun=True)
    assert len(calls)==2*initial
    frame=pd.read_csv(directory/'benchmark_results.csv')
    assert not frame.duplicated(['benchmark_type','request_id','strategy']+(['repetition_number'] if mode=='stability' else [])).any()
    assert frame[frame.strategy=='jev'].execution_path.eq('JEV_VIA_OPENROUTER').all()
    assert frame[frame.strategy=='jev'].execution_provider.eq('openrouter').all()
    assert frame[frame.strategy=='openai'].execution_path.eq('OPENAI_DIRECT').all()
    manifest=json.loads((directory/'run_manifest.json').read_text())
    assert manifest['pricing_config_hash'] and manifest['statistical_configuration']['bootstrap_iterations']==10000
    assert manifest['experiment']['sample_manifest']['sample_hash']
    if mode=='stability':
        assert (directory/'stability_results.csv').exists()
        assert (directory/'selection_consistency.csv').exists()
    for file in directory.iterdir():
        if file.is_file():
            assert 'secret-openai-final' not in file.read_text()
            assert 'secret-openrouter-final' not in file.read_text()


def test_spend_cap_halts_next_call(execution):
    run,calls,_=execution
    directory,_=run(allow_live_api=True,max_routing_spend_usd=.000001)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    assert manifest['status']=='BUDGET_STOP' and manifest['api_call_count']==1 and len(calls)==1


def test_missing_provider_refuses_execution(execution,monkeypatch):
    run,calls,_=execution
    monkeypatch.delenv('OPENROUTER_API_KEY')
    with pytest.raises(ValueError,match='unavailable'): run(allow_live_api=True)
    assert calls==[]


def test_mcnemar_known_fixture():
    result=mcnemar([1]*9+[0],[0]*9+[1])
    assert result['a_correct_b_wrong']==9 and result['a_wrong_b_correct']==1
    assert result['mcnemar_statistic']==1
    assert result['raw_p_value']==pytest.approx(22/1024)
    assert mcnemar([1,0],[1,0])['raw_p_value']==1


def test_holm_known_fixture():
    assert holm_bonferroni([.01,.04,.03])==pytest.approx([.03,.06,.06])


def test_bootstrap_reproducible():
    a=bootstrap_ci([0,1,1,0,1],123,10000)
    assert a==bootstrap_ci([0,1,1,0,1],123,10000)
    assert 0<=a[0]<=.6<=a[1]<=1
    assert bootstrap_ci([],123)==(None,None)


def test_wilcoxon_known_fixture():
    statistic,p=paired_wilcoxon([1,2,3,4,5,6],[0]*6)
    assert statistic==0 and p==pytest.approx(.03125)
    assert paired_wilcoxon([1,2],[1,2])==(0,1)


def fixture_frame(repetitions=1):
    rows=[]
    for kind in KINDS:
        for i in range(8):
            for strategy in ['rules','weighted','ml','openai','jev']:
                for repeat in range(1,repetitions+1):
                    live=strategy in ['openai','jev']
                    rows.append({'benchmark_type':kind,'request_id':f'{kind}-{i}','strategy':strategy,
                        'repetition_number':repeat,'status':'ROUTED','selected_model':'a' if repeat<5 else 'b',
                        'preferred_correct':i%2==0,'acceptable_correct':i%3!=0,'expected_no_route':False,
                        'execution_mode':'LIVE' if live else 'BASELINE_REPLAY',
                        'latency_measurement_source':'CURRENT_RUN' if live else 'FROZEN_BASELINE',
                        'routing_latency_ms':i+repeat+(1 if strategy=='openai' else 0),
                        'api_calls':int(live),'routing_cost_usd':.001 if live else 0,'retry_count':0,
                        'input_tokens':10 if live else None,'cached_input_tokens':0 if live else None,
                        'output_tokens':5 if live else None,'total_tokens':15 if live else None,
                        'ineligible_selection_attempts':0,'policy_violation':False})
    return pd.DataFrame(rows)


def test_analysis_pairs_correction_and_no_replay_latency_claims():
    result=primary_analysis(fixture_frame(),experiment_config()['statistics'])
    comparisons=result['statistical_comparisons.csv']
    assert len(comparisons)==40  # 10 pairs x 2 benchmarks x 2 agreement objectives.
    assert comparisons.n_paired_observations.eq(8).all()
    assert comparisons.adjusted_p_value.eq(1).all()
    live=result['paired_live_comparisons.csv']
    assert live.router_a.eq('openai').all() and live.router_b.eq('jev').all()
    assert len(result['confidence_intervals.csv'])==40
    assert len(result['generalization_deltas.csv'])==20
    assert result['generalization_deltas.csv'].held_out_minus_iid_pp.eq(0).all()
    replay=fixture_frame(); replay['execution_mode']='BASELINE_REPLAY'
    assert paired_live_analysis(replay).empty


def test_generalization_delta_known_fixture():
    frame=pd.DataFrame([{'benchmark_type':k,'strategy':'ml','metric':'preferred_correct','scope':'ROUTED_ONLY',
                         'agreement':v} for k,v in zip(KINDS,[.9,.8])])
    assert generalization_deltas(frame).iloc[0].held_out_minus_iid_pp==pytest.approx(-10)


def test_repetitions_cannot_enter_primary_statistics():
    with pytest.raises(ValueError,match='repeated'):
        primary_analysis(fixture_frame(5),experiment_config()['statistics'])


def test_stability_metrics_and_failure_denominators():
    frame=fixture_frame(5)
    frame=frame[frame.strategy.isin(['openai','jev'])]
    result=stability_analysis(frame)
    per_request=result['selection_consistency.csv']
    assert per_request.modal_selection_rate.eq(.8).all()
    assert per_request.pairwise_repetition_agreement.eq(.6).all()
    assert per_request.unique_selected_models.eq(2).all()
    assert not per_request.full_consistency.any()
    frame.loc[frame.repetition_number>1,'status']='TIMEOUT'
    result=stability_analysis(frame)
    assert result['stability_summary.csv'].full_consistency_rate.isna().all()
    assert result['stability_summary.csv'].requests_with_one_success.eq(8).all()


def test_statistics_require_completed_experiment(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'experiment_mode':'final','status':'RUNNING'}))
    with pytest.raises(ValueError,match='completed'):
        write_primary_analysis(tmp_path,experiment_config()['statistics'])


@pytest.mark.parametrize('dry_run',[False,True])
def test_cli_without_permission_is_preflight_only(monkeypatch,capsys,dry_run):
    from evaluation import benchmark,experiment_preflight,live_benchmark
    import sys
    args=['benchmark','--final-experiment']+(['--dry-run'] if dry_run else [])
    monkeypatch.setattr(sys,'argv',args)
    monkeypatch.setattr(experiment_preflight,'preflight',lambda *a,**kw:{'external_api_calls_made':0})
    monkeypatch.setattr(live_benchmark,'run_live',lambda **kw:pytest.fail('CLI dispatched without permission'))
    benchmark.main()
    assert '"external_api_calls_made": 0' in capsys.readouterr().out


def test_primary_analysis_files_only_for_matching_completed_sample(tmp_path):
    # Explicit synthetic test fixture in pytest's temporary directory, never a research run.
    frame=fixture_frame()
    parts=[]
    for offset in range(25):
        part=frame.copy()
        part['request_id']=part.request_id+'-'+str(offset)
        parts.append(part)
    frame=pd.concat(parts,ignore_index=True)
    expected={k:{'sampled_request_ids':frame[frame.benchmark_type==k].request_id.unique().tolist()} for k in KINDS}
    manifest={'experiment_mode':'final','status':'COMPLETED',
              'experiment':{'sample_manifest':{'benchmarks':expected}},
              'strategy_versions':{s:'test-fixture' for s in frame.strategy.unique()}}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    frame.to_csv(tmp_path/'benchmark_results.csv',index=False)
    write_primary_analysis(tmp_path,experiment_config()['statistics'])
    assert len(pd.read_csv(tmp_path/'statistical_comparisons.csv'))==40
    assert len(pd.read_csv(tmp_path/'confidence_intervals.csv'))==40
    frame.iloc[:-1].to_csv(tmp_path/'benchmark_results.csv',index=False)
    with pytest.raises(ValueError,match='exact 200'): write_primary_analysis(tmp_path,experiment_config()['statistics'])

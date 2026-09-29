"""Live runner tests use the frozen files read-only, mock all providers, and never fit ML."""
import json
from collections import Counter
import pandas as pd
import pytest
from app.core.config_loader import ROOT, fingerprint
from app.strategies.llm import OpenAILiveRouter
from app.strategies.live_jev import JevLiveRouter
from app.strategies.live_base import live_settings, ProviderFailure
from app.strategies.ml import MLRouter
from evaluation.frozen_baseline import FrozenBaseline
from evaluation.live_benchmark import run_live, make_plan
from evaluation.live_store import LiveJournal


@pytest.fixture
def setup(router,monkeypatch):
    def no_fit(*args,**kwargs): pytest.fail('Live comparison must NEVER train/reconstruct ML')
    monkeypatch.setattr(MLRouter,'fit',no_fit)
    monkeypatch.setenv('OPENAI_API_KEY','sensitive-openai-key')
    monkeypatch.setenv('OPENROUTER_API_KEY','sensitive-openrouter-key')
    monkeypatch.setenv('OPENAI_ROUTER_ENABLED','true')
    monkeypatch.setenv('OPENAI_ROUTER_MODEL','mock-model')
    monkeypatch.setenv('JEV_ROUTER_ENABLED','true')
    monkeypatch.setenv('JEV_MOCK_MODE','false')
    bases={k:FrozenBaseline(k,router) for k in ['IID_SYNTHETIC','TEMPLATE_HELD_OUT']}
    adapters={'openai':OpenAILiveRouter(), 'jev':JevLiveRouter()}
    captured=[]
    for name,adapter in adapters.items():
        adapter.settings['backoff_seconds']=0
        def call(value,name=name):
            captured.append((name,value.model_dump(mode='json')))
            return {'selected_model':value.eligible_models[0].model_id,'confidence':.8 if name=='openai' else None,'reason_codes':[]}, {
                'input_tokens':10,'output_tokens':5,'total_tokens':15,'cached_tokens':0,'api_status':200,
                'api_request_id':'req-test','returned_model':'mock-model', 'service_tier':'default'}
        monkeypatch.setattr(adapter,'call_once',call)
    return router,bases,adapters,captured


def run(setup,tmp_path,**kwargs):
    router,bases,adapters,_=setup
    return run_live(router=router,baselines=bases,adapters=adapters,output_root=tmp_path,**kwargs)


def test_permission_and_cap_preflight(setup,tmp_path):
    with pytest.raises(ValueError,match='allow-live-api'):
        run(setup,tmp_path)
    with pytest.raises(ValueError,match='cap exceeded before'):
        run(setup,tmp_path,allow_live_api=True,max_live_api_calls=1)
    assert not setup[3]
    assert not list(tmp_path.glob('*/run_manifest.json'))


def test_pilot_replay_fairness_resume_and_force(setup,tmp_path):
    directory,summary=run(setup,tmp_path,allow_live_api=True)
    frame=pd.read_csv(directory/'benchmark_results.csv')
    assert len(frame)==150
    assert frame.groupby(['benchmark_type','request_id']).canonical_input_hash.nunique().eq(1).all()
    assert frame.groupby(['benchmark_type','request_id']).eligible_models.nunique().eq(1).all()
    baseline=frame[frame.strategy.isin(['rules','weighted','ml'])]
    assert baseline.execution_mode.eq('BASELINE_REPLAY').all()
    assert baseline.decision_source.eq('FROZEN_BASELINE').all()
    assert baseline.latency_measurement_source.eq('FROZEN_BASELINE').all()
    assert baseline.research_eligible.all()
    for row in baseline.itertuples():
        source=setup[1][row.benchmark_type].results[(row.request_id,row.strategy)][0]
        assert row.routing_latency_ms==pytest.approx(float(source['routing_latency_ms']))
        assert row.source_dataset_hash==source['dataset_hash']
    live=frame[frame.strategy.isin(['openai','jev'])]
    assert live[live.status=='NO_ROUTE'].api_calls.eq(0).all()
    assert live[live.status=='ROUTED'].execution_mode.eq('LIVE').all()
    assert live[live.status=='ROUTED'].evidence_type.eq('EMPIRICAL').all()
    assert live[live.status=='ROUTED'].research_eligible.all()
    assert not frame.policy_violation.any()
    count=len(setup[3]); assert count <= 50 and count>0
    for path in directory.iterdir():
        if path.is_file():
            assert 'sensitive-openai-key' not in path.read_text()
            assert 'sensitive-openrouter-key' not in path.read_text()
    run(setup,tmp_path,allow_live_api=True,resume=directory)
    assert len(setup[3])==count
    run(setup,tmp_path,allow_live_api=True,resume=directory,force_rerun=True,max_live_api_calls=100)
    assert len(setup[3])==2*count
    manifest=json.loads((directory/'run_manifest.json').read_text())
    assert manifest['api_call_count']==2*count
    assert set(manifest['execution_mode_per_strategy']['ml'])=={'BASELINE_REPLAY'}
    assert pd.read_csv(directory/'benchmark_results.csv').revision.eq(2).all()


def test_frozen_replay_missing_or_ambiguous(setup):
    base=setup[1]['IID_SYNTHETIC']
    rid=base.rows[0]['request']['request_id']
    originals=base.results[(rid,'ml')]
    base.results[(rid,'ml')]=originals*2
    assert base.replay(rid,'ml')['status']=='BASELINE_REPLAY_UNAVAILABLE'
    base.results[(rid,'ml')]=originals
    originals[0]['canonical_input_hash']='wrong'
    assert base.replay(rid,'ml')['status']=='BASELINE_REPLAY_UNAVAILABLE'


def test_sampling_deterministic_strata_and_frozen_hashes(setup):
    router,bases,_,_=setup
    _,a=make_plan(router,['openai'],baselines=bases)
    _,b=make_plan(router,['openai'],baselines=bases)
    assert a==b
    assert Counter(i['benchmark_type'] for i in a)=={'IID_SYNTHETIC':15,'TEMPLATE_HELD_OUT':15}
    for kind in bases:
        group=[i for i in a if i['benchmark_type']==kind]
        assert any(not i['eligible_models'] for i in group) and any(i['eligible_models'] for i in group)
        assert len({i['data_classification'] for i in group})>=3
        assert len({i['task_type'] for i in group})>=8


def test_invalid_live_selection_recorded_as_blocked(setup,tmp_path,monkeypatch):
    monkeypatch.setattr(setup[2]['openai'],'call_once',lambda value: ({'selected_model':'outside','confidence':.5,'reason_codes':[]},{'api_status':200}))
    directory,summary=run(setup,tmp_path,allow_live_api=True,strategies=['openai'])
    frame=pd.read_csv(directory/'benchmark_results.csv')
    invalid=frame[frame.status=='INVALID_DECISION']
    assert len(invalid)>0 and invalid.ineligible_selection_attempts.eq(1).all()
    assert invalid.selected_model.isna().all() and not frame.policy_violation.any()
    assert invalid.error_type.eq('INELIGIBLE_MODEL_SELECTED').all()


def test_resume_refuses_changed_model(setup,tmp_path,monkeypatch):
    directory,_=run(setup,tmp_path,allow_live_api=True,strategies=['openai'])
    monkeypatch.setenv('OPENAI_ROUTER_MODEL','different-model')
    with pytest.raises(ValueError,match='Resume refused'):
        run(setup,tmp_path,allow_live_api=True,strategies=['openai'],resume=directory)


def test_interrupted_attempt_not_reissued(setup,tmp_path,monkeypatch):
    adapter=setup[2]['openai']; original=adapter.call_once
    def interrupted(value): raise KeyboardInterrupt()
    monkeypatch.setattr(adapter,'call_once',interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(setup,tmp_path,allow_live_api=True,strategies=['openai'])
    directory=next(tmp_path.glob('*/run_manifest.json')).parent
    monkeypatch.setattr(adapter,'call_once',original)
    run(setup,tmp_path,allow_live_api=True,strategies=['openai'],resume=directory)
    frame=pd.read_csv(directory/'benchmark_results.csv')
    assert (frame.status=='INDETERMINATE').sum()==1
    assert json.loads((directory/'run_manifest.json').read_text())['status']=='INCOMPLETE'


def test_spend_configuration_required(setup,tmp_path):
    with pytest.raises(ValueError,match='documented pricing'):
        run(setup,tmp_path,allow_live_api=True,strategies=['openai'],max_routing_spend_usd=1)
    assert not setup[3]


def test_journal_budget_and_redaction(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','private-key')
    journal=LiveJournal(tmp_path,1)
    journal.reserve('key',1)
    from app.strategies.live_base import BudgetStop
    with pytest.raises(BudgetStop): journal.reserve('key',1)
    journal.append({'event':'attempt_result','key':'key','revision':1,'data':{'routing_cost_usd':.1,'error_type':'private-key'}})
    assert 'private-key' not in journal.path.read_text()
    journal=LiveJournal(tmp_path,10,.05)
    with pytest.raises(BudgetStop): journal.reserve('next',1)


def test_actual_spend_threshold_stops_next_call(setup,tmp_path,monkeypatch):
    from evaluation import live_benchmark
    cfg=live_benchmark.load_config('live_routing')
    pricing={'input_per_million_tokens':100,'output_per_million_tokens':100,'cached_input_per_million_tokens':100,
             'source':'mock pricing fixture','version':'test-1','date':'2026-09-29'}
    cfg['routing_api_pricing']['openai']['mock-model']=pricing
    real_load=live_benchmark.load_config
    monkeypatch.setattr(live_benchmark,'load_config',lambda name:cfg if name=='live_routing' else real_load(name))
    setup[2]['openai'].settings['pricing']=pricing
    directory,_=run(setup,tmp_path,allow_live_api=True,strategies=['openai'],max_routing_spend_usd=.0001)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    assert manifest['status']=='BUDGET_STOP' and manifest['api_call_count']==1
    assert manifest['routing_cost_usd']==pytest.approx(.0015)
    assert len(setup[3])==1  # Threshold is checked after each observed charge, not an invented upper bound.


def test_retries_never_exceed_call_cap(setup,tmp_path,monkeypatch):
    adapter=setup[2]['openai']
    def always_limited(value): raise ProviderFailure('RATE_LIMIT',429)
    monkeypatch.setattr(adapter,'call_once',always_limited)
    _,items=make_plan(setup[0],['openai'],baselines=setup[1])
    cap=sum(bool(i['eligible_models']) for i in items)
    directory,_=run(setup,tmp_path,allow_live_api=True,strategies=['openai'],max_live_api_calls=cap)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    assert manifest['status']=='BUDGET_STOP'
    assert manifest['api_call_count']==cap


def test_live_streamlit_visibility(setup,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from evaluation import live_reporting
    directory,_=run(setup,tmp_path,allow_live_api=True,strategies=['ml','openai'])
    monkeypatch.setattr(live_reporting,'live_report_manifests',lambda:[directory/'run_manifest.json'])
    app=AppTest.from_file(str(ROOT/'ui/streamlit_app.py'),default_timeout=30).run()
    app.sidebar.radio[0].set_value('Live / Research').run()
    assert not app.exception
    assert app.dataframe
    assert any('Empirical live routing execution' in text.value for text in app.info)
    assert any('not a newly measured' in text.value for text in app.caption)


def test_pending_force_revision_does_not_hide_behind_older_success(tmp_path):
    journal=LiveJournal(tmp_path,10)
    journal.reserve('obs',1)
    journal.append({'event':'observation','key':'obs','revision':1,'row':{'revision':1,'status':'ROUTED'}})
    assert not journal.pending('obs')
    journal.reserve('obs',2)
    assert journal.pending('obs')

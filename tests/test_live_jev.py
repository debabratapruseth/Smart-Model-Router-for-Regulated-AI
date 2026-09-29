import json
from types import SimpleNamespace as NS
import pytest
from app.strategies.live_base import live_settings
from app.strategies.live_jev import JevLiveRouter, RoutingChoiceResponse
from app.strategies import live_jev


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','test-typesafe-key')
    monkeypatch.setenv('JEV_MOCK_MODE','false')
    cfg=live_settings('jev');cfg.update(enabled=True,backoff_seconds=0)
    return JevLiveRouter(cfg)


@pytest.mark.parametrize('confidence',[None,.73])
def test_verified_choice_contract(adapter,router,request_factory,monkeypatch,confidence):
    value=router.prepare(request_factory()).canonical()
    captured={}
    class Client:
        def __init__(self,**kwargs): assert kwargs['retry'].max_retries==0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def system_one(self,**kwargs):
            captured.update(kwargs)
            return NS(model='jev-returned-version',request_id='typesafe_req_1',usage=NS(input_tokens=100,output_tokens=5),
                      answers={'selected_model':NS(choice=value.eligible_models[0].model_id,confidence=confidence)})
    monkeypatch.setattr(live_jev,'TypeSafeClient',Client)
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='ROUTED' and obs.telemetry.research_eligible
    assert obs.decision.confidence==confidence and obs.decision.reason_codes==[]
    assert obs.telemetry.cached_tokens is None and obs.telemetry.routing_cost_usd is None
    assert captured['state']==value.model_dump(mode='json')
    assert set(captured['questions']['selected_model'].criteria)=={m.model_id for m in value.eligible_models}
    assert captured['response_model'] is RoutingChoiceResponse
    assert all(k not in captured['state'] for k in ['prompt','request_id','preferred_model','acceptable_models','reference'])


def test_missing_key_and_mock_mode_no_execution(adapter,router,request_factory,monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY')
    obs=adapter.execute(router.prepare(request_factory()).canonical(),allow_live_api=True)
    assert obs.status=='UNAVAILABLE' and obs.telemetry.api_calls==0 and not obs.telemetry.research_eligible
    monkeypatch.setenv('JEV_MOCK_MODE','true')
    result=router.route(request_factory(),'jev',allow_fallback=False)
    assert result.strategy_mode=='mock'


def test_jev_ineligible_choice_blocked(adapter,router,request_factory,monkeypatch):
    monkeypatch.setattr(adapter,'call_once',lambda value: ({'selected_model':'not-eligible','confidence':None,'reason_codes':[]},{}))
    obs=adapter.execute(router.prepare(request_factory()).canonical(),allow_live_api=True)
    assert obs.status=='INVALID_DECISION' and obs.telemetry.ineligible_selection_attempts==1
    assert obs.decision is None


def test_nullable_sdk_response_schema():
    value=RoutingChoiceResponse.model_validate({'model':'jev-latest','usage':{},'answers':{
        'selected_model':{'type':'choice','choice':'model-a','probabilities':{'model-a':1}}}})
    assert value.answers['selected_model'].confidence is None


def test_real_typesafe_sdk_mock_transport(adapter,router,request_factory,monkeypatch):
    import httpx2
    from typesafe_sdk import TypeSafeClient
    from app.core.config_loader import load_config
    assert load_config('live_routing')['jev']['model']=='typesafe/jev-1.13'
    adapter.settings['model']='typesafe/jev-1.13'
    value=router.prepare(request_factory()).canonical()
    def handle(request):
        payload=json.loads(request.content)
        assert request.url.path=='/api/v1/systemone'
        assert payload['model']=='typesafe/jev-1.13'
        assert payload['state']==value.model_dump(mode='json')
        assert payload['questions']['selected_model']['type']=='choice'
        return httpx2.Response(200,headers={'x-typesafe-request-id':'jev-wire-id'},json={
            'model':'jev-verified-wire','usage':{'input_tokens':100,'output_tokens':5},
            'answers':{'selected_model':{'type':'choice','choice':value.eligible_models[0].model_id,
                                         'probabilities':{value.eligible_models[0].model_id:1}}}})
    monkeypatch.setattr(live_jev,'TypeSafeClient',lambda **kwargs:TypeSafeClient(**kwargs,transport=httpx2.MockTransport(handle)))
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='ROUTED'
    assert obs.decision.confidence is None and obs.telemetry.api_request_id=='jev-wire-id'

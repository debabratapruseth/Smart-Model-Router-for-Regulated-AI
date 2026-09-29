"""Official SDK boundary tested entirely with local mocks."""
import json
from types import SimpleNamespace as NS
import httpx
import openai
import pytest
from app.models.live_routing import LiveRoutingDecision
from app.strategies.llm import OpenAILiveRouter
from app.strategies.live_base import live_settings


@pytest.fixture
def value(router, request_factory):
    return router.prepare(request_factory(prompt='RAW_PROMPT_NEVER_TRANSMIT_778899')).canonical()


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-key-test')
    cfg = live_settings('openai')
    cfg.update(enabled=True, model='configured-test-model', max_retries=2, backoff_seconds=0)
    return OpenAILiveRouter(cfg)


def sdk_mock(monkeypatch, sequence, captured=None):
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            self.responses = self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def parse(self, **kwargs):
            if captured is not None: captured.append(kwargs)
            item = sequence.pop(0)
            if isinstance(item, Exception): raise item
            return item
    monkeypatch.setattr(openai, 'OpenAI', Client)


def response(value, **changes):
    data = dict(status='completed', model='returned-test-model', _request_id='req_test_1',
                usage=NS(input_tokens=100, output_tokens=20, total_tokens=120, input_tokens_details=NS(cached_tokens=10)),
                output=[], output_parsed=LiveRoutingDecision(selected_model=value.eligible_models[0].model_id,
                                                           confidence=.8, reason_codes=['TASK_FIT']))
    return NS(**(data | changes))


def test_sdk_structured_response_and_payload(adapter, value, monkeypatch):
    calls=[]
    sdk_mock(monkeypatch,[response(value)],calls)
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='ROUTED' and obs.telemetry.research_eligible
    assert obs.telemetry.input_tokens==100 and obs.telemetry.routing_cost_usd is None
    assert calls[0]['text_format'] is LiveRoutingDecision
    assert 'temperature' not in calls[0]
    payload=json.loads(calls[0]['input'][0]['content'])
    assert payload==value.model_dump(mode='json')
    assert all(key not in payload for key in ['prompt','request_id','preferred_model','acceptable_models','reference','expected_answer'])
    assert 'RAW_PROMPT_NEVER_TRANSMIT' not in json.dumps(calls,default=str)


@pytest.mark.parametrize('permission,key',[(False,True),(True,False)])
def test_permission_and_missing_key(adapter,value,monkeypatch,permission,key):
    if not key: monkeypatch.delenv('OPENAI_API_KEY')
    sdk_mock(monkeypatch,[])
    obs=adapter.execute(value,allow_live_api=permission)
    assert obs.status=='UNAVAILABLE' and obs.telemetry.api_calls==0


def test_ineligible_and_refusal(adapter,value,monkeypatch):
    wrong=LiveRoutingDecision(selected_model='outside-candidates',confidence=.8,reason_codes=[])
    sdk_mock(monkeypatch,[response(value,output_parsed=wrong)])
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='INVALID_DECISION' and obs.decision is None
    assert obs.telemetry.ineligible_selection_attempts==1 and obs.telemetry.api_calls==1
    sdk_mock(monkeypatch,[response(value,output_parsed=None,output=[NS(content=[NS(type='refusal')])])])
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='REFUSAL' and obs.telemetry.refusal


def test_retry_and_permanent_errors(adapter,value,monkeypatch):
    request=httpx.Request('POST','https://api.openai.com/v1/responses')
    rate=openai.RateLimitError('unsafe secret-key-test',response=httpx.Response(429,request=request),body=None)
    calls=[]
    sdk_mock(monkeypatch,[rate,response(value)],calls)
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='ROUTED' and obs.telemetry.retry_count==1 and len(calls)==2
    assert obs.telemetry.input_tokens is None  # Unknown usage on failed first attempt.
    auth=openai.AuthenticationError('unsafe secret-key-test',response=httpx.Response(401,request=request),body=None)
    sdk_mock(monkeypatch,[auth],calls:=[])
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='AUTH_ERROR' and len(calls)==1
    assert 'secret-key-test' not in obs.model_dump_json()


def test_timeout_and_empty_candidates(adapter,value,monkeypatch):
    timeout=openai.APITimeoutError(request=httpx.Request('POST','https://api.openai.com/v1/responses'))
    sdk_mock(monkeypatch,[timeout,timeout,timeout])
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='TIMEOUT' and obs.telemetry.retry_count==2
    value.eligible_models=[]
    sdk_mock(monkeypatch,[])
    assert adapter.execute(value,allow_live_api=True).status=='NO_ROUTE'


def test_real_sdk_parses_mock_http_response(adapter,value,monkeypatch):
    """Exercise the actual installed SDK parser/schema serializer without networking."""
    real_client=openai.OpenAI
    def handle(request):
        payload=json.loads(request.content)
        assert payload['text']['format']['strict'] is True
        assert payload['text']['format']['type']=='json_schema'
        assert request.url.path=='/v1/responses'
        decision={'selected_model':value.eligible_models[0].model_id,'confidence':.8,'reason_codes':['TASK_FIT']}
        return httpx.Response(200,headers={'x-request-id':'req-wire'},json={
            'id':'resp_mock','object':'response','created_at':1,'status':'completed','model':'mock-returned',
            'output':[{'type':'message','id':'msg_mock','status':'completed','role':'assistant',
                       'content':[{'type':'output_text','text':json.dumps(decision),'annotations':[]}]}],
            'usage':{'input_tokens':20,'output_tokens':10,'total_tokens':30,'input_tokens_details':{'cached_tokens':0},
                     'output_tokens_details':{'reasoning_tokens':0}}})
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:real_client(**kwargs,http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='ROUTED'
    assert obs.telemetry.api_request_id=='req-wire' and obs.telemetry.total_tokens==30


@pytest.mark.parametrize('bad',[{'selected_model':'x','confidence':2,'reason_codes':[]},
                                {'selected_model':'x','confidence':.5,'reason_codes':['INVENTED']},
                                {'selected_model':'x','confidence':'0.5','reason_codes':[]}])
def test_strict_invalid_decision_no_retry(adapter,value,monkeypatch,bad):
    calls=[]
    def once(value):
        calls.append(1)
        return bad,{'api_status':200}
    monkeypatch.setattr(adapter,'call_once',once)
    obs=adapter.execute(value,allow_live_api=True)
    assert obs.status=='INVALID_DECISION' and len(calls)==1 and obs.decision is None


def test_usage_pricing_and_null_cache():
    from app.strategies.live_base import usage_cost
    rates={'input_per_million_tokens':2,'output_per_million_tokens':4,'cached_input_per_million_tokens':1,
           'source':'verified test fixture','version':'v1','date':'2026-09-29'}
    assert usage_cost({'input_tokens':100,'output_tokens':10,'cached_tokens':20},rates)==pytest.approx(.00022)
    assert usage_cost({'input_tokens':100,'output_tokens':10,'cached_tokens':None},rates) is None
    rates['cached_input_per_million_tokens']=2
    assert usage_cost({'input_tokens':100,'output_tokens':10,'cached_tokens':None},rates)==pytest.approx(.00024)

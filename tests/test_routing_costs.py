"""Cost telemetry only: real SDK parsing with in-memory transports, never paid calls."""
import json
import socket
from types import SimpleNamespace as NS
import httpx2
import pandas as pd
import pytest
from typesafe_sdk import TypeSafeClient
from app.core.config_loader import fingerprint
from app.models.live_routing import LiveRoutingDecision
from app.services.routing_costs import attempt_cost, pricing_snapshot
from app.strategies.live_base import live_settings, ProviderFailure
from app.strategies.llm import OpenAILiveRouter
from app.strategies.live_jev import JevLiveRouter
from app.strategies import live_jev, llm
from evaluation.live_benchmark import run_live
from evaluation.live_reporting import summarize_live
from evaluation.live_store import LiveJournal

MODEL = 'gpt-4.1-mini-2025-04-14'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Cost tests must never open a network connection')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def cost(**changes):
    usage = dict(returned_model=MODEL, service_tier='default', input_tokens=1000,
                 cached_input_tokens=0, output_tokens=100)
    usage.update(changes)
    return attempt_cost('openai', usage, {'pricing_catalog': pricing_snapshot()})


@pytest.mark.parametrize('cached,expected', [(0, .00056), (200, .0005), (1000, .00026)])
def test_openai_cache_split(cached, expected):
    result = cost(cached_input_tokens=cached)
    assert result['routing_cost_usd'] == pytest.approx(expected)
    assert result['cost_status'] == 'CALCULATED_FROM_PROVIDER_USAGE'


@pytest.mark.parametrize('field', ['input_tokens', 'cached_input_tokens', 'output_tokens'])
def test_null_usage(field):
    result = cost(**{field: None})
    assert result['routing_cost_usd'] is None
    assert result['cost_status'] == 'USAGE_NOT_RETURNED'


@pytest.mark.parametrize('changes', [{'returned_model': 'unknown'}, {'service_tier': 'priority'}, {'service_tier': None}])
def test_unconfigured_model_or_tier(changes):
    result = cost(**changes)
    assert result['routing_cost_usd'] is None
    assert result['cost_status'] == 'PRICING_NOT_CONFIGURED'


def test_invalid_cache_count():
    assert cost(cached_input_tokens=1001)['cost_status'] == 'INVALID_PROVIDER_USAGE'


@pytest.mark.parametrize('charge', [None, .00003, 0.0, -1, float('nan'), 'secret'])
def test_real_jev_sdk_preserves_cost(charge, router, request_factory, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret-openrouter')
    monkeypatch.setenv('JEV_MOCK_MODE', 'false')
    cfg = live_settings('jev'); cfg.update(enabled=True)
    value = router.prepare(request_factory()).canonical()
    def handle(request):
        assert str(request.url) == 'https://openrouter.ai/api/v1/systemone'
        assert json.loads(request.content)['questions']['selected_model']['type'] == 'choice'
        body = {'id': 'gen-cost-test', 'provider': 'TypeSafe', 'model': 'typesafe/jev-test',
                'usage': {'input_tokens': 100, 'output_tokens': 5, 'cost': charge},
                'answers': {'selected_model': {'type': 'choice', 'choice': value.eligible_models[0].model_id,
                                               'probabilities': {value.eligible_models[0].model_id: 1}}}}
        # JSON NaN is deliberately malformed telemetry, never a reason to lose a valid route.
        return httpx2.Response(200, content=json.dumps(body).encode(), headers={'content-type':'application/json'})
    monkeypatch.setattr(live_jev, 'TypeSafeClient',
                        lambda **kw: TypeSafeClient(**kw, transport=httpx2.MockTransport(handle)))
    obs = JevLiveRouter(cfg).execute(value, allow_live_api=True)
    assert obs.status == 'ROUTED'
    valid = type(charge) in (int, float) and charge >= 0
    assert obs.telemetry.cost_status == ('PROVIDER_REPORTED' if valid else 'PROVIDER_COST_UNAVAILABLE')
    assert obs.telemetry.routing_cost_usd == (charge if valid else None)
    assert obs.telemetry.cached_input_tokens is None
    assert 'secret' not in json.dumps(obs.telemetry.provider_usage)


def test_openai_usage_and_service_tier(router, request_factory, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-openai')
    cfg = live_settings('openai'); cfg.update(enabled=True, model=MODEL)
    value = router.prepare(request_factory()).canonical()
    response = NS(status='completed', model=MODEL, service_tier='default', output=[],
                  usage=NS(input_tokens=1000, output_tokens=100, total_tokens=1100,
                           input_tokens_details=NS(cached_tokens=200),
                           output_tokens_details=NS(reasoning_tokens=10), secret='secret-openai'),
                  output_parsed=LiveRoutingDecision(selected_model=value.eligible_models[0].model_id,
                                                   confidence=.8, reason_codes=[]))
    class Client:
        def __init__(self, **kwargs): self.responses = self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def parse(self, **kwargs): return response
    monkeypatch.setattr(llm.openai, 'OpenAI', Client)
    obs = OpenAILiveRouter(cfg).execute(value, allow_live_api=True)
    t = obs.telemetry
    assert t.routing_cost_usd == pytest.approx(.0005)
    assert t.uncached_input_tokens == 800 and t.cached_input_tokens == t.cached_tokens == 200
    assert t.total_tokens == 1100 and t.reasoning_tokens == 10
    assert t.service_tier == 'default' and t.pricing_model_id == MODEL
    assert 'secret-openai' not in t.model_dump_json()
    # Reasoning tokens are already within output_tokens, not charged twice.
    assert t.provider_usage[0]['output_tokens_details']['reasoning_tokens'] == 10


def test_no_route_zero_cost(router, request_factory):
    value = router.prepare(request_factory()).canonical(); value.eligible_models = []
    for adapter in [OpenAILiveRouter(), JevLiveRouter()]:
        obs = adapter.execute(value)
        assert obs.telemetry.routing_cost_usd == 0
        assert obs.telemetry.cost_status == 'NOT_APPLICABLE_NO_ROUTE'


def test_unknown_retry_cost_not_hidden(router, request_factory, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    cfg = live_settings('openai'); cfg.update(enabled=True, model=MODEL, backoff_seconds=0)
    adapter = OpenAILiveRouter(cfg)
    value = router.prepare(request_factory()).canonical()
    calls = []
    def once(value):
        calls.append(1)
        if len(calls) == 1: raise ProviderFailure('RATE_LIMIT', 429)
        return {'selected_model': value.eligible_models[0].model_id, 'confidence': .8, 'reason_codes': []}, {
            'api_status':200, 'returned_model':MODEL, 'service_tier':'default',
            'input_tokens':1000, 'cached_input_tokens':0, 'output_tokens':100, 'total_tokens':1100}
    monkeypatch.setattr(adapter, 'call_once', once)
    records = []
    obs = adapter.execute(value, allow_live_api=True, after_attempt=records.append)
    assert obs.status == 'ROUTED' and obs.telemetry.routing_cost_usd is None
    assert records[1]['routing_cost_usd'] == pytest.approx(.00056)
    assert records[0]['routing_cost_usd'] is None


def test_reports_provenance_replay_and_secret_redaction(router, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','secret-openai')
    monkeypatch.setenv('OPENROUTER_API_KEY','secret-openrouter')
    monkeypatch.setenv('OPENAI_ROUTER_ENABLED','true')
    monkeypatch.setenv('OPENAI_ROUTER_MODEL',MODEL)
    monkeypatch.setenv('JEV_ROUTER_ENABLED','true')
    monkeypatch.setenv('JEV_MOCK_MODE','false')
    adapters = {'openai': OpenAILiveRouter(), 'jev': JevLiveRouter()}
    for strategy, adapter in adapters.items():
        def once(value, strategy=strategy):
            meta = {'api_status':200, 'returned_model':MODEL if strategy=='openai' else 'typesafe/jev-test',
                    'input_tokens':1000, 'cached_input_tokens':0 if strategy=='openai' else None,
                    'output_tokens':100, 'total_tokens':1100, 'service_tier':'default',
                    'api_request_id':'secret-openrouter secret-openai',
                    'provider_reported_cost_usd':.00003 if strategy=='jev' else None}
            return {'selected_model':value.eligible_models[0].model_id, 'confidence':.8, 'reason_codes':[]}, meta
        monkeypatch.setattr(adapter, 'call_once', once)
    directory, summary = run_live(router=router, adapters=adapters, output_root=tmp_path, allow_live_api=True)
    frame = pd.read_csv(directory/'benchmark_results.csv')
    baseline = frame[frame.strategy.isin(['rules','weighted','ml'])]
    assert baseline.cost_status.eq('BASELINE_REPLAY_ZERO_EXTERNAL_COST').all()
    assert baseline.routing_cost_usd.eq(0).all()
    assert summary.unknown_cost_decisions.eq(0).all()
    assert summary[summary.strategy=='jev'].total_cached_input_tokens.isna().all()
    manifest = json.loads((directory/'run_manifest.json').read_text())
    config = pricing_snapshot()
    assert manifest['pricing_config_hash'] == fingerprint(config)
    assert manifest['pricing_configuration'] == config
    assert manifest['pricing_version'] == config['pricing_version']
    assert manifest['pricing_effective_date'] == config['effective_date']
    assert manifest['pricing_source'] == config['source']
    assert manifest['total_routing_cost_usd'] == pytest.approx(23*(.00056+.00003))
    for path in directory.iterdir():
        if path.is_file():
            assert 'secret-openai' not in path.read_text()
            assert 'secret-openrouter' not in path.read_text()
    # Unknown costs and token counts stay unknown in totals, never silently zeroed.
    frame.loc[(frame.strategy=='openai') & (frame.api_calls>0), 'routing_cost_usd'] = None
    result = summarize_live(frame)
    assert result[result.strategy=='openai'].total_routing_cost_usd.isna().all()
    assert result[result.strategy=='openai'].unknown_cost_decisions.sum() == 23


def test_jev_native_charge_stops_spend_budget(tmp_path):
    journal = LiveJournal(tmp_path, 10, .01)
    journal.reserve('one', 1)
    journal.append({'event':'attempt_result', 'key':'one', 'revision':1,
                    'data':{'routing_cost_usd':.02}})
    from app.strategies.live_base import BudgetStop
    with pytest.raises(BudgetStop): journal.reserve('two', 1)

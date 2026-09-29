"""Explicit opt-in, validation, retry and telemetry shared by live routing providers."""
import os
import logging
import math
from contextlib import contextmanager
import time
from datetime import datetime, timezone
from pydantic import ValidationError
from app.core.config_loader import load_config
from app.services.routing_costs import usage_cost, pricing_snapshot, pricing_provenance, attempt_cost, token_count
from app.models.live_routing import LiveRoutingDecision, LiveTelemetry, LiveObservation


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def live_settings(provider, config=None):
    cfg = config if config is not None else load_config('live_routing')
    if cfg['input_mode'] != 'STRUCTURED_ONLY' or cfg['research_mode_no_fallback'] is not True:
        raise ValueError('Live research requires STRUCTURED_ONLY and no fallback')
    settings = dict(cfg[provider])
    prefix = 'OPENAI' if provider == 'openai' else 'JEV'
    settings['enabled'] = os.getenv(f'{prefix}_ROUTER_ENABLED', str(settings['enabled'])).lower() == 'true'
    settings['model'] = os.getenv(f'{prefix}_ROUTER_MODEL') or settings['model']
    if (not isinstance(settings['max_retries'], int) or not 0 <= settings['max_retries'] <= 5 or
            not math.isfinite(settings['timeout_seconds']) or settings['timeout_seconds'] <= 0 or
            not math.isfinite(settings['backoff_seconds']) or settings['backoff_seconds'] < 0):
        raise ValueError('Invalid retry/timeout configuration')
    if provider == 'openai':
        if not isinstance(settings['max_output_tokens'], int) or settings['max_output_tokens'] < 1:
            raise ValueError('Invalid output-token limit')
        temperature = settings.get('temperature')
        if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError('Invalid temperature')
    settings['pricing_catalog'] = pricing_snapshot()
    settings['pricing'] = cfg['routing_api_pricing'][provider].get(settings['model'])
    return settings



@contextmanager
def quiet_sdk_logs():
    """SDK debug logs can contain raw bodies. Keep only our allowlisted telemetry."""
    prefixes = ('openai', 'typesafe_sdk', 'httpx', 'httpx2', 'httpcore', 'httpcore2')
    names = set(prefixes) | {name for name in logging.Logger.manager.loggerDict
                             if name.startswith(tuple(p + '.' for p in prefixes))}
    previous = []
    for name in names:
        logger = logging.getLogger(name)
        previous.append((logger, logger.level, logger.disabled))
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
    try:
        yield
    finally:
        for logger, level, disabled in previous:
            logger.setLevel(level)
            logger.disabled = disabled


class ProviderFailure(Exception):
    def __init__(self, code, status=None, request_id=None):
        self.code, self.status, self.request_id = code, status, request_id
        super().__init__(code)  # Never retain provider bodies, exception messages or credentials.


class BudgetStop(Exception):
    pass


class LiveRoutingStrategy:
    provider = ''
    strategy = ''
    prompt_version = ''
    key_env = ''

    def __init__(self, settings=None):
        self.settings = dict(settings or live_settings(self.strategy))
        self.settings.setdefault('pricing_catalog', pricing_snapshot())

    def configured(self):
        return bool(self.settings['enabled'] and self.settings['model'] and os.getenv(self.key_env))

    def call_once(self, canonical):
        """Return (decision dict or None, safe usage/response metadata). No SDK retries."""
        raise NotImplementedError

    def execute(self, canonical, *, allow_live_api=False, before_attempt=None, after_attempt=None):
        t = LiveTelemetry(provider=self.provider, strategy=self.strategy, router_model=self.settings.get('model') or '',
                          prompt_version=self.prompt_version, start_timestamp=utc_now())
        for key, value in pricing_provenance(self.settings['pricing_catalog']).items():
            setattr(t, key, value)
        start = time.perf_counter()

        def finish(status, decision=None):
            t.end_timestamp = utc_now()
            t.routing_latency_ms = (time.perf_counter()-start)*1000
            t.evidence_type = 'EMPIRICAL' if t.api_calls else 'SYNTHETIC'
            if not t.api_calls:
                t.routing_cost_usd = 0.0
                t.cost_status = 'NOT_APPLICABLE_NO_ROUTE' if status == 'NO_ROUTE' else 'NO_EXTERNAL_CALL'
            t.research_eligible = bool(status == 'ROUTED' and t.successful_api_calls)
            if status == 'ROUTED':
                t.error_type = t.error_message_sanitized = None
            if decision:
                t.selected_model, t.confidence, t.reason_codes = decision.selected_model, decision.confidence, decision.reason_codes
            return LiveObservation(status=status, decision=decision, telemetry=t)

        if not canonical.eligible_models:
            t.execution_mode = 'LOCAL'
            return finish('NO_ROUTE')
        if not allow_live_api:
            t.error_type = 'LIVE_PERMISSION_REQUIRED'
            t.error_message_sanitized = 'Explicit live API permission is required.'
            return finish('UNAVAILABLE')
        if not self.configured():
            t.error_type = 'MISSING_CONFIGURATION'
            t.error_message_sanitized = 'Provider enable flag, router model or API key is missing.'
            return finish('UNAVAILABLE')
        costs, usages, cost_details = [], [], []
        for attempt in range(self.settings['max_retries']+1):
            try:
                if before_attempt:
                    before_attempt()
            except BudgetStop:
                t.error_type = 'BUDGET_STOP'
                t.error_message_sanitized = 'API-call or observed-spend budget stopped this decision.'
                return finish('BUDGET_STOP')
            t.api_calls += 1
            t.retry_count = attempt
            attempt_start = time.perf_counter()
            meta, decision, error = {}, None, None
            try:
                with quiet_sdk_logs():
                    raw, meta = self.call_once(canonical)
                t.successful_api_calls += 1
                if meta.get('refusal'):
                    t.refusal = True
                    error = 'REFUSAL'
                elif meta.get('incomplete'):
                    error = 'INVALID_DECISION'
                else:
                    decision = LiveRoutingDecision.model_validate(raw)
                    if decision.selected_model not in {m.model_id for m in canonical.eligible_models}:
                        t.ineligible_selection_attempts = 1
                        error = 'INELIGIBLE_MODEL_SELECTED'
                    elif self.strategy == 'openai' and decision.confidence is None:
                        error = 'INVALID_DECISION'
            except ValidationError:
                error = 'INVALID_DECISION'
            except ProviderFailure as exc:
                error = exc.code
                meta.update(api_status=exc.status, api_request_id=exc.request_id)
                t.failed_api_calls += 1
            except Exception:
                # Fail closed without leaking exception text. Unexpected adapter bugs are not retried.
                error = 'API_FAILURE'
                t.failed_api_calls += 1
            elapsed = (time.perf_counter()-attempt_start)*1000
            if t.first_attempt_latency_ms is None:
                t.first_attempt_latency_ms = elapsed
            safe = {k: meta.get(k) for k in ['input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens',
                                           'cached_input_tokens', 'reasoning_tokens', 'service_tier', 'provider_reported_cost_usd',
                                           'provider_usage', 'api_request_id', 'api_status', 'returned_model']}
            safe['cached_input_tokens'] = token_count(meta.get('cached_input_tokens', meta.get('cached_tokens')))
            safe['cached_tokens'] = safe['cached_input_tokens']  # Legacy CSV alias.
            i, c = token_count(safe['input_tokens']), safe['cached_input_tokens']
            safe['uncached_input_tokens'] = i-c if i is not None and c is not None and c <= i else None
            detail = attempt_cost(self.strategy, safe, self.settings)
            costs.append(detail['routing_cost_usd'])
            cost_details.append(detail)
            usages.append(safe)
            t.provider_usage.append(safe.get('provider_usage') or {})
            for key, value in detail.items():
                setattr(t, key, value)
            # Totals are null if any attempt has unknown usage/cost; never under-report retries.
            for key in ['input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens', 'cached_input_tokens',
                        'uncached_input_tokens', 'reasoning_tokens', 'provider_reported_cost_usd']:
                values = [u[key] for u in usages]
                setattr(t, key, sum(values) if all(v is not None for v in values) else None)
            t.routing_cost_usd = sum(costs) if all(c is not None for c in costs) else None
            if t.routing_cost_usd is None:
                t.cost_status = next(d['cost_status'] for d in cost_details if d['routing_cost_usd'] is None)
            for key in ['api_request_id', 'api_status', 'returned_model', 'service_tier']:
                setattr(t, key, safe[key])
            if after_attempt:
                after_attempt({**safe, **detail, 'error_type': error,
                               'latency_ms': elapsed, 'api_success': bool(safe['api_status'] is not None and 200 <= safe['api_status'] < 300)})
            if error is None:
                return finish('ROUTED', decision)
            t.error_type = error
            t.error_message_sanitized = {
                'INELIGIBLE_MODEL_SELECTED': 'Provider selection was outside the eligible candidates and was blocked.',
                'REFUSAL': 'Provider refused the routing decision.',
                'INVALID_DECISION': 'Provider response did not satisfy the routing contract.',
            }.get(error, f'Provider call failed: {error}.')
            transient = error in {'TIMEOUT', 'RATE_LIMIT', 'TEMPORARY_SERVER_ERROR'}
            if not transient or attempt == self.settings['max_retries']:
                return finish('INVALID_DECISION' if error == 'INELIGIBLE_MODEL_SELECTED' else error)
            time.sleep(self.settings['backoff_seconds'] * (2**attempt))
        raise AssertionError('Unreachable retry state')

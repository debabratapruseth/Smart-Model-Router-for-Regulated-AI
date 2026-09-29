"""Official OpenAI Responses adapter. Only an explicit live runner can authorize calls."""
import json
import os
import openai
from app.models.live_routing import LiveRoutingDecision
from app.strategies.base import RoutingStrategy, StrategyUnavailable
from app.strategies.live_base import LiveRoutingStrategy, ProviderFailure
from app.services.routing_costs import token_count

OPENAI_ROUTER_PROMPT_V1 = (
    'You are an enterprise AI model-routing decision component. Select exactly one model from eligible_models. '
    'All supplied models have passed deterministic governance filtering. Optimize using the supplied task, '
    'quality requirement, cost preference, latency requirement and structured model metadata. '
    'Select only an eligible model ID. Return only the defined routing decision with controlled reason codes. '
    'Do not provide reasoning text.'
)


class OpenAILiveRouter(LiveRoutingStrategy):
    provider = 'OPENAI'
    strategy = 'openai'
    key_env = 'OPENAI_API_KEY'
    prompt_version = 'OPENAI_ROUTER_PROMPT_V1'

    def call_once(self, canonical):
        # SDK automatic retries disabled: the common layer owns the attempt/cost ledger.
        try:
            with openai.OpenAI(api_key=os.environ[self.key_env], base_url='https://api.openai.com/v1', max_retries=0,
                               timeout=self.settings['timeout_seconds']) as client:
                response = client.responses.parse(
                    model=self.settings['model'], instructions=OPENAI_ROUTER_PROMPT_V1,
                    input=[{'role': 'user', 'content': json.dumps(canonical.model_dump(mode='json'), sort_keys=True)}],
                    text_format=LiveRoutingDecision, max_output_tokens=self.settings['max_output_tokens'],
                    store=False,
                    **({"temperature": self.settings["temperature"]} if self.settings.get("temperature") is not None else {}),
                )
        except openai.APITimeoutError:
            raise ProviderFailure('TIMEOUT') from None
        except openai.APIStatusError as exc:
            status = exc.status_code
            code = ('AUTH_ERROR' if status in (401, 403) else 'RATE_LIMIT' if status == 429 else
                    'TEMPORARY_SERVER_ERROR' if 500 <= status <= 599 else 'API_FAILURE')
            raise ProviderFailure(code, status, exc.request_id) from None
        except openai.APIConnectionError:
            raise ProviderFailure('API_FAILURE') from None
        usage = response.usage
        details = getattr(usage, 'input_tokens_details', None)
        output_details = getattr(usage, 'output_tokens_details', None)
        # Allowlist numeric usage only; never retain prompts, headers or full responses.
        provider_usage = {
            'input_tokens': token_count(getattr(usage, 'input_tokens', None)),
            'output_tokens': token_count(getattr(usage, 'output_tokens', None)),
            'total_tokens': token_count(getattr(usage, 'total_tokens', None)),
            'input_tokens_details': {
                'cached_tokens': token_count(getattr(details, 'cached_tokens', None)),
                'cache_write_tokens': token_count(getattr(details, 'cache_write_tokens', None))},
            'output_tokens_details': {'reasoning_tokens': token_count(getattr(output_details, 'reasoning_tokens', None))}}
        meta = {
            'api_status': 200, 'api_request_id': getattr(response, '_request_id', None),
            'returned_model': response.model,
            'input_tokens': getattr(usage, 'input_tokens', None),
            'output_tokens': getattr(usage, 'output_tokens', None),
            'total_tokens': getattr(usage, 'total_tokens', None),
            'cached_tokens': getattr(details, 'cached_tokens', None),
            'cached_input_tokens': provider_usage['input_tokens_details']['cached_tokens'],
            'reasoning_tokens': provider_usage['output_tokens_details']['reasoning_tokens'],
            'service_tier': getattr(response, 'service_tier', None),
            'provider_usage': provider_usage,
            'incomplete': response.status != 'completed',
            'refusal': any(getattr(part, 'type', None) == 'refusal'
                           for output in response.output for part in getattr(output, 'content', [])),
        }
        parsed = response.output_parsed
        return parsed.model_dump() if parsed is not None else None, meta


class LLMRouter(RoutingStrategy):
    """Application/offline façade: credentials alone can never enable live calls."""
    mode = 'external'

    def available(self, request=None):
        return False

    def route_canonical(self, value):
        raise StrategyUnavailable('Use the explicit --allow-live-api live pilot runner')

    def route(self, request, features, candidate_models):
        raise StrategyUnavailable('Use the explicit --allow-live-api live pilot runner')

    def route_live(self, canonical, **kwargs):
        return OpenAILiveRouter().execute(canonical, **kwargs)

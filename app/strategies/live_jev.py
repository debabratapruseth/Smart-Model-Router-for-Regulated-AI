"""TypeSafe SDK System One Choice routing through OpenRouter."""
import os
from typesafe_sdk import (TypeSafeClient, Choice, ChoiceAnswer, SystemOneResponse, RetryPolicy,
                         TypeSafeAPITimeoutError, TypeSafeAPIError, TypeSafeAPIConnectionError,
                         TypeSafeAPIResponseValidationError, TypeSafeError)
from app.strategies.live_base import LiveRoutingStrategy, ProviderFailure
from app.services.routing_costs import nonnegative_number, token_count

JEV_ROUTER_DECISION_V1 = (
    'Select the eligible model best suited to the task, quality requirement, cost preference and latency requirement '
    'in the structured state. All candidates have passed deterministic policy filtering. '
    'Use only the supplied candidate metadata; choose exactly one eligible model.'
)


class NullableChoiceAnswer(ChoiceAnswer):
    # Native confidence is documented; never synthesize it if a compatible response omits it.
    confidence: float | None = None


class RoutingChoiceResponse(SystemOneResponse):
    id: str | None = None  # OpenRouter returns its request ID in the response body.
    answers: dict[str, NullableChoiceAnswer]


class JevLiveRouter(LiveRoutingStrategy):
    provider = 'TYPESAFE_JEV'
    strategy = 'jev'
    key_env = 'OPENROUTER_API_KEY'
    prompt_version = 'JEV_ROUTER_DECISION_V1'

    def configured(self):
        return super().configured() and os.getenv('JEV_MOCK_MODE', 'true').lower() == 'false'

    def call_once(self, canonical):
        try:
            with TypeSafeClient(api_key=os.environ[self.key_env], model=self.settings['model'],
                                base_url='https://openrouter.ai/api', timeout=self.settings['timeout_seconds'],
                                retry=RetryPolicy(max_retries=0)) as client:
                response = client.system_one(
                    state=canonical.model_dump(mode='json'),
                    questions={'selected_model': Choice(instructions=JEV_ROUTER_DECISION_V1,
                        criteria={model.model_id: None for model in canonical.eligible_models})},
                    response_model=RoutingChoiceResponse,
                )
        except TypeSafeAPITimeoutError:
            raise ProviderFailure('TIMEOUT') from None
        except TypeSafeAPIResponseValidationError as exc:
            raise ProviderFailure('INVALID_DECISION', exc.status, exc.request_id) from None
        except TypeSafeAPIError as exc:
            code = ('AUTH_ERROR' if exc.status in (401, 403) else 'RATE_LIMIT' if exc.status == 429 else
                    'TEMPORARY_SERVER_ERROR' if 500 <= exc.status <= 599 else 'API_FAILURE')
            raise ProviderFailure(code, exc.status, exc.request_id) from None
        except TypeSafeAPIConnectionError:
            raise ProviderFailure('API_FAILURE') from None
        answer = response.answers.get('selected_model')
        i, o = response.usage.input_tokens, response.usage.output_tokens
        # SDK 0.7.0 ignores extra Usage fields, but retains the original HTTP response.
        # Read only documented billing fields; missing/malformed cost must not affect routing.
        charge = None
        try:
            raw_usage = response.raw_http_response.json().get('usage', {})
            if isinstance(raw_usage, dict):
                charge = nonnegative_number(raw_usage.get('cost'))
        except (AttributeError, TypeSafeError, ValueError):
            pass
        request_id = getattr(response, 'id', None)
        if request_id is None:
            try:
                request_id = response.request_id
            except TypeSafeError:
                request_id = None  # Missing telemetry must not invalidate a routing decision.
        meta = {'api_status': 200, 'api_request_id': request_id, 'returned_model': response.model,
                'input_tokens': i, 'output_tokens': o, 'total_tokens': i+o if i is not None and o is not None else None,
                'cached_tokens': None, 'provider_reported_cost_usd': charge,
                'provider_usage': {'input_tokens': token_count(i), 'output_tokens': token_count(o), 'cost': charge}}
        result = None if answer is None else {'selected_model': answer.choice, 'confidence': answer.confidence,
                                              'reason_codes': []}
        return result, meta

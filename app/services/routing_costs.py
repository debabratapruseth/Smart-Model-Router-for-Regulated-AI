"""External routing API charges only; never downstream inference or local compute cost."""
import math
from app.core.config_loader import load_config, fingerprint


def nonnegative_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def token_count(value):
    return value if type(value) is int and value >= 0 else None


def pricing_snapshot():
    return load_config('provider_pricing')


def pricing_provenance(config):
    return {'pricing_config_hash': fingerprint(config), 'pricing_version': config['pricing_version'],
            'pricing_effective_date': config['effective_date'], 'pricing_source': config['source']}


def usage_cost(usage, pricing):
    """Legacy rate-key compatibility; input includes cached tokens, so subtract them once."""
    if not pricing or not all(pricing.get(k) for k in ['source', 'version', 'date']):
        return None
    rates = [pricing.get(k) for k in ['input_per_million_tokens', 'output_per_million_tokens',
                                     'cached_input_per_million_tokens']]
    if any(nonnegative_number(v) is None for v in rates):
        return None
    i, o = token_count(usage.get('input_tokens')), token_count(usage.get('output_tokens'))
    c = token_count(usage.get('cached_input_tokens', usage.get('cached_tokens')))
    if i is None or o is None:
        return None
    if c is None:
        # Only legacy callers with verified equal rates can ignore the cache split.
        return (i*rates[0]+o*rates[1])/1_000_000 if rates[0] == rates[2] else None
    if c > i:
        return None
    return ((i-c)*rates[0]+o*rates[1]+c*rates[2])/1_000_000


def openai_rates(config, model, tier):
    entry = config.get('openai', {}).get(model)
    if not entry or tier != entry.get('service_tier'):
        return None
    return {'input_per_million_tokens': entry['input_per_1m_tokens'],
            'cached_input_per_million_tokens': entry['cached_input_per_1m_tokens'],
            'output_per_million_tokens': entry['output_per_1m_tokens'],
            'source': config['source']['openai'], 'version': config['pricing_version'],
            'date': config['effective_date']}


def attempt_cost(strategy, usage, settings):
    """Price the returned model/tier, never silently use a requested alias's price."""
    config = settings['pricing_catalog']
    result = {**pricing_provenance(config), 'pricing_model_id': usage.get('returned_model'),
              'routing_cost_usd': None, 'cost_calculation_method': None}
    if strategy == 'jev':
        charge = nonnegative_number(usage.get('provider_reported_cost_usd'))
        result.update(routing_cost_usd=charge,
                      cost_status='PROVIDER_REPORTED' if charge is not None else 'PROVIDER_COST_UNAVAILABLE',
                      cost_calculation_method='OPENROUTER_USAGE_COST_USD')
        return result
    if not usage.get('returned_model'):
        result['cost_status'] = 'USAGE_NOT_RETURNED'
        return result
    rates = openai_rates(config, usage.get('returned_model'), usage.get('service_tier'))
    # Existing explicitly configured legacy pricing remains usable for its exact model.
    legacy = settings.get('pricing')
    if legacy and usage.get('returned_model') == settings.get('model') and usage.get('service_tier') == 'default':
        rates = legacy
        result.update(pricing_version=legacy.get('version'), pricing_effective_date=legacy.get('date'),
                      pricing_source=legacy.get('source'), pricing_config_hash=fingerprint(legacy))
    if rates is None:
        result['cost_status'] = 'PRICING_NOT_CONFIGURED'
        return result
    result['cost_calculation_method'] = 'UNCACHED_INPUT_PLUS_CACHED_INPUT_PLUS_OUTPUT_PER_1M'
    i, c, o = (token_count(usage.get(k)) for k in ['input_tokens', 'cached_input_tokens', 'output_tokens'])
    if i is None or c is None or o is None:
        result['cost_status'] = 'USAGE_NOT_RETURNED'
    elif c > i:
        result['cost_status'] = 'INVALID_PROVIDER_USAGE'
    else:
        result['routing_cost_usd'] = usage_cost(usage, rates)
        result['cost_status'] = ('CALCULATED_FROM_PROVIDER_USAGE' if result['routing_cost_usd'] is not None
                                 else 'PRICING_NOT_CONFIGURED')
    return result

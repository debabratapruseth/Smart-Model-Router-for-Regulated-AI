"""Live execution metrics against frozen synthetic labels; replay is explicitly separate."""
import json
import os
import pandas as pd
from evaluation.live_store import atomic_json, redact

NOT_EVALUATED = {'UNAVAILABLE', 'BASELINE_REPLAY_UNAVAILABLE', 'NOT_EXECUTED', 'INDETERMINATE', 'BUDGET_STOP'}
FAILURES = {'API_FAILURE', 'INVALID_DECISION', 'REFUSAL', 'TIMEOUT', 'RATE_LIMIT', 'AUTH_ERROR', 'TEMPORARY_SERVER_ERROR'}


def write_csv(path, frame):
    temporary = path.with_suffix('.csv.tmp')
    frame = frame.copy()
    for name in ['provider_usage', 'pricing_source']:
        if name in frame:
            frame[name] = frame[name].map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
    frame.to_csv(temporary, index=False)
    with temporary.open('rb') as handle: os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize_live(frame):
    rows = []
    for (kind, strategy), group in frame.groupby(['benchmark_type', 'strategy'], sort=False):
        evaluated = group[~group.status.isin(NOT_EVALUATED)]
        routed = group[group.status == 'ROUTED']
        failed = group[group.status.isin(FAILURES)]
        api = group[group.api_calls.fillna(0) > 0]
        costs = api.routing_cost_usd
        total_cost = float(costs.sum()) if costs.notna().all() else None
        def token_total(name):
            if api.empty:
                return 0  # No external API token consumption; not local compute/TCO.
            if name not in api or api[name].isna().any():
                return None
            return int(api[name].sum())
        latency = evaluated.routing_latency_ms.dropna()
        modes = sorted(set(group.execution_mode))
        rows.append({
            'benchmark_type': kind, 'strategy': strategy, 'execution_mode': ','.join(modes),
            'evidence_type': ','.join(sorted(set(group.evidence_type))), 'reference_evidence_type': 'SYNTHETIC',
            'research_eligible': bool(group.research_eligible.any()),
            'total_requests': len(group), 'evaluated_requests': len(evaluated), 'routed_requests': len(routed),
            'unavailable_or_pending_requests': len(group)-len(evaluated), 'failed_requests': len(failed),
            'failure_rate': len(failed)/len(group),
            'correct_no_route': int(((group.status == 'NO_ROUTE') & group.expected_no_route).sum()),
            'expected_no_route_count': int(group.expected_no_route.sum()),
            'preferred_reference_route_agreement_including_correct_abstentions': evaluated.preferred_correct.mean(),
            'acceptable_reference_route_agreement_including_correct_abstentions': evaluated.acceptable_correct.mean(),
            'preferred_reference_route_agreement_routed_only': routed.preferred_correct.mean(),
            'acceptable_reference_route_agreement_routed_only': routed.acceptable_correct.mean(),
            'agreement_denominator_including_correct_abstentions': len(evaluated),
            'average_routing_latency_ms': latency.mean(), 'p50_routing_latency_ms': latency.median(),
            'p95_routing_latency_ms': latency.quantile(.95) if len(latency) else None,
            'latency_measurement_source': ','.join(sorted(set(group.latency_measurement_source))),
            'api_calls': int(group.api_calls.fillna(0).sum()), 'retry_count': int(group.retry_count.fillna(0).sum()),
            'total_input_tokens': token_total('input_tokens'),
            'total_cached_input_tokens': token_total('cached_input_tokens'),
            'total_output_tokens': token_total('output_tokens'),
            'total_tokens': token_total('total_tokens'),
            'total_routing_cost_usd': total_cost,
            'cost_status': ','.join(sorted(set(group.cost_status.dropna()))) if 'cost_status' in group else 'LEGACY_NOT_RECORDED',
            'pricing_version': ','.join(sorted(set(group.pricing_version.dropna()))) if 'pricing_version' in group else None,
            'routing_cost_usd': total_cost, 'routing_cost_usd_per_request': total_cost/len(group) if total_cost is not None else None,
            'routing_cost_usd_per_api_decision': total_cost/len(api) if total_cost is not None and len(api) else None,
            'unknown_cost_decisions': int(costs.isna().sum()),
            'invalid_decision_rate': (group.status == 'INVALID_DECISION').mean(),
            'ineligible_selection_attempts': int(group.ineligible_selection_attempts.fillna(0).sum()),
            'policy_violations': int(group.policy_violation.sum()),
        })
    return pd.DataFrame(rows)


def write_live_reports(directory, rows, manifest, journal):
    frame = pd.DataFrame(redact(rows))
    summary = summarize_live(frame)
    if manifest.get('experiment_mode')=='stability':
        # Compatibility files remain available, but repetitions are observations, not independent requests.
        summary = summary.rename(columns={'total_requests':'total_observations',
            'evaluated_requests':'evaluated_observations','routed_requests':'routed_observations',
            'unavailable_or_pending_requests':'unavailable_or_pending_observations',
            'routing_cost_usd_per_request':'routing_cost_usd_per_observation'})
        summary['inference_unit']='REQUEST_CLUSTER_NOT_REPETITION'
        write_csv(directory/'stability_results.csv',frame)
    write_csv(directory / 'benchmark_results.csv', frame)
    write_csv(directory / 'benchmark_summary.csv', summary)
    write_csv(directory / 'api_errors.csv', frame[~frame.status.isin(['ROUTED', 'NO_ROUTE'])])
    attempts = journal.attempt_results
    known = [e['data']['routing_cost_usd'] for e in attempts if e['data']['routing_cost_usd'] is not None]
    unknown = len(journal.attempts)-len(known)
    manifest.update(api_call_count=len(journal.attempts),
        failed_call_count=sum(bool(e['data']['error_type']) for e in attempts),
        successful_api_response_count=sum(e['data'].get('api_success', False) for e in attempts),
        retry_count=sum(max(0, count-1) for count in _attempt_groups(journal).values()),
        routing_cost_usd=sum(known) if unknown == 0 else None,
        known_routing_cost_usd=sum(known), unknown_cost_attempts=unknown,
        completed_observation_count=len(journal.latest),
        execution_mode_per_strategy={s: sorted(set(g.execution_mode)) for s,g in frame.groupby('strategy')},
        evidence_type_per_strategy={s: sorted(set(g.evidence_type)) for s,g in frame.groupby('strategy')},
        research_eligible_per_strategy={s: bool(g.research_eligible.any()) for s,g in frame.groupby('strategy')},
        returned_model_versions_per_strategy={s: sorted(set(g.returned_model.dropna())) for s,g in frame.groupby('strategy')})
    for token in ['input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens', 'cached_input_tokens']:
        values = [e['data'].get(token) for e in attempts]
        manifest[token] = sum(values) if len(values)==len(journal.attempts) and all(v is not None for v in values) else None
    manifest['total_routing_cost_usd'] = manifest['routing_cost_usd']
    manifest['cost_status_per_strategy'] = {s: sorted(set(g.cost_status.dropna())) for s,g in frame.groupby('strategy')}
    write_csv(directory / 'routing_cost_summary.csv', pd.DataFrame([{
        'scope': 'ALL_ATTEMPTS_INCLUDING_FORCE_RERUN_HISTORY', 'api_calls': manifest['api_call_count'],
        'total_input_tokens': manifest['input_tokens'], 'total_cached_input_tokens': manifest['cached_input_tokens'],
        'total_output_tokens': manifest['output_tokens'], 'total_routing_cost_usd': manifest['routing_cost_usd'],
        'routing_cost_usd': manifest['routing_cost_usd'], 'known_routing_cost_usd': sum(known),
        'unknown_cost_attempts': unknown}]))
    atomic_json(directory / 'run_manifest.json', manifest)
    return summary


def _attempt_groups(journal):
    groups = {}
    for e in journal.attempts:
        key = (e['key'], e['revision']); groups[key] = groups.get(key, 0)+1
    return groups


def live_report_manifests():
    from app.core.config_loader import ROOT
    return sorted((ROOT / 'reports/live').glob('*/run_manifest.json'), reverse=True)

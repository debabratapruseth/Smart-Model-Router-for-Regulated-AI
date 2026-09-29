"""Read-only frozen datasets and verified baseline replay. Never generate or fit data."""
import csv
import json
import random
import math
from hashlib import sha256
from app.core.config_loader import ROOT, fingerprint, load_config
from app.models.request import RoutingRequest
from evaluation.research_design import dataset_hash

FROZEN_RUNS = {
    'IID_SYNTHETIC': ROOT / 'reports/iid_synthetic/20260928T153247-eb297b76',
    'TEMPLATE_HELD_OUT': ROOT / 'reports/template_held_out/20260928T153148-82538755',
}
PROTECTED_SOURCES = ['app/core/policies.py', 'app/core/features.py', 'app/core/scoring.py',
                     'app/models/routing_input.py', 'app/strategies/ml.py', 'scripts/generate_banking_benchmark.py']


class FrozenBaseline:
    def __init__(self, kind, router, directory=None):
        self.kind = kind
        self.directory = directory or FROZEN_RUNS[kind]
        self.manifest = json.loads((self.directory / 'run_manifest.json').read_text())
        self.rows = json.loads((self.directory / 'test_dataset.json').read_text())
        labels = json.loads((self.directory / 'test_reference_labels.json').read_text())
        self.labels = {r['request_id']: r for r in labels}
        self.hash = dataset_hash(self.rows, labels)
        if self.manifest['benchmark_type'] != kind or self.hash != self.manifest['test_dataset_hash']:
            raise ValueError('Frozen dataset/manifest mismatch')
        if len(self.labels) != len(self.rows) or len({r['request']['request_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Frozen dataset has ambiguous request IDs')
        for key, name in [('evaluation','evaluation'),('routing','routing'),('policies','policies'),('catalog','models')]:
            if fingerprint(load_config(name)) != self.manifest['config_hashes'][key]:
                raise ValueError(f'Frozen configuration changed: {name}')
        for name in PROTECTED_SOURCES:
            if sha256((ROOT / name).read_bytes()).hexdigest() != self.manifest['source_hashes'][name]:
                raise ValueError(f'Frozen implementation changed: {name}')
        self.prepared = {row['request']['request_id']: router.prepare(RoutingRequest.model_validate(row['request']))
                         for row in self.rows}
        self.results = {}
        with (self.directory / 'benchmark_results.csv').open() as handle:
            for result in csv.DictReader(handle):
                self.results.setdefault((result['request_id'], result['strategy']), []).append(result)

    def sample(self, size, seed):
        if not 2 <= size <= len(self.rows):
            raise ValueError('Pilot size must be between 2 and the frozen dataset size')
        rng = random.Random(seed)
        pools = {False: [], True: []}
        for row in self.rows:
            empty = not self.prepared[row['request']['request_id']].eligible
            pools[empty].append(row)
        n_empty = round(size * len(pools[True]) / len(self.rows))
        if pools[True] and pools[False]:
            n_empty = max(1, min(size-1, n_empty))
        n_empty = min(n_empty, len(pools[True]))
        quotas = {True: n_empty, False: size-n_empty}
        covered, selected = set(), []
        for empty in [False, True]:
            candidates = list(pools[empty]); rng.shuffle(candidates)
            def strata(row):
                value = self.prepared[row['request']['request_id']].canonical()
                return {('task', row['task_type']), ('complexity', value.features['complexity']),
                        ('classification', value.metadata.data_classification), ('cost', value.metadata.cost_preference),
                        ('quality', value.metadata.quality_requirement)}
            for _ in range(quotas[empty]):
                chosen = max(range(len(candidates)), key=lambda i: len(strata(candidates[i])-covered))
                row = candidates.pop(chosen)
                selected.append(row); covered.update(strata(row))
        return sorted(selected, key=lambda row: row['request']['request_id'])

    def replay(self, request_id, strategy):
        matches = self.results.get((request_id, strategy), [])
        prepared = self.prepared[request_id]
        ids = [m.model_id for m in prepared.eligible]
        canonical_hash = fingerprint(prepared.canonical().model_dump(mode='json'))
        base = {'execution_mode': 'BASELINE_REPLAY', 'evidence_type': 'SYNTHETIC',
                'decision_source': 'FROZEN_BASELINE', 'latency_measurement_source': 'FROZEN_BASELINE',
                'source_run_id': self.manifest['run_id'], 'source_dataset_hash': self.hash,
                'source_request_id': request_id, 'research_eligible': False,
                'status': 'BASELINE_REPLAY_UNAVAILABLE', 'selected_model': None, 'confidence': None,
                'reason_codes': '[]', 'routing_latency_ms': None, 'api_calls': 0, 'routing_cost_usd': 0.0}
        if len(matches) != 1:
            return base
        stored = matches[0]
        if stored['dataset_hash'] != self.hash or stored['benchmark_type'] != self.kind:
            return base
        if stored.get('canonical_input_hash') and stored['canonical_input_hash'] != canonical_hash:
            return base
        try:
            confidence = float(stored['confidence']) if stored.get('confidence') else None
            latency = float(stored['routing_latency_ms'])
            reasons = json.loads(stored.get('reason_codes') or '[]')
            if json.loads(stored['eligible_models']) != ids or not math.isfinite(latency) or latency < 0:
                return base
            if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
                return base
            if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
                return base
        except (ValueError, TypeError, KeyError):
            return base
        selected = stored['selected_model'] or None
        if not ((stored['status'] == 'ROUTED' and selected in ids) or
                (stored['status'] == 'NO_ROUTE' and not ids and selected is None)):
            return base
        # The same deterministic policy is checked again at the replay boundary.
        model = next((m for m in prepared.eligible if m.model_id == selected), None)
        if model and self.router_policy_failure(prepared, model):
            return base
        return {**base, 'strategy_version': stored.get('strategy_version', 'not_recorded'), 'status': stored['status'], 'selected_model': selected, 'research_eligible': True,
                'confidence': confidence, 'reason_codes': json.dumps(reasons), 'routing_latency_ms': latency}

    @staticmethod
    def router_policy_failure(prepared, model):
        from app.core.policies import PolicyEngine
        policy = PolicyEngine(load_config('policies'), load_config('routing'))
        return policy.evaluate_model(prepared.request, prepared.features, model)

"""Deterministic, immutable subsets of frozen benchmarks; no generation or training."""
import copy
import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from app.core.config_loader import ROOT, fingerprint, load_config
from evaluation.frozen_baseline import FrozenBaseline, FROZEN_RUNS
from evaluation.research_design import dataset_hash

KINDS = tuple(FROZEN_RUNS)
LOCAL_STRATEGIES = ('rules', 'weighted', 'ml')


def experiment_config(sampling_seed=None, stability_seed=None):
    cfg = load_config('final_experiment')
    if sampling_seed is not None:
        cfg['sampling_seed'] = sampling_seed
    if stability_seed is not None:
        cfg['stability_seed'] = stability_seed
    if cfg['primary_requests_per_benchmark'] != 200 or cfg['stability_requests_per_benchmark'] != 25:
        raise ValueError('Final design requires exactly 200 primary and 25 stability requests per benchmark')
    if cfg['repetitions'] != 5:
        raise ValueError('Final stability design requires five repetitions')
    if cfg['statistics']['bootstrap_iterations'] < 10000:
        raise ValueError('Final analysis requires at least 10,000 bootstrap iterations')
    if cfg['statistics']['confidence_level'] != .95 or cfg['statistics']['correction'] != 'holm-bonferroni':
        raise ValueError('Final design requires 95% intervals and Holm-Bonferroni correction')
    return cfg


def largest_remainder(counts, size):
    """Integer arithmetic; tied remainders break by the sorted stratum label."""
    total = sum(counts.values())
    if not 0 <= size <= total or total == 0:
        raise ValueError('Invalid proportional sample size')
    allocated = {k: size*n//total for k,n in counts.items()}
    order = sorted(counts, key=lambda k: (-(size*counts[k] % total), str(k)))
    for k in order[:size-sum(allocated.values())]:
        allocated[k] += 1
    return allocated


def stratified_sample(items, size, seed, fields, min_secondary=2):
    """Always allocate by routability then task; avoid fine, sparsely sampled cells.

    A secondary split is used only if each nonempty child receives at least
    min_secondary slots. Otherwise skip that field within this branch. At the
    last usable level, sample uniformly without replacement with a seeded RNG.
    """
    rng = random.Random(seed)
    audit = []
    def select(pool, n, depth, path):
        pool = sorted(pool, key=lambda r: r['request_id'])
        if not n:
            return []
        if depth == len(fields):
            return rng.sample(pool, n)
        field = fields[depth]
        groups = defaultdict(list)
        for row in pool:
            groups[json.dumps(row[field], sort_keys=True)].append(row)
        quotas = largest_remainder({k:len(v) for k,v in groups.items()}, n)
        if depth >= 2 and len(groups)>1 and min(quotas.values()) < min_secondary:
            return select(pool, n, depth+1, path)
        chosen = []
        for key in sorted(groups):
            child = path + [[field, json.loads(key)]]
            audit.append({'stratum':child, 'source_count':len(groups[key]), 'sample_count':quotas[key]})
            chosen.extend(select(groups[key], quotas[key], depth+1, child))
        return chosen
    if size > len(items) or size < 1:
        raise ValueError('Insufficient source rows for sample')
    return sorted(select(items, size, 0, []), key=lambda r:r['request_id']), audit


def item_for(base, row):
    rid = row['request']['request_id']
    canonical = base.prepared[rid].canonical()
    metadata = canonical.metadata.model_dump(mode='json')
    eligible = [m.model_id for m in canonical.eligible_models]
    expected = base.labels[rid]['expected_no_route']
    if bool(expected) != (not eligible):
        raise ValueError(f'Reference/policy routability mismatch: {base.kind}/{rid}')
    return {'benchmark_type':base.kind, 'request_id':rid, 'dataset_hash':base.hash,
            'canonical_input_hash':fingerprint(canonical.model_dump(mode='json')),
            'eligible_models':eligible, 'template_id':row['template_id'], 'task_type':row['task_type'],
            'expected_no_route':expected, 'data_classification':metadata['data_classification'],
            'modality':metadata['modality'], 'cost_preference':metadata['cost_preference'],
            'quality_requirement':metadata['quality_requirement'], 'complexity':canonical.features['complexity'],
            'reasoning_requirement':metadata['reasoning_requirement']}


def check_sources(bases, cfg):
    if set(bases) != set(KINDS):
        raise ValueError('Both frozen benchmarks are required')
    for kind, base in bases.items():
        expected = cfg['sources'][kind]
        if base.manifest['run_id'] != expected['source_run_id'] or base.hash != expected['dataset_hash']:
            raise ValueError(f'Pinned source dataset/run mismatch: {kind}')
        if sha256((base.directory/'benchmark_results.csv').read_bytes()).hexdigest() != expected['results_hash']:
            raise ValueError(f'Pinned baseline results changed: {kind}')
        labels = list(base.labels.values())
        if dataset_hash(base.rows, labels) != expected['dataset_hash']:
            raise ValueError(f'Frozen source/reference content changed: {kind}')
        if kind == 'TEMPLATE_HELD_OUT':
            test = set(base.manifest['test_template_ids'])
            if test & (set(base.manifest['training_template_ids']) | set(base.manifest['validation_template_ids'])):
                raise ValueError('Held-out template leakage')


def distributions(items, fields):
    return {field:dict(sorted(Counter(json.dumps(r[field],sort_keys=True) for r in items).items())) for field in fields}


def pilot_exclusions(cfg):
    """Exclude all selected pilot requests, independently of their outcomes.

    Plans include policy abstentions and unexecuted requests. Results are also
    included so an observed request cannot escape exclusion through a plan omission.
    Source hashes make the historical evidence auditable without altering it.
    """
    ids = {kind:set() for kind in KINDS}
    runs = {kind:set() for kind in KINDS}
    sources = {}
    for run_id in sorted(cfg['prior_live_pilot_run_ids']):
        directory = ROOT/'reports/live'/run_id
        manifest = json.loads((directory/'run_manifest.json').read_text())
        plan = json.loads((directory/'pilot_plan.json').read_text())
        if manifest['run_id'] != run_id or plan.get('full') is not False:
            raise ValueError('Exclusion source must be the specified historical pilot')
        with (directory/'benchmark_results.csv').open() as handle:
            observations = list(csv.DictReader(handle))
        for item in [*plan['items'], *observations]:
            kind = item['benchmark_type']
            if kind not in ids or item['dataset_hash'] != cfg['sources'][kind]['dataset_hash']:
                raise ValueError('Pilot exclusion source dataset mismatch')
            ids[kind].add(item['request_id'])
            runs[kind].add(run_id)
        sources[run_id] = {name:sha256((directory/name).read_bytes()).hexdigest()
            for name in ['run_manifest.json','pilot_plan.json','benchmark_results.csv']}
    exclusion = {'exclusion_reason':'PRIOR_LIVE_PILOT',
        'source_pilot_run_ids':sorted(sources), 'source_report_hashes':sources,
        'benchmarks':{kind:{'excluded_request_count':len(ids[kind]),
            'excluded_request_ids':sorted(ids[kind]), 'source_pilot_run_ids':sorted(runs[kind])}
            for kind in KINDS}}
    return {**exclusion, 'exclusion_set_hash':fingerprint(exclusion)}


def build_sample(bases, cfg):
    check_sources(bases, cfg)
    exclusions = pilot_exclusions(cfg)
    payload, summaries, stability = {}, {}, {}
    fields = cfg['stratification_fields']
    for kind, base in bases.items():
        excluded = set(exclusions['benchmarks'][kind]['excluded_request_ids'])
        originals_ids = {row['request']['request_id'] for row in base.rows}
        if not excluded <= originals_ids:
            raise ValueError('Pilot exclusion request absent from frozen source')
        # Filter the population BEFORE allocating any strata or consuming RNG draws.
        source = [item_for(base,row) for row in base.rows if row['request']['request_id'] not in excluded]
        chosen, strata = stratified_sample(source, 200, cfg['sampling_seed'], fields, cfg['minimum_secondary_allocation'])
        originals = {r['request']['request_id']:r for r in base.rows}
        rows = [copy.deepcopy(originals[i['request_id']]) for i in chosen]
        labels = [copy.deepcopy(base.labels[i['request_id']]) for i in chosen]
        payload[kind] = {'rows':rows, 'reference_labels':labels, 'items':chosen,
                         'sample_dataset_hash':dataset_hash(rows,labels)}
        routable = [i for i in chosen if not i['expected_no_route']]
        stable, _ = stratified_sample(routable, 25, cfg['stability_seed'], fields, cfg['minimum_secondary_allocation'])
        stability[kind] = [i['request_id'] for i in stable]
        summaries[kind] = {'sample_size':len(chosen), 'source_size':len(base.rows),
            'sampling_pool_size':len(source), 'prior_live_pilot_overlap':0,
            'sampled_request_ids':[i['request_id'] for i in chosen],
            'sample_dataset_hash':payload[kind]['sample_dataset_hash'],
            'source_dataset_hash':base.hash, 'source_run_id':base.manifest['run_id'],
            'expected_no_route_count':len(chosen)-len(routable), 'routable_count':len(routable),
            'distributions_before':distributions(source,fields), 'distributions_after':distributions(chosen,fields),
            'stratum_counts':strata, 'template_ids':sorted({i['template_id'] for i in chosen}),
            'source_test_template_ids':base.manifest['test_template_ids'],
            'source_training_template_ids':base.manifest['training_template_ids'],
            'source_validation_template_ids':base.manifest['validation_template_ids']}
    identity = {'configuration':cfg, 'primary':payload, 'stability_request_ids':stability,
                'exclusions':exclusions}
    sample_id = 'final-'+fingerprint(identity)[:16]
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    manifest = {'experiment_name':cfg['experiment_name'], 'experiment_version':cfg['version'],
                'sample_id':sample_id, 'sample_hash':fingerprint(identity),
                'created_at':datetime.now(timezone.utc).isoformat(), 'sampling_seed':cfg['sampling_seed'],
                'sampling_method':cfg['sampling_method'], 'stratification_fields':fields,
                'source_benchmarks':list(bases), 'benchmarks':summaries, 'exclusions':exclusions,
                'stability_seed':cfg['stability_seed'], 'stability_request_ids':stability,
                'repetitions':cfg['repetitions'], 'python_version':sys.version,
                'git_commit':git.stdout.strip() if git.returncode==0 else None}
    document = {**identity, 'manifest':manifest}
    validate_sample(document,bases,cfg)
    return document


def validate_sample(document, bases, cfg):
    check_sources(bases,cfg)
    identity = {k:document[k] for k in ['configuration','primary','stability_request_ids','exclusions']}
    if document['configuration'] != cfg or fingerprint(identity) != document['manifest']['sample_hash']:
        raise ValueError('Sample configuration/content hash mismatch')
    exclusions = pilot_exclusions(cfg)
    if document['exclusions'] != exclusions or document['manifest'].get('exclusions') != exclusions:
        raise ValueError('Prior live pilot exclusion provenance changed')
    all_ids = []
    for kind,base in bases.items():
        part = document['primary'][kind]
        rows, labels, items = part['rows'], part['reference_labels'], part['items']
        ids = [r['request']['request_id'] for r in rows]
        if len(ids)!=200 or len(set(ids))!=200 or len(labels)!=200 or len(items)!=200:
            raise ValueError('Primary sample must have exactly 200 unique requests per benchmark')
        excluded = set(exclusions['benchmarks'][kind]['excluded_request_ids'])
        if set(ids) & excluded:
            raise ValueError('Prior live pilot request leaked into final sample')
        all_ids.extend(ids)
        originals = {r['request']['request_id']:r for r in base.rows}
        for row,label,item in zip(rows,labels,items):
            rid = row['request']['request_id']
            if rid not in originals or row != originals[rid] or label != base.labels[rid]:
                raise ValueError('Sample request or reference differs from frozen source')
            if item != item_for(base,row):
                raise ValueError('Sample canonical input, candidates or template provenance changed')
            if row['template_id'] not in base.manifest['test_template_ids']:
                raise ValueError('Sample template provenance lost')
            for strategy in LOCAL_STRATEGIES:
                stored = base.results.get((rid,strategy),[])
                if (len(stored)!=1 or stored[0].get('run_id') != base.manifest['run_id'] or
                        stored[0].get('request_id')!=rid or stored[0].get('strategy')!=strategy):
                    raise ValueError('Frozen replay source run is missing or ambiguous')
                if base.replay(rid,strategy)['status'] == 'BASELINE_REPLAY_UNAVAILABLE':
                    raise ValueError(f'Frozen replay unavailable: {kind}/{rid}/{strategy}')
        if dataset_hash(rows,labels) != part['sample_dataset_hash']:
            raise ValueError('Sample dataset hash mismatch')
        summary = document['manifest']['benchmarks'][kind]
        expected_metadata = {'sampled_request_ids':ids, 'sample_dataset_hash':part['sample_dataset_hash'],
            'source_size':len(base.rows), 'sampling_pool_size':len(base.rows)-len(excluded),
            'prior_live_pilot_overlap':0,
            'source_dataset_hash':base.hash, 'source_run_id':base.manifest['run_id'],
            'source_test_template_ids':base.manifest['test_template_ids'],
            'source_training_template_ids':base.manifest['training_template_ids'],
            'source_validation_template_ids':base.manifest['validation_template_ids'],
            'template_ids':sorted({r['template_id'] for r in rows})}
        if any(summary.get(k)!=v for k,v in expected_metadata.items()):
            raise ValueError('Sample manifest lost source/template provenance')
        stable = document['stability_request_ids'][kind]
        allowed = {i['request_id'] for i in items if not i['expected_no_route']}
        if len(stable)!=25 or len(set(stable))!=25 or not set(stable)<=allowed:
            raise ValueError('Stability must contain 25 unique routable primary requests per benchmark')
    if len(set(all_ids))!=400:
        raise ValueError('Primary experiment must contain 400 globally unique request IDs')


def prepare_experiment(router, cfg, sample_root=None, baselines=None):
    bases = baselines or {k:FrozenBaseline(k,router) for k in KINDS}
    document = build_sample(bases,cfg)
    directory = (Path(sample_root) if sample_root else ROOT/'data/final_experiment')/document['manifest']['sample_id']
    path = directory/'sample_manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        validate_sample(saved,bases,cfg)
        if saved['manifest']['sample_hash'] != document['manifest']['sample_hash']:
            raise ValueError('Existing sample differs from deterministic selection')
        stable_metadata = lambda m:{k:v for k,v in m.items() if k not in ['created_at','python_version','git_commit']}
        if stable_metadata(saved['manifest']) != stable_metadata(document['manifest']):
            raise ValueError('Existing sample manifest metadata differs from validated selection')
        document = saved
    else:
        directory.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:  # Never overwrite any sample or historical dataset.
            json.dump(document,handle,indent=2,allow_nan=False)
            handle.write('\n')
    return bases,document,path


def experiment_items(document, mode):
    primary = [i for part in document['primary'].values() for i in part['items']]
    if mode == 'final':
        return primary
    if mode != 'stability':
        raise ValueError('Unknown experiment mode')
    stable = [i for i in primary if i['request_id'] in document['stability_request_ids'][i['benchmark_type']]]
    # Complete a full pass before the next repetition; never mix with primary observations.
    return [{**i,'repetition_number':repeat} for repeat in range(1,6) for i in stable]

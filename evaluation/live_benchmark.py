"""Explicit live pilot/full runner over frozen requests; no generation, fitting or downstream execution."""
import json
import os
import math
import subprocess
import tempfile
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
from app.core.config_loader import ROOT, load_config, fingerprint
from app.core.router import ModelRouter
from app.strategies.llm import OpenAILiveRouter, OPENAI_ROUTER_PROMPT_V1
from app.strategies.live_jev import JevLiveRouter, JEV_ROUTER_DECISION_V1
from app.strategies.live_base import live_settings, utc_now, usage_cost
from app.services.routing_costs import pricing_snapshot, pricing_provenance, openai_rates
from evaluation.frozen_baseline import FrozenBaseline, FROZEN_RUNS
from evaluation.live_store import LiveJournal, atomic_json, run_lock
from evaluation.live_reporting import write_live_reports

STRATEGIES = ['rules', 'weighted', 'ml', 'openai', 'jev']


def source_hashes():
    names = ['app/strategies/llm.py', 'app/strategies/live_jev.py', 'app/strategies/live_base.py',
             'app/models/live_routing.py', 'evaluation/live_benchmark.py', 'evaluation/frozen_baseline.py',
             'evaluation/live_store.py', 'evaluation/live_reporting.py', 'app/services/routing_costs.py']
    names += ['evaluation/experiment_sampling.py', 'evaluation/experiment_preflight.py',
              'evaluation/experiment_statistics.py', 'evaluation/experiment_stability.py']
    return {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def make_plan(router, strategies, full=False, kinds=None, baselines=None):
    cfg = load_config('live_routing')
    bases = baselines or {kind: FrozenBaseline(kind, router) for kind in (kinds or FROZEN_RUNS)}
    items = []
    for kind, base in bases.items():
        size = cfg['pilot']['iid_requests' if kind == 'IID_SYNTHETIC' else 'held_out_requests']
        selected = base.rows if full else base.sample(size, cfg['pilot']['seed'])
        for row in selected:
            rid = row['request']['request_id']
            canonical = base.prepared[rid].canonical()
            items.append({'benchmark_type': kind, 'request_id': rid, 'dataset_hash': base.hash,
                          'canonical_input_hash': fingerprint(canonical.model_dump(mode='json')),
                          'eligible_models': [m.model_id for m in canonical.eligible_models],
                          'template_id': row['template_id'], 'task_type': row['task_type'],
                          'complexity': canonical.features['complexity'],
                          'data_classification': canonical.metadata.data_classification.value,
                          'cost_preference': canonical.metadata.cost_preference,
                          'quality_requirement': canonical.metadata.quality_requirement})
    return bases, items


def base_row(item, strategy, run_id, target):
    return {**item, 'eligible_models': json.dumps(item['eligible_models']), 'run_id': run_id,
        'timestamp': utc_now(), 'strategy': strategy, 'strategy_version': 'live-routing-1.0',
        'input_mode': 'STRUCTURED_ONLY', 'reference_evidence_type': 'SYNTHETIC',
        'expected_no_route': target['expected_no_route'], 'preferred_correct': False, 'acceptable_correct': False,
        'policy_violation': False, 'ineligible_selection_attempts': 0, 'retry_count': 0,
        'selected_model': None, 'confidence': None, 'reason_codes': '[]', 'api_calls': 0,
        'routing_cost_usd': None, 'routing_latency_ms': None, 'returned_model': None,
        'cost_status': 'NO_EXTERNAL_CALL' if strategy in ['openai','jev'] else 'BASELINE_REPLAY_ZERO_EXTERNAL_COST',
        'input_tokens': None, 'cached_input_tokens': None, 'output_tokens': None, 'total_tokens': None,
        'pricing_version': None,
        'execution_provider': {'openai':'openai','jev':'openrouter'}.get(strategy,'local'),
        'execution_path': {'openai':'OPENAI_DIRECT','jev':'JEV_VIA_OPENROUTER'}.get(strategy,'FROZEN_BASELINE_REPLAY'),
        'strategy_display_name': {'openai':'OpenAI direct','jev':'Jev via OpenRouter'}.get(strategy,strategy),
        'execution_mode': 'LIVE' if strategy in ['openai','jev'] else 'BASELINE_REPLAY',
        'evidence_type': 'SYNTHETIC', 'research_eligible': False, 'status': 'NOT_EXECUTED',
        'decision_source': 'LIVE_PROVIDER' if strategy in ['openai','jev'] else 'FROZEN_BASELINE',
        'latency_measurement_source': 'CURRENT_RUN' if strategy in ['openai','jev'] else 'FROZEN_BASELINE'}


def apply_reference(row, target):
    abstained = row['status']=='NO_ROUTE' and target['expected_no_route']
    row['preferred_correct'] = bool(abstained or (row['selected_model'] is not None and row['selected_model']==target['preferred_model']))
    row['acceptable_correct'] = bool(abstained or row['selected_model'] in target['acceptable_models'])
    return row


def run_live(*, strategies=None, allow_live_api=False, max_live_api_calls=None, max_routing_spend_usd=None,
             full=False, resume=None, force_rerun=False, output_root=None, router=None, adapters=None, baselines=None,
             experiment_mode=None, experiment_cfg=None, sample_root=None):
    cfg = load_config('live_routing')
    max_live_api_calls = cfg['pilot']['max_live_api_calls'] if max_live_api_calls is None else max_live_api_calls
    if experiment_mode not in (None,'final','stability'):
        raise ValueError('Unknown experiment mode')
    strategies = strategies or (['openai','jev'] if experiment_mode=='stability' else STRATEGIES)
    if experiment_mode=='stability' and not set(strategies)<= {'openai','jev'}:
        raise ValueError('Stability repetitions are only for OpenAI and Jev')
    if len(set(strategies)) != len(strategies) or not set(strategies).issubset(STRATEGIES):
        raise ValueError('Unknown or duplicate strategy')
    live_names = [s for s in strategies if s in ['openai','jev']]
    if live_names and not allow_live_api:
        raise ValueError('External calls require explicit --allow-live-api')
    if (not isinstance(max_live_api_calls, int) or max_live_api_calls < 0 or
            (max_routing_spend_usd is not None and (not math.isfinite(max_routing_spend_usd) or max_routing_spend_usd <= 0))):
        raise ValueError('Invalid API-call/spend cap')
    if force_rerun and not resume:
        raise ValueError('--force-rerun requires --resume')
    settings = {s: live_settings(s, cfg) for s in live_names}
    pricing = pricing_snapshot()
    if 'jev' in settings:
        settings['jev']['mock_mode'] = os.getenv('JEV_MOCK_MODE', 'true').lower() == 'true'
    if max_routing_spend_usd is not None:
        for s in live_names:
            if s == 'jev':
                continue  # OpenRouter reports USD cost; the journal stops if any charge is unknown.
            rates = settings[s]['pricing'] or openai_rates(pricing, settings[s]['model'], 'default')
            if usage_cost({'input_tokens':1,'output_tokens':1,'cached_tokens':0}, rates) is None:
                raise ValueError('Spend cap requires documented pricing for every requested live model')
    with tempfile.TemporaryDirectory(prefix='router-live-') as runtime:
        router = router or ModelRouter(runtime_dir=Path(runtime))
        experiment = None
        if experiment_mode:
            from evaluation.experiment_sampling import experiment_config, prepare_experiment, experiment_items
            experiment_cfg = experiment_cfg or experiment_config()
            bases, document, sample_path = prepare_experiment(router,experiment_cfg,sample_root,baselines)
            items = experiment_items(document,experiment_mode)
            experiment = {'mode':experiment_mode, 'sample_manifest_path':str(sample_path),
                          'sample_manifest':document['manifest'], 'configuration':experiment_cfg}
        else:
            bases, items = make_plan(router, strategies, full=full, baselines=baselines)
        adapters = adapters or {s: (OpenAILiveRouter(settings[s]) if s=='openai' else JevLiveRouter(settings[s])) for s in live_names}
        if experiment_mode and any(not adapters[s].configured() for s in live_names):
            raise ValueError('Requested live provider unavailable: check enable flags, keys, model and Jev mock mode')
        identity = {'strategies': strategies, 'full': full, 'pilot_seed': cfg['pilot']['seed'], 'items': items,
                    'baseline_results_hashes':{k:sha256((b.directory/'benchmark_results.csv').read_bytes()).hexdigest() for k,b in bases.items()},
                    'runtime_versions':{'python':sys.version, **{p:version(p) for p in ['openai','typesafe-sdk','httpx','httpx2','pydantic']}},
                    'settings': settings, 'pricing_configuration': pricing,
                    'live_config_hash': fingerprint(cfg), 'source_hashes': source_hashes(),
                    'prompt_hashes': {'openai': fingerprint({'text':OPENAI_ROUTER_PROMPT_V1}),
                                      'jev': fingerprint({'text':JEV_ROUTER_DECISION_V1})}}
        if experiment:
            identity['experiment'] = experiment
        plan_hash = fingerprint(identity)
        if resume:
            directory = Path(resume)
            manifest = json.loads((directory/'run_manifest.json').read_text())
            if manifest.get('plan_hash') != plan_hash:
                raise ValueError('Resume refused: dataset, sample, model, settings, prompts or implementation changed')
            run_id = manifest['run_id']
        else:
            run_id = utc_now().replace(':','').replace('-','').split('.')[0]+'-'+uuid4().hex[:8]
            default_root = ROOT/'reports'/({'final':'final_experiment','stability':'stability'}.get(experiment_mode,'live'))
            directory = (Path(output_root) if output_root else default_root) / run_id
            manifest = {'run_id':run_id, 'timestamp':utc_now(), 'plan_hash':plan_hash,
                **pricing_provenance(pricing), 'pricing_configuration': pricing,
                'cost_scope': 'EXTERNAL_ROUTING_API_ONLY_EXCLUDES_DOWNSTREAM_AND_LOCAL_COMPUTE',
                'cost_calculation_method': {'openai':'UNCACHED_INPUT_PLUS_CACHED_INPUT_PLUS_OUTPUT_PER_1M',
                                            'jev':'OPENROUTER_USAGE_COST_USD'},
                'benchmark_type':list(bases), 'input_mode':'STRUCTURED_ONLY', 'reference_evidence_type':'SYNTHETIC',
                'description':'Empirical live routing execution evaluated against a synthetic reference-routing objective.',
                'research_eligible_definition':'Live provider rows require a successful valid external routing decision; replay rows are exact frozen synthetic decisions. Local policy abstentions are not empirical provider evidence.',
                'datasets':{k:{'dataset_hash':b.hash, 'source_run_id':b.manifest['run_id'],
                              'test_seed':b.manifest['test_seed'], 'training_seed':b.manifest['training_seed'],
                              'test_template_ids':b.manifest['test_template_ids'],
                              'config_hashes':b.manifest['config_hashes'],
                              'ml_model_fingerprint':b.manifest['model_fingerprint'],
                              'source_results_hash':sha256((b.directory/'benchmark_results.csv').read_bytes()).hexdigest()} for k,b in bases.items()},
                'live_config_hash':identity['live_config_hash'],
                'python_version':sys.version, 'packages':{p:version(p) for p in ['openai','typesafe-sdk','httpx','httpx2','pydantic']},
                'strategy_versions':{s:('FROZEN_BASELINE' if s not in live_names else adapters[s].prompt_version) for s in strategies},
                'router_models':{s:settings[s]['model'] for s in live_names},
                'prompt_versions':{s:adapters[s].prompt_version for s in live_names},
                'source_hashes':identity['source_hashes'], 'configuration':cfg, 'resolved_provider_settings':settings,
                'execution_paths':{'openai':'OPENAI_DIRECT','jev':'JEV_VIA_OPENROUTER'},
                'execution_providers':{'openai':'openai','jev':'openrouter'},
                'planned_request_counts':{k:len({i['request_id'] for i in items if i['benchmark_type']==k}) for k in bases},
                'planned_no_route_counts':{k:sum(i['benchmark_type']==k and not i['eligible_models'] for i in items) for k in bases}}
            if experiment:
                manifest.update(experiment_mode=experiment_mode, experiment=experiment,
                    statistical_configuration=experiment_cfg['statistics'],
                    planned_external_decisions={s:sum(bool(i['eligible_models']) for i in items) for s in live_names},
                    repetitions=5 if experiment_mode=='stability' else 1,
                    analysis_status='NOT_RUN_NO_COMPLETED_RESULTS')
            git = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
            manifest['git_commit'] = git.stdout.strip() if git.returncode==0 else None
        # The initial required decision count is checked before creating a run or making any calls.
        journal = LiveJournal(directory, max_live_api_calls, max_routing_spend_usd)
        def key(item, s):
            parts = [run_id,item['benchmark_type'],item['request_id'],s]
            if experiment_mode=='stability':
                parts.append(item['repetition_number'])
            return json.dumps(parts)
        needed = sum(bool(i['eligible_models']) for i in items for s in live_names
                     if force_rerun or (key(i,s) not in journal.latest and not journal.was_started(key(i,s))))
        if len(journal.attempts)+needed > max_live_api_calls:
            raise ValueError(f'API-call cap exceeded before execution: {len(journal.attempts)} spent/reserved + {needed} planned > {max_live_api_calls}')
        directory.mkdir(parents=True,exist_ok=bool(resume))
        with run_lock(directory):
            journal = LiveJournal(directory, max_live_api_calls, max_routing_spend_usd)
            manifest.update(status='RUNNING', max_live_api_calls=max_live_api_calls,
                            max_routing_spend_usd=max_routing_spend_usd, last_resume_timestamp=utc_now())
            atomic_json(directory/'run_manifest.json',manifest)
            atomic_json(directory/'pilot_plan.json',identity)
            rows, stopped = [], False
            for item in items:
                base = bases[item['benchmark_type']]
                target = base.labels[item['request_id']]
                prepared = base.prepared[item['request_id']]
                canonical = prepared.canonical()
                for s in strategies:
                    observation_key = key(item,s)
                    row = base_row(item,s,run_id,target)
                    if experiment_mode:
                        row['sample_dataset_hash']=document['primary'][item['benchmark_type']]['sample_dataset_hash']
                        row['experiment_sample_hash']=document['manifest']['sample_hash']
                    if not force_rerun and journal.pending(observation_key):
                        latest_revision = max(e['revision'] for e in journal.attempts if e['key']==observation_key)
                        row.update(status='INDETERMINATE',error_type='INTERRUPTED_ATTEMPT',evidence_type='EMPIRICAL',
                                   api_calls=sum(e['key']==observation_key and e['revision']==latest_revision for e in journal.attempts))
                        rows.append(row); continue
                    if not force_rerun and observation_key in journal.latest:
                        rows.append(journal.latest[observation_key]); continue
                    revision = journal.revision(observation_key)
                    if s not in live_names:
                        row.update(base.replay(item['request_id'],s))
                    elif stopped:
                        rows.append(row); continue
                    else:
                        def reserve(): journal.reserve(observation_key,revision)
                        def complete_attempt(data): journal.append({'event':'attempt_result','key':observation_key,'revision':revision,'data':data})
                        obs = adapters[s].execute(canonical.model_copy(deep=True), allow_live_api=allow_live_api,
                                                  before_attempt=reserve, after_attempt=complete_attempt)
                        row.update(obs.telemetry.model_dump())
                        row.update(status=obs.status,reason_codes=json.dumps(obs.telemetry.reason_codes),
                                   api_routing_latency_ms=obs.telemetry.routing_latency_ms,
                                   preparation_latency_ms=prepared.preparation_latency_ms)
                        if obs.status=='NO_ROUTE':
                            row['decision_source']='DETERMINISTIC_POLICY'
                            row['execution_provider']='local'
                            row['execution_path']='LOCAL_POLICY_NO_ROUTE'
                            row['routing_cost_usd']=0.0
                        # Include identical shared preparation cost for current measurements only.
                        row['routing_latency_ms'] += prepared.preparation_latency_ms
                        if obs.decision:
                            selected = next((m for m in prepared.eligible if m.model_id==obs.decision.selected_model),None)
                            if selected is None or router.policy.evaluate_model(prepared.request,prepared.features,selected):
                                row.update(status='INVALID_DECISION',selected_model=None,research_eligible=False,
                                           ineligible_selection_attempts=1,error_type='INELIGIBLE_MODEL_SELECTED')
                        if obs.status=='BUDGET_STOP':
                            stopped=True
                            if not obs.telemetry.api_calls:
                                rows.append(row); continue  # Safe to try this uncalled decision on resume.
                    row = apply_reference(row,target)
                    row['revision']=revision
                    journal.append({'event':'observation','key':observation_key,'revision':revision,'row':row})
                    rows.append(row)
                    # Persist every completed decision; journal remains authoritative on interruption.
                    manifest['status']='RUNNING'
                    write_live_reports(directory,rows,manifest,journal)
            manifest['status'] = 'BUDGET_STOP' if stopped else ('INCOMPLETE' if any(r['status']=='INDETERMINATE' for r in rows) else 'COMPLETED')
            manifest['end_timestamp']=utc_now()
            summary = write_live_reports(directory,rows,manifest,journal)
            if experiment_mode and manifest['status']=='COMPLETED':
                if experiment_mode=='final':
                    from evaluation.experiment_statistics import write_primary_analysis
                    write_primary_analysis(directory,experiment_cfg['statistics'])
                else:
                    from evaluation.experiment_stability import write_stability_analysis
                    write_stability_analysis(directory)
                manifest['analysis_status']='COMPLETED'
                atomic_json(directory/'run_manifest.json',manifest)
        return directory, summary

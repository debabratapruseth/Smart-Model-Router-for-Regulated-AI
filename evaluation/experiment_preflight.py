"""Read/validate frozen sources and prepare a plan; never initialize provider clients."""
import os
import math
import tempfile
from pathlib import Path
from app.core.config_loader import ROOT, fingerprint
from app.core.router import ModelRouter
from app.strategies.live_base import live_settings
from app.services.routing_costs import pricing_snapshot, pricing_provenance, openai_rates
from evaluation.experiment_sampling import experiment_config, prepare_experiment


def preflight(mode='final', strategies=None, sampling_seed=None, stability_seed=None,
              max_live_api_calls=None, max_routing_spend_usd=None, sample_root=None, router=None,
              baselines=None, resume=None, force_rerun=False):
    from evaluation.live_benchmark import source_hashes
    from evaluation.live_store import LiveJournal
    cfg = experiment_config(sampling_seed,stability_seed)
    strategies = strategies or (['openai','jev'] if mode=='stability' else ['rules','weighted','ml','openai','jev'])
    allowed = {'openai','jev'} if mode=='stability' else {'rules','weighted','ml','openai','jev'}
    if mode not in ('final','stability') or len(strategies)!=len(set(strategies)) or not set(strategies)<=allowed:
        raise ValueError('Invalid strategies or experiment mode')
    if max_live_api_calls is not None and (type(max_live_api_calls) is not int or max_live_api_calls<0):
        raise ValueError('Invalid API-call cap')
    if max_routing_spend_usd is not None and (not math.isfinite(max_routing_spend_usd) or max_routing_spend_usd<=0):
        raise ValueError('Invalid spend cap')
    if force_rerun and not resume:
        raise ValueError('force-rerun requires resume')
    with tempfile.TemporaryDirectory(prefix='router-preflight-') as temporary:
        router = router or ModelRouter(runtime_dir=Path(temporary))
        bases,document,path = prepare_experiment(router,cfg,sample_root,baselines)
    providers = {}
    pricing = pricing_snapshot()
    for name,key in [('openai','OPENAI_API_KEY'),('jev','OPENROUTER_API_KEY')]:
        settings = live_settings(name)
        mock = name=='jev' and os.getenv('JEV_MOCK_MODE','true').lower()!='false'
        present = bool(os.getenv(key))
        providers[name] = {'strategy':name,'selected':name in strategies,'key_present':present,'enabled':settings['enabled'],
            'model':settings['model'],'mock_mode':mock,
            'available':bool(present and settings['enabled'] and settings['model'] and not mock),
            'execution_provider':'openai' if name=='openai' else 'openrouter',
            'execution_path':'OPENAI_DIRECT' if name=='openai' else 'JEV_VIA_OPENROUTER',
            'pricing_available':bool(openai_rates(pricing,settings['model'],'default')) if name=='openai' else True,
            'cost_method':'VERSIONED_USAGE_RATES' if name=='openai' else 'PROVIDER_REPORTED_WHEN_RETURNED'}
    primary = document['manifest']['benchmarks']
    routable = sum(v['routable_count'] for v in primary.values())
    per_provider = routable if mode=='final' else 50*5
    total = per_provider*sum(s in strategies for s in ['openai','jev'])
    cap = max_live_api_calls if max_live_api_calls is not None else 50
    result = {'mode':'DRY_RUN_PREFLIGHT_ONLY','experiment_mode':mode,'external_api_calls_made':0,
        'sample_manifest_path':str(path),'sample_hash':document['manifest']['sample_hash'],
        'exclusions':document['manifest']['exclusions'],
        'prior_live_pilot_overlap':{k:v['prior_live_pilot_overlap'] for k,v in document['manifest']['benchmarks'].items()},
        'sampling_seed':cfg['sampling_seed'],'sampling_method':cfg['sampling_method'],
        'stratification_fields':cfg['stratification_fields'],
        'primary':{k:{n:v[n] for n in ['sample_size','expected_no_route_count','routable_count','source_dataset_hash','sample_dataset_hash']}
                   for k,v in primary.items()},
        'primary_planned_calls_per_provider':{'openai':routable,'jev_via_openrouter':routable},
        'primary_planned_calls_all_live_providers':2*routable,
        'stability':{'IID_SYNTHETIC':25,'TEMPLATE_HELD_OUT':25,'repetitions':5,
                     'planned_openai_calls':250,'planned_jev_via_openrouter_calls':250,'planned_total_calls':500},
        'selected_strategies':strategies, 'selected_planned_external_calls':total,
        'max_live_api_calls':cap,'call_cap_sufficient_for_new_run':cap>=total,
        'max_routing_spend_usd':max_routing_spend_usd,'estimated_maximum_spend_usd':None,
        'spend_estimate_note':'No reliable combined maximum: output usage, retries, served tier and native Jev charge are not known in advance.',
        'providers':providers,'source_hashes':source_hashes(),'source_validation':'PASSED',
        'sample_validation':'PASSED','baseline_replay_available':{s:400 for s in ['rules','weighted','ml']},
        'held_out_template_leakage':0,'statistics':cfg['statistics'],**pricing_provenance(pricing)}
    if resume:
        import json
        saved = json.loads((Path(resume)/'run_manifest.json').read_text())
        if saved.get('experiment_mode')!=mode or saved['experiment']['sample_manifest']['sample_hash']!=result['sample_hash']:
            raise ValueError('Resume experiment/sample mismatch')
        journal = LiveJournal(Path(resume),cap,max_routing_spend_usd)
        from evaluation.experiment_sampling import experiment_items
        def observation_key(item,strategy):
            key = [saved['run_id'],item['benchmark_type'],item['request_id'],strategy]
            if mode=='stability': key.append(item['repetition_number'])
            return json.dumps(key)
        remaining = sum(bool(i['eligible_models']) for i in experiment_items(document,mode)
            for s in strategies if s in ['openai','jev'] and (force_rerun or
                (observation_key(i,s) not in journal.latest and not journal.was_started(observation_key(i,s)))))
        result['resume']={'run_id':saved['run_id'],'reserved_attempts':len(journal.attempts),
                          'completed_observations':len(journal.latest),
                          'remaining_planned_external_calls':remaining,
                          'call_cap_sufficient':len(journal.attempts)+remaining<=cap,
                          'force_rerun':force_rerun,
                          'note':'Execution also checks the complete settings/source/pricing plan fingerprint before dispatch.'}
    return result

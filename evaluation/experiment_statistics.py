"""Request-paired analysis of synthetic reference-route agreement, never model quality."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from evaluation.live_reporting import NOT_EVALUATED, write_csv

AGREEMENT_METRICS = {'preferred_correct':'Preferred Reference-Route Agreement',
                     'acceptable_correct':'Acceptable Reference-Route Agreement'}


def bootstrap_ci(values, seed=6042, iterations=10000):
    values = np.asarray(values,dtype=float)
    if not len(values):
        return None,None
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0,iterations,1000):
        draws = rng.choice(values,size=(min(1000,iterations-start),len(values)),replace=True)
        means.extend(draws.mean(axis=1))
    return tuple(float(x) for x in np.quantile(means,[.025,.975]))


def mcnemar(a,b):
    a,b = np.asarray(a,dtype=bool),np.asarray(b,dtype=bool)
    if len(a)!=len(b):
        raise ValueError('McNemar requires paired observations')
    wins,losses = int((a & ~b).sum()),int((~a & b).sum())
    return {'a_correct_b_wrong':wins, 'a_wrong_b_correct':losses,
            'mcnemar_statistic':min(wins,losses),
            'raw_p_value':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0}


def holm_bonferroni(pvalues):
    p = np.asarray(pvalues,dtype=float)
    order = np.argsort(p,kind='stable')
    adjusted = np.empty(len(p))
    previous = 0.0
    for rank,index in enumerate(order):
        previous = max(previous,min(1.0,(len(p)-rank)*p[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def paired_wilcoxon(a,b):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if len(a)!=len(b):
        raise ValueError('Wilcoxon requires paired observations')
    if not len(a):
        return None,None
    if np.all(a==b):
        return 0.0,1.0
    result = wilcoxon(a-b,alternative='two-sided',zero_method='wilcox',method='auto')
    return float(result.statistic),float(result.pvalue)


def describe(values):
    x = pd.Series(values,dtype=float).dropna()
    return {'n':len(x), 'mean':x.mean(), 'median':x.median(), 'p50':x.median(),
            'p95':x.quantile(.95), 'standard_deviation':x.std(ddof=1),
            'iqr':x.quantile(.75)-x.quantile(.25),
            'coefficient_of_variation':x.std(ddof=1)/x.mean() if len(x)>1 and x.mean()>0 else None}


def confidence_intervals(frame, config):
    records = []
    for (kind,strategy), group in frame.groupby(['benchmark_type','strategy'],sort=True):
        for scope,valid in [('INCLUDING_CORRECT_ABSTENTIONS',group[~group.status.isin(NOT_EVALUATED)]),
                            ('ROUTED_ONLY',group[group.status=='ROUTED'])]:
            for field,label in AGREEMENT_METRICS.items():
                values = valid[field].astype(float).to_numpy()
                low,high = bootstrap_ci(values,config['bootstrap_seed'],config['bootstrap_iterations'])
                records.append({'benchmark_type':kind,'strategy':strategy,'metric':field,'metric_label':label,
                    'scope':scope,'n':len(values),'agreement':values.mean() if len(values) else None,
                    'ci_low':low,'ci_high':high,'confidence_level':.95,'method':'REQUEST_PERCENTILE_BOOTSTRAP',
                    'bootstrap_seed':config['bootstrap_seed'],'bootstrap_iterations':config['bootstrap_iterations'],
                    'unavailable_or_pending_requests':int(group.status.isin(NOT_EVALUATED).sum())})
    return pd.DataFrame(records)


def paired_comparisons(frame):
    records = []
    for kind, group in frame.groupby('benchmark_type',sort=True):
        evaluated = group[~group.status.isin(NOT_EVALUATED)]
        for a,b in combinations(sorted(group.strategy.unique()),2):
            pairs = evaluated[evaluated.strategy==a].merge(evaluated[evaluated.strategy==b],on='request_id',
                                                         suffixes=('_a','_b'),validate='one_to_one')
            for metric,label in AGREEMENT_METRICS.items():
                aa,bb = pairs[metric+'_a'].astype(bool),pairs[metric+'_b'].astype(bool)
                n = len(pairs)
                diff = 100*(aa.mean()-bb.mean()) if n else None
                records.append({'benchmark_type':kind,'router_a':a,'router_b':b,'metric':metric,'metric_label':label,
                    'n_paired_observations':n,'a_agreement':aa.mean() if n else None,'b_agreement':bb.mean() if n else None,
                    'difference_pp_a_minus_b':diff,'absolute_difference_pp':abs(diff) if n else None,
                    'test':'EXACT_MCNEMAR_BINOMIAL','statistic_definition':'min(discordant_counts)',
                    **(mcnemar(aa,bb) if n else {'mcnemar_statistic':None,'raw_p_value':None,
                                               'a_correct_b_wrong':0,'a_wrong_b_correct':0})})
    result = pd.DataFrame(records)
    if not result.empty:
        result['adjusted_p_value'] = np.nan
        result['correction_method'] = 'HOLM_BONFERRONI'
        result['correction_family'] = 'ALL_ROUTER_PAIRS_WITHIN_BENCHMARK_AND_METRIC'
        for _, group in result.groupby(['benchmark_type','metric']):
            valid = group.raw_p_value.dropna()
            result.loc[valid.index,'adjusted_p_value'] = holm_bonferroni(valid.values)
    return result


def generalization_deltas(intervals):
    iid = intervals[intervals.benchmark_type=='IID_SYNTHETIC']
    held = intervals[intervals.benchmark_type=='TEMPLATE_HELD_OUT']
    joined = iid.merge(held,on=['strategy','metric','scope'],suffixes=('_iid','_held'),validate='one_to_one')
    return pd.DataFrame([{'strategy':r.strategy,'metric':r.metric,'scope':r.scope,
        'iid_agreement':r.agreement_iid,'held_out_agreement':r.agreement_held,
        'held_out_minus_iid_pp':100*(r.agreement_held-r.agreement_iid),
        'interpretation':'DESCRIPTIVE_DISTINCT_POPULATIONS_NOT_A_PAIRED_TEST'} for r in joined.itertuples()])


def latency_analysis(frame):
    records = []
    valid = frame[frame.status=='ROUTED']
    for (kind,strategy,source),group in valid.groupby(['benchmark_type','strategy','latency_measurement_source']):
        records.append({'benchmark_type':kind,'strategy':strategy,'latency_measurement_source':source,
                        **describe(group.routing_latency_ms)})
    return pd.DataFrame(records)


def paired_live_analysis(frame):
    """Only contemporary, actually executed OpenAI/Jev requests; no replay-latency tests."""
    records = []
    live = frame[(frame.execution_mode=='LIVE') & (frame.latency_measurement_source=='CURRENT_RUN') & (frame.api_calls>0)]
    for kind,group in live.groupby('benchmark_type'):
        for metric in ['routing_latency_ms','routing_cost_usd']:
            valid = group[group.status=='ROUTED'] if metric=='routing_latency_ms' else group
            paired = valid[valid.strategy=='openai'].merge(valid[valid.strategy=='jev'],on='request_id',
                                                         suffixes=('_a','_b'),validate='one_to_one')
            paired = paired.dropna(subset=[metric+'_a',metric+'_b'])
            a,b = paired[metric+'_a'],paired[metric+'_b']
            statistic,pvalue = paired_wilcoxon(a,b)
            records.append({'benchmark_type':kind,'router_a':'openai','router_b':'jev','metric':metric,
                'n_paired_observations':len(paired),'mean_difference_a_minus_b':(a-b).mean(),
                'test':'WILCOXON_SIGNED_RANK','statistic':statistic,'raw_p_value':pvalue})
    result = pd.DataFrame(records)
    if not result.empty:
        result['adjusted_p_value'] = np.nan
        result['correction_method'] = 'HOLM_BONFERRONI_ACROSS_LIVE_COST_AND_LATENCY_WITHIN_BENCHMARK'
        for _,g in result.groupby('benchmark_type'):
            p = g.raw_p_value.dropna()
            result.loc[p.index,'adjusted_p_value'] = holm_bonferroni(p)
    return result


def cost_analysis(frame):
    records = []
    for (kind,strategy),group in frame.groupby(['benchmark_type','strategy']):
        api = group[group.api_calls>0]
        costs = api.routing_cost_usd
        total = costs.sum() if costs.notna().all() else None
        row = {'benchmark_type':kind,'strategy':strategy,'api_calls':int(group.api_calls.sum()),
               'api_decisions':len(api),'total_routing_api_cost_usd':total,
               'known_cost_lower_bound_usd':costs.sum(),'unknown_cost_decisions':int(costs.isna().sum()),
               'mean_cost_per_api_decision':total/len(api) if total is not None and len(api) else None,
               'median_cost_per_api_decision':costs.median() if costs.notna().all() and len(api) else None,
               'cost_per_request_including_no_route':total/len(group) if total is not None else None,
               'cost_scope':'EXTERNAL_API_ONLY_NOT_LOCAL_COMPUTE_TCO'}
        for field in ['input_tokens','cached_input_tokens','output_tokens','total_tokens']:
            name = field if field=='total_tokens' else 'total_'+field
            row[name] = api[field].sum() if field in api and api[field].notna().all() else (0 if api.empty else None)
        records.append(row)
    return pd.DataFrame(records)


def primary_analysis(frame, config):
    if frame.duplicated(['benchmark_type','request_id','strategy']).any():
        raise ValueError('Primary analysis cannot pool repeated observations')
    ci = confidence_intervals(frame,config)
    return {'confidence_intervals.csv':ci, 'statistical_comparisons.csv':paired_comparisons(frame),
            'generalization_deltas.csv':generalization_deltas(ci), 'latency_summary.csv':latency_analysis(frame),
            'paired_live_comparisons.csv':paired_live_analysis(frame), 'cost_summary.csv':cost_analysis(frame)}


def write_primary_analysis(directory, config):
    """Called only after execution has produced a completed final-experiment run."""
    manifest = json.loads((directory/'run_manifest.json').read_text())
    if manifest.get('experiment_mode')!='final' or manifest['status']!='COMPLETED':
        raise ValueError('Statistical analysis requires completed primary experiment results')
    frame = pd.read_csv(directory/'benchmark_results.csv')
    if frame.empty or frame.status.isin(['NOT_EXECUTED','INDETERMINATE','BUDGET_STOP']).any():
        raise ValueError('No complete primary observations to analyze')
    expected = manifest['experiment']['sample_manifest']['benchmarks']
    strategies = set(manifest['strategy_versions'])
    if set(frame.strategy)!=strategies or set(frame.benchmark_type)!=set(expected):
        raise ValueError('Primary results are missing a strategy or benchmark')
    for (kind,strategy),group in frame.groupby(['benchmark_type','strategy']):
        if len(group)!=200 or set(group.request_id)!=set(expected[kind]['sampled_request_ids']):
            raise ValueError('Primary results do not match the exact 200-request sample')
    for filename, table in primary_analysis(frame,config).items():
        write_csv(directory/filename,table)

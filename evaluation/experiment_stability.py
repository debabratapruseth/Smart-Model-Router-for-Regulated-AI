"""Repeated decisions are clustered by request, never treated as independent requests."""
import numpy as np
import pandas as pd
from evaluation.experiment_statistics import describe, cost_analysis, AGREEMENT_METRICS
from evaluation.live_reporting import FAILURES, NOT_EVALUATED, write_csv


def stability_analysis(frame):
    keys = ['benchmark_type','request_id','strategy','repetition_number']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate stability observation identity')
    consistency,summary,latencies,costs,repetitions = [],[],[],[],[]
    for (kind,strategy),group in frame.groupby(['benchmark_type','strategy']):
        request_rows = []
        for rid,request in group.groupby('request_id'):
            successes = request[request.status=='ROUTED']
            counts = successes.selected_model.value_counts()
            n = len(successes)
            possible_pairs = n*(n-1)//2
            equal_pairs = sum(int(c)*(int(c)-1)//2 for c in counts)
            row = {'benchmark_type':kind,'strategy':strategy,'request_id':rid,'successful_repetitions':n,
                'planned_repetitions':5,'observed_repetitions':len(request),
                'modal_selection_rate':counts.max()/n if n else None,
                'unique_selected_models':len(counts),
                'full_consistency':len(counts)==1 if n>=2 else None,
                'all_five_successful_and_consistent':n==5 and len(counts)==1,
                'agreeing_repetition_pairs':equal_pairs,'comparable_repetition_pairs':possible_pairs,
                'pairwise_repetition_agreement':equal_pairs/possible_pairs if possible_pairs else None}
            request_rows.append(row); consistency.append(row)
        request_frame = pd.DataFrame(request_rows)
        measurable = request_frame[request_frame.successful_repetitions>=2]
        pairs = request_frame.comparable_repetition_pairs.sum()
        summary.append({'benchmark_type':kind,'strategy':strategy,'requests':len(request_frame),
            'requests_with_at_least_two_successes':len(measurable),
            'requests_with_zero_successes':int((request_frame.successful_repetitions==0).sum()),
            'requests_with_one_success':int((request_frame.successful_repetitions==1).sum()),
            'exact_selection_consistency':measurable.modal_selection_rate.mean(),
            'full_consistency_rate':measurable.full_consistency.astype(float).mean(),
            'all_five_successful_and_consistent_rate':request_frame.all_five_successful_and_consistent.mean(),
            'pairwise_repetition_agreement':request_frame.agreeing_repetition_pairs.sum()/pairs if pairs else None,
            'comparable_repetition_pairs':pairs,
            'unique_models_mean':measurable.unique_selected_models.mean(),
            'unique_models_median':measurable.unique_selected_models.median(),
            'unique_models_max':measurable.unique_selected_models.max(),
            'failure_count':int(group.status.isin(FAILURES).sum()), 'retry_count':int(group.retry_count.sum()),
            'successful_repetitions':int((group.status=='ROUTED').sum()),
            'interpretation':'SUCCESS_CONDITIONAL_REQUEST_CLUSTERED_DESCRIPTIVE'})
        attempted = group[group.api_calls>0]
        latencies.append({'benchmark_type':kind,'strategy':strategy,**describe(attempted.routing_latency_ms)})
        cost = cost_analysis(group).iloc[0].to_dict()
        successes = int((group.status=='ROUTED').sum())
        total = cost['total_routing_api_cost_usd']
        cost['cost_per_successful_routing_decision'] = total/successes if successes and pd.notna(total) else None
        costs.append(cost)
        for repeat,part in group.groupby('repetition_number'):
            evaluated = part[~part.status.isin(NOT_EVALUATED)]
            row = {'benchmark_type':kind,'strategy':strategy,'repetition_number':repeat,
                   'evaluated_requests':len(evaluated),'failed_requests':int(part.status.isin(FAILURES).sum()),
                   'retry_count':int(part.retry_count.sum()),'invalid_decisions':int((part.status=='INVALID_DECISION').sum()),
                   'timeouts':int((part.status=='TIMEOUT').sum()),
                   'unavailable_or_pending':int(part.status.isin(NOT_EVALUATED).sum()),
                   'ineligible_selection_attempts':int(part.ineligible_selection_attempts.sum()),
                   'policy_violations':int(part.policy_violation.sum())}
            for status in sorted(FAILURES):
                row['errors_'+status.lower()] = int((part.status==status).sum())
            for field in AGREEMENT_METRICS:
                row[field+'_agreement'] = evaluated[field].mean()
            repetitions.append(row)
    repeat_frame = pd.DataFrame(repetitions)
    variability = []
    for (kind,strategy),group in repeat_frame.groupby(['benchmark_type','strategy']):
        for field,label in AGREEMENT_METRICS.items():
            variability.append({'benchmark_type':kind,'strategy':strategy,'metric':label,
                                **describe(group[field+'_agreement'])})
    return {'stability_summary.csv':pd.DataFrame(summary), 'selection_consistency.csv':pd.DataFrame(consistency),
            'latency_stability.csv':pd.DataFrame(latencies), 'cost_stability.csv':pd.DataFrame(costs),
            'repetition_summary.csv':repeat_frame, 'agreement_variability.csv':pd.DataFrame(variability)}


def write_stability_analysis(directory):
    frame = pd.read_csv(directory/'benchmark_results.csv')
    write_csv(directory/'stability_results.csv',frame)
    for filename,table in stability_analysis(frame).items():
        write_csv(directory/filename,table)

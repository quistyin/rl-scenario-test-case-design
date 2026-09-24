"""Post-execution descriptive audit; does not alter frozen primary metrics."""
from collections import Counter, defaultdict

import numpy as np

from risk_design import ROOT, ScenarioConfig, read, dump
from run_frozen import load_rows, check_frozen
from supplementary_audit import physical_signature
from magent2_experiment.coverage_model import ALLOWED_TRANSITIONS


def main():
    frozen = check_frozen()
    rows = load_rows()
    lookup = {(r['candidate_id'], r['model_seed'], r['environment_seed']): r for r in rows}
    metrics = read(ROOT / 'results/analysis/suite_metrics.json')
    out = {'status': 'post-execution descriptive, not additional confirmatory tests'}
    edges = {tuple(e) for r in rows for e in r['transition_edges']}
    out['whole_primary_execution'] = {
        'terminal_counts': dict(Counter(r['test_case']['terminal_tag'] for r in rows)),
        'mutual_elimination': sum(r['red_alive'] == r['blue_alive'] == 0 for r in rows),
        'edges_observed': len(edges), 'edges_defined': len(ALLOWED_TRANSITIONS),
        'missing_edges': sorted(ALLOWED_TRANSITIONS - edges),
    }
    risk = [s for s in frozen['suites'] if s['batch'] == 'main' and s['method'] == 'risk']
    miss, target_hits, suite_edges = Counter(), defaultdict(list), []
    examples, shape_groups = [], []
    for s in risk:
        groups = defaultdict(list)
        for i, c in enumerate(s['cases'], 1):
            sig = physical_signature(ScenarioConfig(**c['config']))
            groups[sig[:-2] + sig[-1:]].append({'position': i, 'cycles': c['config']['max_cycles'], 'id': c['id']})
        shape_groups.append({'design': s['design'], 'distinct_except_time_limit': len(groups),
                             'time_limit_families': [g for g in groups.values() if len(g) > 1]})
        for p in frozen['policies']:
            for seed in frozen['environment_seeds']:
                rr = [lookup[(cid, p, seed)] for cid in s['ids']]
                covered5 = set().union(*(set(r['covered_paths']) for r in rr[:5]))
                missing = sorted(set(('P1','P2','P3','P4','P5')) - covered5)
                miss.update(missing)
                e = {tuple(edge) for r in rr for edge in r['transition_edges']}
                suite_edges.append({'design': s['design'], 'model': p, 'seed': seed, 'count': len(e),
                                    'missing': sorted(ALLOWED_TRANSITIONS - e)})
                if missing and not any(x['design'] == s['design'] and x['model'] == p for x in examples):
                    examples.append({'design': s['design'], 'model': p, 'seed': seed, 'missing5': missing,
                        'first10': [{'position': i+1, 'history': c['history_id'], 'target': c['target'],
                                     'covered_paths': r['covered_paths'], 'terminal': r['test_case']['terminal_tag'],
                                     'trace': r['compact_trace']} for i, (c,r) in enumerate(zip(s['cases'][:10],rr[:10]))]})
                for c, r in zip(s['cases'], rr):
                    target_hits[c['target']].append(c['target'] in r['covered_paths'])
    out['risk_missing_after5'] = dict(miss)
    out['risk_missing5_examples'] = examples
    out['risk_assigned_target_hit'] = {k: {'hits': sum(v), 'total': len(v), 'rate': float(np.mean(v))} for k,v in target_hits.items()}
    out['risk_time_limit_families'] = shape_groups
    out['risk_transition_audit'] = {'min': min(x['count'] for x in suite_edges),
        'max': max(x['count'] for x in suite_edges), 'mean': float(np.mean([x['count'] for x in suite_edges])),
        'missing_union': sorted({tuple(e) for x in suite_edges for e in x['missing']})}
    rm = [m for m in metrics if m['batch'] == 'main' and m['method'] == 'risk']
    out['risk_tfc_distribution'] = dict(Counter(m['tfc'] for m in rm))
    out['risk_state_first_full_distribution'] = dict(Counter(next((i+1 for i,v in enumerate(m['sc']) if v==1), None) for m in rm))
    out['risk_seed_metric_variation'] = [{'design': d, 'model': p,
        'distinct_sc_kpc_curves': len({(tuple(m['sc']),tuple(m['kpc'])) for m in rm if m['design']==d and m['model']==p})}
        for d in range(5) for p in frozen['policies']]
    case_models = defaultdict(list)
    for cid in {cid for s in risk for cid in s['ids'][:5]}:
        for p in frozen['policies']:
            rr = [lookup[(cid,p,seed)] for seed in frozen['environment_seeds']]
            variants = len({(tuple(r['compact_trace']),tuple(r['covered_paths']),r['steps'],r['red_alive'],r['blue_alive']) for r in rr})
            case_models[rr[0]['test_case']['blue_controller']].append(variants)
    out['risk_first5_seed_outcome_variants'] = {k: {'config_model_pairs': len(v), 'constant_across20': sum(n==1 for n in v),
        'min_variants': min(v), 'max_variants': max(v)} for k,v in case_models.items()}
    summary = read(ROOT / 'results/analysis/summary.json')
    out['kpc_at_common_step_budget'] = {'budget': summary['common_step_budget'], **{m: v[-1] for m,v in summary['step_curves'].items()}}
    dump(ROOT / 'results/analysis/result_diagnostics.json', out)
    print({k:v for k,v in out.items() if k not in ('risk_missing5_examples','risk_time_limit_families')})


if __name__ == '__main__':
    main()

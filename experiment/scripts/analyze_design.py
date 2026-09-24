from __future__ import annotations

from collections import defaultdict
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

from risk_design import ROOT, POLICIES, dump, read, suite_metrics
from run_frozen import load_rows, check_frozen

METHODS = ('random', 'reward', 'combination', 'risk', 'no_history', 'state_only')
LABELS = {'random': '随机设计', 'reward': '历史奖励引导', 'combination': '参数组合设计',
          'risk': '风险场景方法', 'no_history': '不使用历史结果', 'state_only': '仅状态与原子场景'}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def holm(values):
    out = [0.0] * len(values)
    running = 0.0
    for rank, i in enumerate(np.argsort(values)):
        running = max(running, min(1, values[i] * (len(values)-rank)))
        out[i] = running
    return out


def paired_stats(differences):
    d = np.round(np.asarray(differences, dtype=float), 12)
    p = 1.0 if np.all(d == 0) else float(wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    rng = np.random.default_rng(202609209)
    bootstrap = rng.choice(d, (10000, len(d)), replace=True).mean(axis=1)
    return {'mean_difference': float(d.mean()), 'ci_low': float(np.quantile(bootstrap, .025)),
            'ci_high': float(np.quantile(bootstrap, .975)), 'p_raw': p, 'paired_seed_blocks': len(d)}


def coverage_at_steps(steps, kpc, budgets):
    return [0.0 if np.searchsorted(steps, b, side='right') == 0 else
            float(kpc[np.searchsorted(steps, b, side='right')-1]) for b in budgets]


def aggregate(rows):
    success = [r['tfc'] for r in rows if r['tfc'] is not None]
    result = {'combinations': len(rows), 'sc30': float(np.mean([r['sc'][-1] for r in rows])),
              'auc': float(np.mean([r['auc'] for r in rows])),
              'full_count': sum(r['complete'] for r in rows),
              'full_rate': float(np.mean([r['complete'] for r in rows])),
              'steps': float(np.mean([r['steps'] for r in rows])),
              'tfc_success_mean': float(np.mean(success)) if success else None,
              'tfc_censored': len(rows)-len(success),
              'atomic_mean': float(np.mean([r['atomic'][-1] for r in rows])),
              'atomic_complete': float(np.mean([r['atomic'][-1] == 1 for r in rows])),
              'wall_hit': float(np.mean([r['wall_hit'] for r in rows])),
              'boundary_hit': float(np.mean([r['boundary_hit'] for r in rows]))}
    for k in (1, 5, 10, 30):
        result[f'kpc{k}'] = float(np.mean([r['kpc'][k-1] for r in rows]))
        result[f'sc{k}'] = float(np.mean([r['sc'][k-1] for r in rows]))
        result[f'complete{k}'] = float(np.mean([r['kpc'][k-1] == 1 for r in rows]))
    for path in ('P1', 'P2', 'P3', 'P4', 'P5'):
        result[path] = float(np.mean([path in r['paths'] for r in rows]))
    return result


def deduplicated_suites(suites):
    used, output = set(), []
    for s in suites:
        sig = (s['batch'], s['method'], tuple(s['ids']))
        if sig not in used:
            used.add(sig)
            output.append(s)
    return output


def main():
    frozen = check_frozen()
    rows = load_rows()
    assert len(rows) == frozen['expected_unique_episodes']
    lookup = {(r['candidate_id'], r['model_seed'], r['environment_seed']): r for r in rows}
    assert len(lookup) == len(rows)
    out = ROOT / 'results/analysis'
    out.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    metrics, first, target_rows, missing = [], [], [], []
    bundles = []
    for suite in deduplicated_suites(frozen['suites']):
        batch, method, design = (suite[x] for x in ('batch', 'method', 'design'))
        for model in POLICIES:
            for seed in frozen['environment_seeds']:
                records = [lookup[(cid, model, seed)] for cid in suite['ids']]
                m = suite_metrics(records)
                m.update(batch=batch, method=method, design=design, model=model, seed=seed,
                         wall_hit=any(r['wall_engagement'] for r in records),
                         boundary_hit=any(r['boundary_engagement'] for r in records))
                grouped[(batch, method)].append(m)
                metrics.append(m)
                if batch == 'main':
                    bundles.append((m, records))
                if not m['complete']:
                    missing.append({'batch': batch, 'method': method, 'design': design, 'model': model, 'seed': seed,
                                    'missing_paths': [p for p in ('P1','P2','P3','P4','P5') if p not in m['paths']]})
                for i, (case, record) in enumerate(zip(suite['cases'], records), 1):
                    target = case['target']
                    key = 'covered_paths' if target.startswith('P') else 'covered_states' if target.startswith('M') else 'covered_scenarios'
                    target_rows.append({'batch': batch, 'method': method, 'design': design, 'model': model,
                                        'seed': seed, 'case': record['candidate_id'], 'position': i, 'target': target,
                                        'hit': int(target in record[key]), 'target_evaluable': target[0] in 'PMS',
                                        'paths': record['covered_paths'], 'scenes': record['covered_scenarios']})
                    if batch == 'main' and design == 0 and seed == frozen['environment_seeds'][0]:
                        first.append({'method': method, 'model': model, 'position': i, 'case': case['id'],
                                      'history_id': case['history_id'], 'target': target, 'actual_paths': record['covered_paths'],
                                      'target_hit': target in record[key], 'cumulative_kpc': m['kpc'][i-1],
                                      'cumulative_sc': m['sc'][i-1], 'steps': record['steps'],
                                      'trace': record['compact_trace'], 'config': record['test_case']})
    summaries = {f'{b}/{method}': aggregate(values) for (b, method), values in grouped.items()}
    per_model = []
    for (batch, method), values in grouped.items():
        for model in POLICIES:
            per_model.append({'batch': batch, 'method': method, 'model': model,
                              **aggregate([r for r in values if r['model'] == model])})
    comparisons = []
    for family, others in [('primary', ('random','reward','combination')), ('ablation', ('no_history','state_only'))]:
        local = []
        for other in others:
            for metric in ('kpc5','kpc10','kpc30','auc'):
                differences = []
                for seed in frozen['environment_seeds']:
                    def mean(method):
                        group = [r for r in grouped[('main', method)] if r['seed'] == seed]
                        return np.mean([r['auc'] if metric == 'auc' else r['kpc'][int(metric[3:])-1] for r in group])
                    differences.append(float(mean('risk') - mean(other)))
                local.append({'family': family, 'comparison': f'risk-{other}', 'metric': metric, **paired_stats(differences)})
        for row, adjusted in zip(local, holm([r['p_raw'] for r in local])):
            row['p_holm'] = adjusted
        comparisons.extend(local)
    # Same 100 fixed permutations for every suite, not 100 new experimental samples.
    rng = np.random.default_rng(202609210)
    orders = [rng.permutation(30) for _ in range(100)]
    reorder_values = defaultdict(list)
    for metric, records in bundles:
        means = []
        for order in orders:
            covered, area = set(), 0
            for index in order:
                covered.update(records[int(index)]['covered_paths'])
                area += len(covered)/5
            means.append(area/30)
        reorder_values[metric['method']].append(float(np.mean(means)))
    reordered = {method: float(np.mean(v)) for method, v in reorder_values.items()}
    main_rows = [m for m in metrics if m['batch'] == 'main']
    common_steps = min(m['steps'] for m in main_rows)
    step_budgets = np.linspace(0, common_steps, 101).tolist()
    step_curves = {method: np.mean([coverage_at_steps(m['cumulative_steps'], m['kpc'], step_budgets)
                                  for m in main_rows if m['method'] == method], axis=0).tolist() for method in METHODS}
    design_results = []
    for suite in deduplicated_suites(frozen['suites']):
        g = [m for m in metrics if m['batch'] == suite['batch'] and m['method'] == suite['method'] and m['design'] == suite['design']]
        design_results.append({'batch': suite['batch'], 'method': suite['method'], 'design': suite['design'], **aggregate(g)})
    dump(out / 'summary.json', {'summaries': summaries, 'comparisons': comparisons, 'reordered_auc': reordered,
                               'common_step_budget': common_steps, 'step_budgets': step_budgets, 'step_curves': step_curves,
                               'per_model': per_model, 'per_design': design_results})
    dump(out / 'suite_metrics.json', metrics)
    write_csv(out / 'comparisons.csv', comparisons)
    write_csv(out / 'per_model.csv', per_model)
    write_csv(out / 'per_design.csv', design_results)
    write_csv(out / 'first_single_pass.csv', first)
    write_csv(out / 'target_realization.csv', target_rows)
    write_csv(out / 'missing_paths.csv', missing)
    curve_rows = []
    for (batch, method), g in grouped.items():
        for k in range(30):
            curve_rows.append({'batch': batch, 'method': method, 'cases': k+1,
                               'sc': float(np.mean([m['sc'][k] for m in g])),
                               'kpc': float(np.mean([m['kpc'][k] for m in g])),
                               'complete': float(np.mean([m['kpc'][k] == 1 for m in g]))})
    write_csv(out / 'curves.csv', curve_rows)
    plots(grouped, summaries, step_budgets, step_curves, out)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def plots(grouped, summaries, budgets, step_curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ('#657786','#b86c22','#5266ac','#15805a','#b34f6f','#7f6895')))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in METHODS:
        g = grouped[('main', method)]
        for ax, key in zip(axes, ('sc', 'kpc')):
            ax.plot(range(1,31), np.mean([r[key] for r in g], axis=0), label=method, color=colors[method])
            ax.set(xlabel='Distinct cases, one execution each', ylabel=key.upper(), ylim=(-.02, 1.04))
            ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'coverage_curves.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [[summaries[f'main/{m}'][p] for p in ('P1','P2','P3','P4','P5')] for m in METHODS]
    im = ax.imshow(data, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(5), ('P1','P2','P3','P4','P5'))
    ax.set_yticks(range(6), METHODS)
    for i, values in enumerate(data):
        for j, val in enumerate(values):
            ax.text(j, i, f'{val:.1%}', ha='center', va='center', color='white' if val>.7 else 'black')
    fig.colorbar(im, ax=ax, label='Path coverage across single-pass combinations')
    fig.tight_layout()
    fig.savefig(out / 'path_coverage.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5))
    for method in METHODS:
        ax.plot(budgets, step_curves[method], label=method, color=colors[method])
    ax.set(xlabel='Cumulative environment steps (completed cases)', ylabel='Mean KPC', ylim=(-.02,1.04))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'equal_steps.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()

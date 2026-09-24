"""Reproduce paper summaries from public combination metrics, without simulation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('random', 'combination', 'reward', 'risk')
LABELS = {'random': 'Random', 'combination': 'Parameter combination',
          'reward': 'Historical reward-guided', 'risk': 'Proposed method'}
MODELS = (11, 22, 33)
DESIGNS = tuple(range(5))
SEEDS = tuple(range(172920000, 172920020))
METRICS = ('kpc5', 'kpc10', 'kpc30', 'auc')
FIELDS = {'method', 'design', 'model', 'seed', 'sc', 'kpc', 'auc', 'complete', 'tfc', 'paths'}


def validate_rows(rows):
    groups = {method: [] for method in METHODS}
    for row in rows:
        if set(row) != FIELDS or row['method'] not in groups:
            raise ValueError('Unexpected method or public data fields')
        for metric, denominator in (('sc', 9), ('kpc', 5)):
            curve = row[metric]
            if len(curve) != 30 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in curve):
                raise ValueError('Coverage must have 30 finite values in [0, 1]')
            if any(a > b + 1e-12 for a, b in zip(curve, curve[1:])):
                raise ValueError('Cumulative coverage cannot decrease')
            if any(not math.isclose(v * denominator, round(v * denominator), abs_tol=1e-10) for v in curve):
                raise ValueError('Coverage denominator mismatch')
        if not math.isclose(row['auc'], fmean(row['kpc']), abs_tol=1e-12):
            raise ValueError('AUC is not the discrete mean of cumulative KPC')
        tfc = next((i + 1 for i, value in enumerate(row['kpc']) if value == 1), None)
        if row['tfc'] != tfc or row['complete'] != int(tfc is not None):
            raise ValueError('TFC or completion flag mismatch')
        if len(set(row['paths'])) != len(row['paths']) or not set(row['paths']) <= {'P1', 'P2', 'P3', 'P4', 'P5'}:
            raise ValueError('Invalid final path set')
        if not math.isclose(len(row['paths']) / 5, row['kpc'][-1], abs_tol=1e-12):
            raise ValueError('Final path set and KPC disagree')
        groups[row['method']].append(row)
    expected = set(itertools.product(DESIGNS, MODELS, SEEDS))
    for method, group in groups.items():
        keys = [(r['design'], r['model'], r['seed']) for r in group]
        if len(keys) != 300 or len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f'{method}: missing or duplicate matched combination')
    return groups


def signed_rank_exact(differences):
    values = [float(value) for value in differences if value != 0]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Finite nonzero differences are required')
    ordered = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks_twice = [0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and abs(values[ordered[end]]) == abs(values[ordered[start]]):
            end += 1
        for index in ordered[start:end]:
            ranks_twice[index] = start + 1 + end
        start = end
    # Doubled midranks permit exact integer counts, including absolute-rank ties.
    counts = Counter({0: 1})
    for rank in ranks_twice:
        updated = counts.copy()
        for subtotal, count in counts.items():
            updated[subtotal + rank] += count
        counts = updated
    positive = sum(rank for rank, value in zip(ranks_twice, values) if value > 0)
    negative = sum(ranks_twice) - positive
    permutations = 1 << len(values)
    lower = sum(count for subtotal, count in counts.items() if subtotal <= positive)
    upper = sum(count for subtotal, count in counts.items() if subtotal >= positive)
    return {'statistic': min(positive, negative) / 2,
            'positive_rank_sum': positive / 2, 'negative_rank_sum': negative / 2,
            'nonzero_pairs': len(values), 'positive_pairs': sum(v > 0 for v in values),
            'negative_pairs': sum(v < 0 for v in values),
            'zero_pairs': len(differences) - len(values),
            'absolute_difference_ties': len(values) - len(set(map(abs, values))),
            'sign_assignments': permutations,
            'p_raw': min(1.0, 2 * min(lower, upper) / permutations)}


def holm(p_values):
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def metric_value(row, metric):
    return row['auc'] if metric == 'auc' else row['kpc'][int(metric[3:]) - 1]


def statistics_report(groups):
    blocks = {method: {metric: [fmean(metric_value(row, metric) for row in group if row['seed'] == seed)
                               for seed in SEEDS] for metric in METRICS}
              for method, group in groups.items()}
    comparisons = []
    for baseline in METHODS[:-1]:
        for metric in METRICS:
            left, right = blocks['risk'][metric], blocks[baseline][metric]
            differences = [round(a - b, 12) for a, b in zip(left, right)]
            comparisons.append({'comparison': f'risk-{baseline}', 'metric': metric,
                                'paired_seed_blocks': 20, 'risk_mean': fmean(left),
                                'baseline_mean': fmean(right), 'mean_difference': fmean(differences),
                                'block_differences': differences, **signed_rank_exact(differences)})
    for row, value in zip(comparisons, holm([r['p_raw'] for r in comparisons])):
        row['p_holm'] = value
    return {'protocol': {'family_size': 12, 'seed_blocks': 20, 'combinations_per_block': 15,
                         'aggregation': 'Equal mean of five designs and three fixed models per seed',
                         'test': 'Two-sided exact sign-permutation distribution of Wilcoxon midranks',
                         'zero_method': 'wilcox', 'difference_rounding_decimals': 12,
                         'adjustment': 'One Holm family for three baselines times four metrics',
                         'reward_baseline': 'Geometry-corrected historical reward-guided design'},
            'block_means': blocks, 'comparisons': comparisons}


def table_rows(groups):
    result = []
    for method in METHODS:
        group = groups[method]
        row = {'method': method}
        for metric in ('sc', 'kpc'):
            for budget in (5, 10, 30):
                row[f'{metric.upper()}@{budget}/%'] = 100 * fmean(r[metric][budget - 1] for r in group)
        row['KPC-AUC'] = fmean(r['auc'] for r in group)
        row['FCR@30/%'] = 100 * fmean(r['complete'] for r in group)
        result.append(row)
    return result


def compare_reference(table, stats):
    reference = json.loads((ROOT / 'data/table5_reference.json').read_text(encoding='utf-8'))
    for actual, expected in zip(table, reference):
        if actual['method'] != expected['method']:
            raise ValueError('Reference method order mismatch')
        for name, value in expected.items():
            if name != 'method':
                digits = 4 if name == 'KPC-AUC' else 2
                if round(actual[name], digits) != value:
                    raise ValueError(f'Table 5 mismatch: {actual["method"]} {name}')
    old = json.loads((ROOT / 'data/reference_statistics.json').read_text(encoding='utf-8'))
    for actual, expected in zip(stats['comparisons'], old['comparisons']):
        for key in ('comparison', 'metric', 'statistic', 'p_raw', 'p_holm', 'positive_pairs', 'negative_pairs', 'zero_pairs'):
            if actual[key] != expected[key]:
                raise ValueError(f'Statistical reference mismatch: {key}')
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
               for a, b in zip(actual['block_differences'], expected['block_differences'])):
            raise ValueError('Seed-block differences mismatch')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'random': '#527ca5', 'combination': '#d08b28', 'reward': '#8c659e', 'risk': '#247e61'}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout='constrained')
    for axis, metric, label in zip(axes, ('sc', 'kpc'), ('State coverage (%)', 'Key-path coverage (%)')):
        for method in METHODS:
            rows = [r for r in curves if r['method'] == method]
            axis.plot([r['budget'] for r in rows], [100 * r[metric] for r in rows],
                      label=LABELS[method], color=colors[method], linewidth=2)
        axis.set(xlabel='Number of executed test cases', ylabel=label, xlim=(1, 30), ylim=(0, 103))
        axis.set_xticks([1, 5, 10, 15, 20, 25, 30])
        axis.grid(alpha=.2)
    axes[1].legend(loc='lower right', fontsize=8)
    fig.savefig(output / 'coverage_curves.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    rows = json.loads((ROOT / 'data/combination_metrics.json').read_text(encoding='utf-8'))
    groups = validate_rows(rows)
    table = table_rows(groups)
    stats = statistics_report(groups)
    compare_reference(table, stats)
    curves = [{'method': method, 'budget': k, 'sc': fmean(r['sc'][k - 1] for r in group),
               'kpc': fmean(r['kpc'][k - 1] for r in group),
               'full_path_completion_rate': fmean(r['kpc'][k - 1] == 1 for r in group)}
              for method, group in groups.items() for k in range(1, 31)]
    extra = {method: {'KPC@1': fmean(r['kpc'][0] for r in group),
                      'FCR@5': fmean(r['kpc'][4] == 1 for r in group),
                      'TFC_distribution': dict(Counter(str(r['tfc']) if r['tfc'] is not None else '>30' for r in group)),
                      'TFC_completed_mean': fmean(r['tfc'] for r in group if r['tfc'] is not None)
                      if any(r['tfc'] is not None for r in group) else None,
                      'path_coverage_at_30': {p: fmean(p in r['paths'] for r in group) for p in ('P1', 'P2', 'P3', 'P4', 'P5')}}
             for method, group in groups.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'table5.csv', table)
    write_csv(args.output / 'coverage_curves.csv', curves)
    for filename, value in [('statistics.json', stats), ('supplementary_summary.json', extra)]:
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not args.no_plots:
        plot_curves(curves, args.output)
    print(json.dumps({'combinations': len(rows), 'methods': list(groups), 'reference_checks': 'passed',
                      'comparisons': len(stats['comparisons']), 'simulations_run': 0,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

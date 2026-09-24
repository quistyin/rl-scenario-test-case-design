from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from risk_design import ROOT, ScenarioConfig, signature, semantic_neighbor, digest, dump, read, suite_metrics
from supplementary_audit import physical_signature
from run_frozen import load_rows, run_one, worker_init, check_frozen
from analyze_design import aggregate, paired_stats, holm
from verify_design import replay_paths

OUT = ROOT / 'results/reward_correction'


def correct_cases(suite):
    out = deepcopy(suite['cases'])
    rng = np.random.default_rng(suite['design_seed'] + 7000000)
    occupied = {signature(ScenarioConfig(**c['config'])) for c in out}
    physical = {physical_signature(ScenarioConfig(**c['config'])) for c in out}
    seen, changes = set(), []
    for i, item in enumerate(out):
        sig = physical_signature(ScenarioConfig(**item['config']))
        if sig in seen:
            assert i % 2 == 1, 'Unexpected duplicate parent; requires new design review'
            parent = ScenarioConfig(**out[i-1]['config'])
            for _ in range(1000):
                candidate = semantic_neighbor(parent, rng, occupied)
                if candidate is None:
                    raise RuntimeError('No physical neighbor')
                occupied.add(signature(candidate))
                sig = physical_signature(candidate)
                if sig not in physical:
                    break
            else:
                raise RuntimeError('Physical neighbors exhausted')
            changes.append({'position': i+1, 'old_config': item['config'], 'new_config': candidate.to_dict()})
            item['config'] = candidate.to_dict()
            item['evidence'] = 'static_geometry_corrected_reward_neighbor'
            physical.add(sig)
        seen.add(sig)
    return out, changes


def freeze():
    if (OUT / 'frozen.json').exists():
        raise RuntimeError('Correction already frozen')
    f = check_frozen()
    reward = [s for s in f['suites'] if s['batch'] == 'main' and s['method'] == 'reward']
    reward_ids = {cid for s in reward for cid in s['ids']}
    completed_ids = {r['candidate_id'] for r in load_rows()}
    assert not reward_ids & completed_ids, 'Correction must register before any reward evaluation'
    catalog = read(ROOT / 'results/catalog.json')
    by_sig = {signature(ScenarioConfig(**c['config'])): c['id'] for c in catalog}
    new, suites, changes = [], [], []
    for original in reward:
        cases, diff = correct_cases(original)
        for c in cases:
            sig = signature(ScenarioConfig(**c['config']))
            if sig not in by_sig:
                cid = f'RC{len(new)+1:04d}'
                by_sig[sig] = cid
                new.append({'id': cid, 'config': {**c['config'], 'case_id': cid}})
            c['id'] = by_sig[sig]
        suites.append({**original, 'method': 'reward_corrected', 'cases': cases, 'ids': [c['id'] for c in cases]})
        changes.extend({'design': original['design'], **x} for x in diff)
    payload = {'frozen_utc': datetime.now(timezone.utc).isoformat(),
               'reason': 'Geometry-only duplicate audit before any reward evaluation; original primary results retained.',
               'completed_reward_configs_at_freeze': 0, 'suites': suites, 'new_candidates': new,
               'changes': changes, 'extra_episodes': len(new)*60,
               'policies': f['policies'], 'environment_seeds': f['environment_seeds'],
               'hashes': {str(Path(__file__).relative_to(ROOT)): digest(__file__),
                          'scripts/supplementary_audit.py': digest(ROOT / 'scripts/supplementary_audit.py')}}
    dump(OUT / 'frozen.json', payload)
    print('CORRECTION_FROZEN', len(changes), 'replacement slots', len(new), 'new configs', payload['extra_episodes'], 'episodes')


def execute():
    f = read(OUT / 'frozen.json')
    for p, sha in f['hashes'].items():
        assert digest(ROOT / p) == sha
    path = OUT / 'episodes.jsonl'
    old = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {(r['candidate_id'],r['model_seed'],r['environment_seed']) for r in old}
    tasks = [(c,p,s) for c in f['new_candidates'] for p in f['policies'] for s in f['environment_seeds']
             if (c['id'],p,s) not in done]
    with ProcessPoolExecutor(max_workers=8, initializer=worker_init) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        with path.open('a', encoding='utf-8', buffering=1) as handle:
            for i, future in enumerate(as_completed(futures), 1):
                handle.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                if i % 60 == 0:
                    print('CORRECTION', i, '/', len(tasks), flush=True)


def analyze():
    f = read(OUT / 'frozen.json')
    extra = [json.loads(x) for x in (OUT / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(c['id'],p,s) for c in f['new_candidates'] for p in f['policies'] for s in f['environment_seeds']}
    assert len(extra) == len(expected) and {(r['candidate_id'],r['model_seed'],r['environment_seed']) for r in extra} == expected
    for row in extra:
        assert digest(ROOT / row['trace_file']) == row['trace_sha256']
        with gzip.open(ROOT / row['trace_file'], 'rt', encoding='utf-8') as handle:
            raw = json.load(handle)
        assert replay_paths(raw['state_trace'], raw['test_case']['terminal_tag']) == set(row['covered_paths'])
        assert set().union(*map(set, raw['state_flags'])) == set(row['covered_states'])
    all_rows = load_rows() + extra
    lookup = {(r['candidate_id'],r['model_seed'],r['environment_seed']): r for r in all_rows}
    results = []
    for suite in f['suites']:
        for p in f['policies']:
            for seed in f['environment_seeds']:
                records = [lookup[(cid,p,seed)] for cid in suite['ids']]
                m = suite_metrics(records)
                m.update(model=p, seed=seed, design=suite['design'], wall_hit=any(r['wall_engagement'] for r in records),
                         boundary_hit=any(r['boundary_engagement'] for r in records))
                results.append(m)
    primary = read(ROOT / 'results/analysis/suite_metrics.json')
    risk = [r for r in primary if r['batch'] == 'main' and r['method'] == 'risk']
    comparisons = []
    for metric in ('kpc5','kpc10','kpc30','auc'):
        diffs = []
        for seed in f['environment_seeds']:
            def value(rows):
                return np.mean([r['auc'] if metric == 'auc' else r['kpc'][int(metric[3:])-1] for r in rows if r['seed'] == seed])
            diffs.append(value(risk) - value(results))
        comparisons.append({'metric': metric, **paired_stats(diffs)})
    for r, adjusted in zip(comparisons, holm([r['p_raw'] for r in comparisons])):
        r['p_holm'] = adjusted
    dump(OUT / 'analysis.json', {'summary': aggregate(results), 'comparisons': comparisons,
                               'per_model': {str(p): aggregate([r for r in results if r['model'] == p]) for p in f['policies']},
                               'extra_episodes_verified': len(extra), 'extra_steps': sum(r['steps'] for r in extra),
                               'oracle_episodes': sum(bool(r['oracle_failures']) for r in extra)})
    dump(OUT / 'suite_metrics.json', results)
    print(json.dumps(aggregate(results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze','run','analyze'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': execute, 'analyze': analyze}[args.stage]()

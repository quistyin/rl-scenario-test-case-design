"""Audit recorded coverage, regenerate designs, or execute public frozen cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from statistics import fmean
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
sys.path[:0] = [str(ROOT / 'experiment'), str(ROOT / 'experiment/scripts')]
from magent2_experiment.coverage_model import ALLOWED_TRANSITIONS, KEY_PATHS, _VariantTracker
from reproduce import validate_rows, table_rows, statistics_report, compare_reference


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def safe_member(name):
    normalized = name.replace('\\', '/')
    p = PurePosixPath(normalized)
    if p.is_absolute() or '..' in p.parts or ':' in normalized:
        raise ValueError(f'Unsafe archive member: {name}')
    return p.as_posix()


def integrity():
    manifest = read(ROOT / 'data/release_manifest.json')['files']
    actual = {p.relative_to(ROOT).as_posix() for p in ART.rglob('*') if p.is_file()}
    if actual != set(manifest):
        raise ValueError('Artifact file set mismatch')
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT / safe_member(name)).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')


def replay_paths(trace, terminal):
    trackers = [_VariantTracker(pid, i, v) for pid, variants in KEY_PATHS.items() for i, v in enumerate(variants)]
    for i, state in enumerate(trace):
        legal = i == 0 or state == trace[i - 1] or (trace[i - 1], state) in ALLOWED_TRANSITIONS
        for tracker in trackers:
            tracker.observe(state, legal, terminal if i == len(trace) - 1 else None)
    return {t.path_id for t in trackers if t.covered}


def coverage_row(episodes, method, design, model, seed):
    states, paths = set(), set()
    sc, kpc = [], []
    for row in episodes:
        states.update(row['covered_states'])
        paths.update(row['covered_paths'])
        if not states <= {f'M{i}' for i in range(9)} or not paths <= set(KEY_PATHS):
            raise ValueError('Unknown state or path')
        sc.append(len(states) / 9)
        kpc.append(len(paths) / 5)
    tfc = next((i + 1 for i, x in enumerate(kpc) if x == 1), None)
    return {'method': method, 'design': design, 'model': model, 'seed': seed,
            'sc': sc, 'kpc': kpc, 'auc': fmean(kpc), 'complete': int(tfc is not None),
            'tfc': tfc, 'paths': sorted(paths)}


def recorded_rows():
    with gzip.open(ART / 'episodes.jsonl.gz', 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    index = {(r['candidate_id'], r['model_seed'], r['environment_seed']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate episode')
    return index


def combinations(index, plan):
    rows = []
    for suite in plan['suites']:
        for model in plan['policies']:
            for seed in plan['environment_seeds']:
                rows.append(coverage_row([index[(c, model, seed)] for c in suite['ids']],
                                        suite['method'], suite['design'], model, seed))
    return rows


def compare_combinations(rows):
    groups = validate_rows(rows)
    key = lambda r: (r['method'], r['design'], r['model'], r['seed'])
    reference = {key(r): r for r in read(ROOT / 'data/combination_metrics.json')}
    for row in rows:
        expected = reference[key(row)]
        for field in expected:
            if field == 'auc':
                if not math.isclose(row[field], expected[field], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f'AUC mismatch: {key(row)}')
            elif row[field] != expected[field]:
                raise ValueError(f'Combination mismatch: {key(row)} {field}')
    compare_reference(table_rows(groups), statistics_report(groups))


def verify_records(output):
    integrity()
    plan = read(ART / 'plan.json')
    index = recorded_rows()
    configs = {c['id']: c['config'] for c in plan['catalog']}
    expected = {(c, m, s) for c in configs for m in plan['policies'] for s in plan['environment_seeds']}
    if set(index) != expected:
        raise ValueError('Missing or extra episode in grid')
    archives = [zipfile.ZipFile(ART / p) for p in plan['trace_archives']]
    try:
        members = {}
        for archive in archives:
            for item in archive.infolist():
                name = safe_member(item.filename)
                if name in members:
                    raise ValueError('Duplicate trace member')
                members[name] = archive
        if set(members) != {safe_member(r['trace_file']) for r in index.values()}:
            raise ValueError('Trace inventory mismatch')
        for key, row in index.items():
            name = safe_member(row['trace_file'])
            compressed = members[name].read(name)
            if hashlib.sha256(compressed).hexdigest() != row['trace_sha256']:
                raise ValueError(f'Trace hash mismatch: {key}')
            raw = json.loads(gzip.decompress(compressed))
            for field, value in configs[key[0]].items():
                if field != 'environment_seed' and raw['test_case'][field] != value:
                    raise ValueError(f'Frozen configuration mismatch: {key} {field}')
            if raw['model_seed'] != key[1] or raw['test_case']['environment_seed'] != key[2]:
                raise ValueError('Model or seed mismatch')
            states = set().union(*map(set, raw['state_flags']))
            paths = replay_paths(raw['state_trace'], raw['test_case']['terminal_tag'])
            if states != set(row['covered_states']) or states != set(raw['covered_states']):
                raise ValueError(f'State recount mismatch: {key}')
            if paths != set(row['covered_paths']) or paths != set(raw['covered_paths']):
                raise ValueError(f'Path recount mismatch: {key}')
            # Aggregate recalculated sets, not the cached coverage fields.
            row['covered_states'], row['covered_paths'] = sorted(states), sorted(paths)
    finally:
        for archive in archives:
            archive.close()
    rows = combinations(index, plan)
    compare_combinations(rows)
    write(output / 'recounted_combinations.json', rows)
    report = {'recorded_episodes_checked': len(index), 'combinations': len(rows),
              'logical_case_references': 36000, 'coverage_and_statistics': 'passed',
              'simulations_run': 0, 'scope': 'Recorded labels and phase sequences, not all original observations.'}
    write(output / 'record_verification.json', report)
    print(json.dumps(report))


def verify_design(output):
    integrity()
    import risk_design as rd
    from reward_correction import correct_cases
    rd.ROOT = ART
    history = rd.History()
    count = 0
    for suite in read(ART / 'plan.json')['suites']:
        method, seed = suite['method'], suite['design_seed']
        if method == 'random':
            cases = rd.random_suite(seed)
        elif method == 'combination':
            cases, _ = rd.combination_suite(seed)
        elif method == 'risk':
            cases = rd.risk_suite(history, seed)
        else:
            cases, _ = correct_cases({'cases': rd.reward_suite(history, seed), 'design_seed': seed})
        got = [(rd.signature(rd.ScenarioConfig(**c['config'])), c['target']) for c in cases]
        expected = [(rd.signature(rd.ScenarioConfig(**c['config'])), c['target']) for c in suite['cases']]
        if got != expected:
            raise ValueError(f'Design mismatch: {method} {suite["design"]}')
        count += 1
        print('DESIGN', method, suite['design'], 'passed', flush=True)
    report = {'designs': count, 'history_episodes': len(history.rows), 'ordered_configurations': count * 30,
              'comparison': 'Effective configuration and target at every ordered position', 'simulations_run': 0}
    write(output / 'design_verification.json', report)


MODELS = {}


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def execute_one(task):
    import torch
    from magent2_experiment.policy import SharedActorCritic
    from magent2_experiment.path_runner import run_path_episode
    from magent2_experiment.scenario_config import ScenarioConfig
    item, model, seed = task
    if model not in MODELS:
        payload = torch.load(ART / f'models/seed_{model}.pt', map_location='cpu', weights_only=True)
        policy = SharedActorCritic((13, 13, 5), 21)
        policy.load_state_dict(payload['model'])
        policy.eval()
        MODELS[model] = policy
    result = run_path_episode(MODELS[model], ScenarioConfig(**{**item['config'], 'environment_seed': seed})).to_dict()
    result.update(candidate_id=item['id'], model_seed=model, environment_seed=seed)
    return result


def rerun(output, full, workers):
    integrity()
    plan = read(ART / 'plan.json')
    config = {c['id']: c for c in plan['catalog']}
    if full:
        ids, seeds = sorted(config), plan['environment_seeds']
    else:
        # Frozen choice: first ten cases of design zero for each method, all three models, first seed.
        ids = sorted({cid for s in plan['suites'] if s['design'] == 0 for cid in s['ids'][:10]})
        seeds = plan['environment_seeds'][:1]
    tasks = [(config[c], m, s) for c in ids for m in plan['policies'] for s in seeds]
    output.mkdir(parents=True, exist_ok=False)
    reference = recorded_rows()
    differences, index = [], {}
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
        with (output / 'executed.jsonl').open('w', encoding='utf-8') as handle:
            for i, row in enumerate(executor.map(execute_one, tasks), 1):
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                key = row['candidate_id'], row['model_seed'], row['environment_seed']
                index[key] = row
                for field in ('covered_states', 'covered_paths', 'steps', 'winner', 'compact_trace'):
                    actual = row[field] if field != 'compact_trace' else [s for j, s in enumerate(row['state_trace']) if j == 0 or s != row['state_trace'][j - 1]]
                    expected = reference[key][field]
                    if actual != expected:
                        differences.append({'episode': key, 'field': field, 'actual': actual, 'expected': expected})
                if i % 30 == 0 or i == len(tasks):
                    print('EXECUTED', i, '/', len(tasks), flush=True)
    report = {'mode': 'full' if full else 'smoke', 'simulations_run': len(tasks),
              'differences': differences, 'fresh_output': True,
              'environment': {'python': sys.version.split()[0], 'platform': sys.platform}}
    write(output / 'execution_verification.json', report)
    if full:
        rows = combinations(index, plan)
        write(output / 'recomputed_combinations.json', rows)
        compare_combinations(rows)
    print(json.dumps({**report, 'differences': len(differences)}))
    if differences:
        raise ValueError('Rerun differs from archive; inspect execution_verification.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify-records', 'verify-design', 'rerun'])
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    output = (args.output or ROOT / 'generated' / args.mode).resolve()
    if not output.is_relative_to((ROOT / 'generated').resolve()) or output == (ROOT / 'generated').resolve():
        parser.error('Output must be a subdirectory of this checkout generated directory')
    if args.mode == 'verify-records':
        verify_records(output)
    elif args.mode == 'verify-design':
        verify_design(output)
    else:
        rerun(output, args.full, args.workers)


if __name__ == '__main__':
    main()

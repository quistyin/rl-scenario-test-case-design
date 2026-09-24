from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import gzip
import json
import os
import time
import traceback

from risk_design import ROOT, ScenarioConfig, digest, dump, read, proximity

MODELS = {}


def pending_tasks(catalog, policies, seeds, completed):
    return [(item, p, s) for item in catalog for p in policies for s in seeds
            if (item['id'], p, s) not in completed]


def worker_init():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_one(task):
    from magent2_experiment.runner import load_policy
    from magent2_experiment import path_runner
    item, policy, seed = task
    if policy not in MODELS:
        MODELS[policy] = load_policy(ROOT / 'models' / f'seed_{policy}.pt')
    config = ScenarioConfig(**{**item['config'], 'environment_seed': seed})
    diagnostic = {'boundary_engagement': False, 'wall_engagement': False}
    original = path_runner._near_obstacle_or_boundary

    def audited(positions, walls, size):
        boundary, wall = proximity(positions, walls, size)
        diagnostic['boundary_engagement'] |= boundary
        diagnostic['wall_engagement'] |= wall
        return boundary or wall

    path_runner._near_obstacle_or_boundary = audited
    try:
        result = path_runner.run_path_episode(MODELS[policy], config).to_dict()
    finally:
        path_runner._near_obstacle_or_boundary = original
    result.update(candidate_id=item['id'], model_seed=policy, environment_seed=seed, **diagnostic)
    trace_path = ROOT / 'results/execution/traces' / f"{item['id']}_p{policy}_s{seed}.json.gz"
    temporary = trace_path.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(temporary, trace_path)
    transitions = result.pop('transitions')
    trace = result.pop('state_trace')
    result.pop('state_flags')
    result['compact_trace'] = [v for i, v in enumerate(trace) if i == 0 or v != trace[i-1]]
    result['transition_edges'] = sorted({(x['source'], x['target']) for x in transitions})
    result['illegal_edges'] = [x for x in transitions if not x['legal']]
    result['trace_file'] = str(trace_path.relative_to(ROOT))
    result['trace_sha256'] = digest(trace_path)
    return result


def load_rows():
    file = ROOT / 'results/execution/episodes.jsonl'
    return [json.loads(x) for x in file.read_text(encoding='utf-8').splitlines() if x] if file.exists() else []


def check_frozen():
    frozen = read(ROOT / 'results/frozen.json')
    for relative, sha in frozen['hashes'].items():
        if digest(ROOT / relative) != sha:
            raise RuntimeError(f'Frozen input changed: {relative}')
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    frozen = check_frozen()
    catalog = read(ROOT / 'results/catalog.json')
    output = ROOT / 'results/execution'
    (output / 'traces').mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    done = {(r['candidate_id'], r['model_seed'], r['environment_seed']) for r in rows}
    assert len(rows) == len(done)
    tasks = pending_tasks(catalog, frozen['policies'], frozen['environment_seeds'], done)
    log = (ROOT / 'logs/execution.log').open('a', encoding='utf-8', buffering=1)
    started = time.perf_counter()
    begin = datetime.now(timezone.utc).isoformat()

    def progress(n):
        value = {'completed': len(done)+n, 'expected': frozen['expected_unique_episodes'],
                 'invocation_completed': n, 'invocation_seconds': time.perf_counter()-started,
                 'updated_utc': datetime.now(timezone.utc).isoformat()}
        dump(output / 'progress.json', value)
        message = json.dumps(value)
        print(message, flush=True)
        log.write(message + '\n')

    progress(0)
    iterator = iter(tasks)
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as executor:
        pending = {}
        for _ in range(args.workers * 3):
            task = next(iterator, None)
            if task is not None:
                pending[executor.submit(run_one, task)] = task
        with (output / 'episodes.jsonl').open('a', encoding='utf-8', buffering=1) as handle:
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    task = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        error = {'candidate': task[0]['id'], 'policy': task[1], 'seed': task[2],
                                 'traceback': traceback.format_exc()}
                        dump(output / 'infrastructure_error.json', error)
                        raise
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                    n += 1
                    if n % 200 == 0 or n == len(tasks):
                        progress(n)
                    task = next(iterator, None)
                    if task is not None:
                        pending[executor.submit(run_one, task)] = task
    rows = load_rows()
    dump(output / 'summary.json', {'episodes': len(rows), 'sum_steps': sum(r['steps'] for r in rows),
                                  'sum_episode_seconds': sum(r['elapsed_seconds'] for r in rows),
                                  'last_invocation_wall_seconds': time.perf_counter()-started,
                                  'invocation_started_utc': begin, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                                  'oracle_episodes': sum(bool(r['oracle_failures']) for r in rows),
                                  'illegal_transition_episodes': sum(bool(r['illegal_edges']) for r in rows)})
    log.close()


if __name__ == '__main__':
    main()

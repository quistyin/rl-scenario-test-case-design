from __future__ import annotations

import gzip
import json

from risk_design import ROOT, ScenarioConfig, digest, dump, read, signature
from run_frozen import load_rows, check_frozen
from magent2_experiment.coverage_model import ALLOWED_TRANSITIONS, KEY_PATHS, _VariantTracker


def replay_paths(trace, terminal_tag):
    trackers = [_VariantTracker(pid, i, variant) for pid, variants in KEY_PATHS.items() for i, variant in enumerate(variants)]
    for i, state in enumerate(trace):
        legal = i == 0 or state == trace[i-1] or (trace[i-1], state) in ALLOWED_TRANSITIONS
        tag = terminal_tag if i == len(trace)-1 else None
        for tracker in trackers:
            tracker.observe(state, legal, tag)
    return {t.path_id for t in trackers if t.covered}


def main():
    frozen = check_frozen()
    catalog = {c['id']: c for c in read(ROOT / 'results/catalog.json')}
    rows = load_rows()
    expected = {(cid, p, s) for cid in catalog for p in frozen['policies'] for s in frozen['environment_seeds']}
    actual = [(r['candidate_id'], r['model_seed'], r['environment_seed']) for r in rows]
    errors = []
    if len(actual) != len(expected) or set(actual) != expected:
        errors.append('missing or duplicated execution grid')
    for suite in frozen['suites']:
        signatures = [signature(ScenarioConfig(**c['config'])) for c in suite['cases']]
        if len(set(signatures)) != 30:
            errors.append('duplicated effective case in suite')
        for cid, case in zip(suite['ids'], suite['cases']):
            if signature(ScenarioConfig(**case['config'])) != signature(ScenarioConfig(**catalog[cid]['config'])):
                errors.append('suite/catalog mismatch')
    for i, row in enumerate(rows, 1):
        path = ROOT / row['trace_file']
        if digest(path) != row['trace_sha256']:
            errors.append(f'hash mismatch {path.name}')
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            raw = json.load(f)
        if raw['test_case']['red_action_rule'] != 'argmax' or raw['test_case']['environment_seed'] != row['environment_seed']:
            errors.append(f'execution mode mismatch {path.name}')
        if signature(ScenarioConfig(**{k: raw['test_case'][k] for k in ScenarioConfig.__dataclass_fields__})) != signature(ScenarioConfig(**catalog[row['candidate_id']]['config'])):
            errors.append(f'case config mismatch {path.name}')
        states = set().union(*map(set, raw['state_flags']))
        if states != set(raw['covered_states']) or states != set(row['covered_states']):
            errors.append(f'SC recount {path.name}')
        paths = replay_paths(raw['state_trace'], raw['test_case']['terminal_tag'])
        if paths != set(row['covered_paths']) or paths != set(raw['covered_paths']):
            errors.append(f'KPC recount {path.name}')
        if bool(raw['boundary_engagement'] or raw['wall_engagement']) != ('S05' in raw['covered_scenarios']):
            errors.append(f'S05 diagnostic mismatch {path.name}')
        if i % 3000 == 0:
            print('VERIFIED', i, flush=True)
    result = {'passed': not errors, 'errors': errors, 'episodes_verified': len(rows),
              'checks': ['frozen source/history/model hashes', 'complete unique execution grid',
                         'effective parameter deduplication', 'each compressed trajectory hash',
                         'SC flag recount', 'KPC trace replay', 'S05 boundary/wall split consistency',
                         'fixed argmax and per-case seed/config identity']}
    dump(ROOT / 'results/final_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from magent2_experiment.scenario_config import ScenarioConfig
from magent2_experiment.scenario_geometry import build_team_positions, build_obstacles, validate_geometry

POLICIES = (11, 22, 33)
PRIORITY = ('P5', 'P3', 'P2', 'P1', 'P4')
DOMAINS = {
    'minimum_distance': (2, 4, 6, 7, 12, 16, 24, 32, 40, 48),
    'red_formation': ('compact', 'line', 'split'),
    'blue_formation': ('compact', 'line', 'split'),
    'relative_position': ('front', 'flank', 'near_red_group_1', 'encircle'),
    'obstacle_layout': ('open', 'single_gap', 'double_gap', 'corridor'),
    'engagement_region': ('center', 'top'),
    'blue_controller': ('random', 'rule_based', 'shared_model'),
    'blue_target_rule': ('nearest', 'lowest_health'),
    'phase_rule': ('constant', 'retreat_after_low_hp'),
    'max_cycles': (20, 40, 60, 80, 100, 120, 150, 180, 200, 300),
}
FIELDS = tuple(DOMAINS)
GEOMETRY = FIELDS[:6]
DESIGN_SEEDS = (202609201, 202609202, 202609203, 202609204, 202609205)
ENV_SEEDS = tuple(range(172920000, 172920020))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(c):
    updates = {'environment_seed': 0, 'red_action_rule': 'argmax'}
    if c.blue_controller != 'rule_based':
        updates.update(blue_target_rule='nearest', phase_rule='constant')
    if c.relative_position == 'encircle':
        updates['blue_formation'] = 'compact'
    return replace(c, **updates)


def signature(c):
    c = normalize(c)
    return tuple(getattr(c, f) for f in FIELDS)


@lru_cache(maxsize=None)
def geometry_legal(values):
    c = ScenarioConfig(**dict(zip(GEOMETRY, values)))
    try:
        red, blue = build_team_positions(c)
        validate_geometry(c, build_obstacles(c.obstacle_layout), red, blue)
        return True
    except ValueError:
        return False


def legal(c):
    return geometry_legal(tuple(getattr(c, f) for f in GEOMETRY))


def entry(c, target, source=None, evidence='unobserved_parameterization'):
    return {'config': normalize(c).to_dict(), 'target': target, 'history_id': source,
            'evidence': evidence}


class History:
    def __init__(self):
        self.catalog = read(ROOT / 'history/catalog.json')['candidates']
        self.rows = []
        self.lookup = defaultdict(list)
        for stage in ('screen', 'refine'):
            for line in (ROOT / f'history/{stage}.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                row['stage'] = stage
                self.rows.append(row)
                self.lookup[(row['candidate_id'], row['model_seed'], stage)].append(row)
        self.items = []
        used = set()
        for x in self.catalog:
            c = normalize(ScenarioConfig(**x['config']))
            if signature(c) not in used:
                self.items.append((x['id'], c))
                used.add(signature(c))

    def count(self, cid, model, target, stage):
        rows = self.lookup[(cid, model, stage)]
        key = 'covered_paths' if target.startswith('P') else 'covered_states' if target.startswith('M') else 'covered_scenarios'
        return sum(target in r[key] for r in rows), len(rows)

    def rates(self, cid, target):
        stage = 'refine' if self.lookup[(cid, 11, 'refine')] else 'screen'
        return np.array([self.count(cid, m, target, stage)[0] / self.count(cid, m, target, stage)[1]
                         for m in POLICIES]), stage

    def reward_rank(self):
        ids = [cid for cid, _ in self.items]
        ranks = []
        for m in POLICIES:
            vals = [np.mean([r['team_return'] for r in self.lookup[(cid, m, 'screen')]]) for cid in ids]
            ranks.append(rankdata(vals) / len(ids))
        return dict(zip(ids, np.mean(ranks, axis=0).tolist()))


def sampled_case(rng, restricted=False, target=None):
    domains = dict(DOMAINS)
    if restricted:
        domains['minimum_distance'] = tuple(x for x in domains['minimum_distance'] if x <= 16)
    if target == 'P4':
        domains.update(minimum_distance=tuple(x for x in domains['minimum_distance'] if x >= 12),
                       blue_controller=('random',), max_cycles=(20, 40))
    elif target == 'P1':
        domains.update(minimum_distance=(12, 16), blue_controller=('random', 'rule_based'),
                       max_cycles=(150, 180, 200, 300))
    elif target == 'P2':
        domains.update(minimum_distance=(6, 7, 12), red_formation=('split',), blue_formation=('compact',),
                       blue_controller=('rule_based',), blue_target_rule=('lowest_health',), max_cycles=(120, 200))
    elif target in ('P3', 'P5'):
        domains.update(minimum_distance=(2, 4, 6, 7), blue_controller=('rule_based',), max_cycles=(60, 80, 120, 200))
    vals = {}
    for f, values in domains.items():
        value = rng.choice(values)
        vals[f] = value.item() if isinstance(value, np.generic) else value
    return normalize(ScenarioConfig(**vals))


def random_suite(seed, restricted=False, pool=None):
    rng = np.random.default_rng(seed)
    out, used = [], set()
    if pool is not None:
        for ix in rng.permutation(len(pool))[:30]:
            cid, c = pool[int(ix)]
            out.append(entry(c, 'random', cid, 'historical_pool_no_outcome_selection'))
        return out
    for _ in range(100000):
        c = sampled_case(rng, restricted)
        if signature(c) not in used and legal(c):
            out.append(entry(c, 'random'))
            used.add(signature(c))
        if len(out) == 30:
            return out
    raise RuntimeError('random domain exhausted')


def semantic_neighbor(c, rng, used, restricted=False):
    for f in rng.permutation(FIELDS):
        values = [v for v in DOMAINS[f] if v != getattr(c, f)
                  and not (restricted and f == 'minimum_distance' and v > 16)]
        for value in rng.permutation(values):
            value = value.item() if isinstance(value, np.generic) else value
            candidate = normalize(replace(c, **{f: value}))
            if signature(candidate) not in used and legal(candidate):
                return candidate
    return None


def reward_suite(h, seed, restricted=False, pool=None):
    rng = np.random.default_rng(seed)
    ranks = h.reward_rank()
    items = [(cid, c) for cid, c in (pool if pool is not None else h.items)
             if not restricted or c.minimum_distance <= 16]
    items.sort(key=lambda x: (ranks[x[0]], x[0]))
    if pool is not None:
        return [entry(c, 'low_reward', cid, 'screen_reward_rank') for cid, c in items[:30]]
    parents = items[:15]
    used = {signature(c) for _, c in parents}
    out = []
    for cid, c in parents:
        out.append(entry(c, 'low_reward', cid, 'screen_reward_rank'))
        child = semantic_neighbor(c, rng, used, restricted)
        if child is None:
            raise RuntimeError('reward neighborhood exhausted')
        used.add(signature(child))
        out.append(entry(child, 'low_reward', cid, 'unobserved_reward_neighbor'))
    return out


def ensure_variety(out, rng, restricted=False, pool=None):
    # Only static input constraints repair late slots, before any formal execution.
    requirements = [(f, v) for f in ('obstacle_layout', 'red_formation', 'blue_formation', 'engagement_region')
                    for v in DOMAINS[f]]
    protected = set()
    for f, v in requirements:
        if any(x['config'][f] == v for x in out):
            continue
        used = {signature(ScenarioConfig(**x['config'])) for x in out}
        for ix in reversed(range(5, len(out))):
            if ix in protected:
                continue
            others = out[:ix] + out[ix + 1:]
            lost = [(ff, vv) for ff, vv in requirements
                    if out[ix]['config'][ff] == vv and not any(x['config'][ff] == vv for x in others)]
            candidates = [c for _, c in pool if getattr(c, f) == v] if pool is not None else []
            if pool is None:
                base = ScenarioConfig(**out[ix]['config'])
                candidates = [normalize(replace(base, **{f: v}))]
                for _ in range(250):
                    candidates.append(normalize(replace(sampled_case(rng, restricted), **{f: v})))
            chosen = next((c for c in candidates if signature(c) not in used and legal(c)
                           and all(getattr(c, ff) == vv for ff, vv in lost)), None)
            if chosen is not None:
                out[ix] = entry(chosen, out[ix]['target'], evidence='static_input_variety_repair')
                protected.add(ix)
                break
        else:
            raise RuntimeError(f'Cannot satisfy input variety: {f}={v}')
    return out


def risk_suite(h, seed, restricted=False, no_history=False, state_only=False, pool=None):
    rng = np.random.default_rng(seed)
    items = [(cid, c) for cid, c in (pool if pool is not None else h.items)
             if not restricted or c.minimum_distance <= 16]
    targets = ([f'S{i:02}' for i in range(1, 13)] + [f'M{i}' for i in range(1, 9)]) if state_only else list(PRIORITY)
    out, used = [], set()
    counts = defaultdict(lambda: np.zeros(3))
    ties = {cid: float(rng.random()) for cid, _ in items}
    for slot in range(30):
        target = targets[slot % len(targets)]
        options = []
        if not no_history:
            for cid, c in items:
                if signature(c) in used:
                    continue
                rates, stage = h.rates(cid, target)
                if rates.max() > 0:
                    score = (stage == 'refine', float(np.sum(rates / (1 + counts[target]))),
                             float(rates.min()), ties[cid])
                    options.append((score, cid, c, rates, stage))
        if options:
            _, cid, c, rates, stage = max(options, key=lambda x: x[0])
            out.append(entry(c, target, cid, f'{stage}_target_evidence'))
            counts[target] += rates
        elif pool is not None:
            available = [(cid, c) for cid, c in items if signature(c) not in used]
            cid, c = available[0]
            out.append(entry(c, target, cid, 'no_historical_target_hit'))
        else:
            for _ in range(50000):
                c = sampled_case(rng, restricted, target)
                if signature(c) not in used and legal(c):
                    break
            else:
                raise RuntimeError('risk parameterization exhausted')
            out.append(entry(c, target, evidence='semantic_parameterization_no_hit_evidence'))
        used.add(signature(c))
    return ensure_variety(out, rng, restricted, pool)


def pairs(c):
    values = signature(c)
    return {(i, values[i], j, values[j]) for i in range(len(FIELDS)) for j in range(i + 1, len(FIELDS))}


@lru_cache(maxsize=2)
def geometry_values(restricted=False):
    domains = [DOMAINS[f] for f in GEOMETRY]
    if restricted:
        domains[0] = tuple(v for v in domains[0] if v <= 16)
    return tuple(values for values in itertools.product(*domains)
                 if not (values[3] == 'encircle' and values[2] != 'compact') and geometry_legal(values))


@lru_cache(maxsize=2)
def feasible_pair_set(restricted=False):
    geom = geometry_values(restricted)
    controllers = [('random', 'nearest', 'constant'), ('shared_model', 'nearest', 'constant')]
    controllers += [('rule_based', t, p) for t in DOMAINS['blue_target_rule'] for p in DOMAINS['phase_rule']]
    all_pairs = set()
    # Geometry and controller relations are constrained; their cross-products are independent.
    relations = [(GEOMETRY, geom), (FIELDS[6:9], controllers), ((FIELDS[9],), [(t,) for t in DOMAINS['max_cycles']])]
    for fields, rows in relations:
        for row in rows:
            for a, b in itertools.combinations(range(len(fields)), 2):
                all_pairs.add((FIELDS.index(fields[a]), row[a], FIELDS.index(fields[b]), row[b]))
    for (fa, ra), (fb, rb) in itertools.combinations(relations, 2):
        for i, f in enumerate(fa):
            for j, g in enumerate(fb):
                for a, b in itertools.product({r[i] for r in ra}, {r[j] for r in rb}):
                    all_pairs.add((FIELDS.index(f), a, FIELDS.index(g), b))
    return all_pairs


def combination_suite(seed, restricted=False, pool=None):
    rng = np.random.default_rng(seed)
    if pool is None:
        from allpairspy import AllPairs
        domains = [list(DOMAINS[f]) for f in FIELDS]
        if restricted:
            domains[0] = [v for v in domains[0] if v <= 16]
        for d in domains:
            rng.shuffle(d)
        geom = geometry_values(restricted)
        prefixes = {n: {g[:n] for g in geom} for n in range(1, 7)}

        def valid(values):
            n = min(len(values), 6)
            if n and tuple(values[:n]) not in prefixes[n]:
                return False
            if len(values) >= 8 and values[6] != 'rule_based' and values[7] != 'nearest':
                return False
            if len(values) >= 9 and values[6] != 'rule_based' and values[8] != 'constant':
                return False
            return True

        candidates = [(None, ScenarioConfig(**dict(zip(FIELDS, row))))
                      for row in AllPairs(domains, filter_func=valid)]
        universe = feasible_pair_set(restricted)
    else:
        candidates = list(pool)
        universe = set().union(*(pairs(c) for _, c in pool))
    generated_pairs = set().union(*(pairs(c) for _, c in candidates))
    raw_missing = len(universe - generated_pairs)
    if pool is None:
        # Complete missing legal pairs with static witnesses; no execution evidence is used.
        controls = [('random', 'nearest', 'constant'), ('shared_model', 'nearest', 'constant')]
        controls += [('rule_based', t, p) for t in DOMAINS['blue_target_rule'] for p in DOMAINS['phase_rule']]
        for i, a, j, b in sorted(universe - generated_pairs):
            if (i, a, j, b) in generated_pairs:
                continue
            required = {i: a, j: b}
            geo = next(g for g in geometry_values(restricted)
                       if all(g[k] == v for k, v in required.items() if k < 6))
            control = next(c for c in controls if all(c[k-6] == v for k, v in required.items() if 6 <= k < 9))
            values = (*geo, *control, required.get(9, DOMAINS['max_cycles'][0]))
            witness = ScenarioConfig(**dict(zip(FIELDS, values)))
            assert legal(witness)
            candidates.append((None, witness))
            generated_pairs |= pairs(witness)
    unique = {}
    for cid, c in candidates:
        unique.setdefault(signature(c), (cid, c))
    candidates = list(unique.values())
    covered, out = set(), []
    while len(out) < 30:
        if not candidates:
            raise RuntimeError('combination array contains fewer than 30 distinct rows')
        scores = [len(pairs(c) - covered) for _, c in candidates]
        best = int(np.argmax(scores))
        cid, c = candidates.pop(best)
        out.append(entry(c, 'parameter_pairs', cid, 'allpairspy_static_array' if pool is None else 'historical_pair_gain'))
        covered |= pairs(c)
    return out, {'array_size': len(unique), 'feasible_pairs': len(universe),
                 'array_pairs': len(generated_pairs), 'selected_pairs': len(covered),
                 'library_missing_pairs_before_static_completion': raw_missing,
                 'full_array_complete': generated_pairs == universe}


def suite_metrics(rows):
    states, paths, scenes = set(), set(), set()
    sc, kpc, atomic = [], [], []
    cumulative_steps, nsteps, tfc = [], 0, None
    for i, r in enumerate(rows, 1):
        states.update(r['covered_states'])
        paths.update(r['covered_paths'])
        scenes.update(r['covered_scenarios'])
        sc.append(len(states) / 9)
        kpc.append(len(paths) / 5)
        atomic.append(len(scenes) / 12)
        nsteps += r['steps']
        cumulative_steps.append(nsteps)
        if len(paths) == 5 and tfc is None:
            tfc = i
    return {'sc': sc, 'kpc': kpc, 'atomic': atomic, 'auc': float(np.mean(kpc)),
            'complete': int(len(paths) == 5), 'tfc': tfc, 'steps': nsteps,
            'cumulative_steps': cumulative_steps, 'paths': sorted(paths), 'scenes': sorted(scenes),
            'oracle_episodes': sum(bool(r['oracle_failures']) for r in rows),
            'illegal_episodes': sum(bool(r['illegal_edges']) for r in rows)}


def proximity(positions, walls, size):
    boundary = any(x <= 2 or y <= 2 or x >= size - 3 or y >= size - 3 for x, y in positions.values())
    wall = any(max(abs(x - wx), abs(y - wy)) <= 2 for x, y in positions.values() for wx, wy in walls)
    return boundary, wall


def freeze():
    target = ROOT / 'results/frozen.json'
    if target.exists():
        raise RuntimeError('Freeze already exists; never overwrite it')
    h = History()
    assert len(h.rows) == 2352
    assert not set(ENV_SEEDS) & {r['environment_seed'] for r in h.rows}
    # Scan all earlier experiment JSONL records, not only the selection subset.
    previous_seed_hits = []
    for old in ('9.17magent2_experiment', '9.20magent2_offline_experiment'):
        for path in (ROOT.parent / old / 'results').rglob('*.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                seed = r.get('environment_seed', r.get('test_case', {}).get('environment_seed'))
                if seed in ENV_SEEDS:
                    previous_seed_hits.append(str(path))
    assert not previous_seed_hits, previous_seed_hits
    suites = []
    catalog, used = [], {}
    source_by_signature = {signature(c): cid for cid, c in h.items}
    for batch in ('main', 'restricted', 'history_pool'):
        methods = ('random', 'reward', 'combination', 'risk', 'no_history', 'state_only') if batch == 'main' else ('random', 'reward', 'combination', 'risk')
        seeds = DESIGN_SEEDS if batch == 'main' else DESIGN_SEEDS[:1]
        pool = h.items if batch == 'history_pool' else None
        for method in methods:
            for d, seed in enumerate(seeds):
                kw = {'restricted': batch == 'restricted', 'pool': pool}
                info = {}
                if method == 'random':
                    cases = random_suite(seed, **kw)
                elif method == 'reward':
                    cases = reward_suite(h, seed, **kw)
                elif method == 'combination':
                    cases, info = combination_suite(seed, **kw)
                else:
                    cases = risk_suite(h, seed, no_history=method == 'no_history', state_only=method == 'state_only', **kw)
                assert len(cases) == 30
                ids = []
                for x in cases:
                    c = normalize(ScenarioConfig(**x['config']))
                    sig = signature(c)
                    assert legal(c)
                    if sig not in used:
                        cid = f'N{len(catalog)+1:04d}'
                        used[sig] = cid
                        catalog.append({'id': cid, 'config': replace(c, case_id=cid, target_scenario='evaluation').to_dict(),
                                        'historical_equivalent': source_by_signature.get(sig)})
                    x['id'] = used[sig]
                    ids.append(x['id'])
                assert len(set(ids)) == 30
                suite = {'batch': batch, 'method': method, 'design': d, 'design_seed': seed, 'ids': ids,
                         'cases': cases, 'combination_audit': info}
                suites.append(suite)
                dump(ROOT / f'results/suites/{batch}_{method}_{d}.json', suite)
                print('DESIGNED', batch, method, d, flush=True)
    history_table = []
    for cid, c in h.items:
        for p in POLICIES:
            for path in ('P1', 'P2', 'P3', 'P4', 'P5'):
                history_table.append({'candidate': cid, 'model': p, 'path': path,
                                      'screen_hits_n': h.count(cid, p, path, 'screen'),
                                      'refine_hits_n': h.count(cid, p, path, 'refine')})
    dump(ROOT / 'results/history_evidence.json', history_table)
    dump(ROOT / 'results/catalog.json', catalog)
    protected = sorted((ROOT / 'magent2_experiment').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    protected += sorted((ROOT / 'models').glob('*.pt')) + sorted((ROOT / 'history').glob('*'))
    protected += [ROOT / 'APPROVED_PLAN.md', ROOT / 'results/catalog.json']
    payload = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'suites': suites,
               'policies': list(POLICIES), 'environment_seeds': list(ENV_SEEDS),
               'candidate_count': len(catalog), 'expected_unique_episodes': len(catalog) * 60,
               'logical_episode_references': len(suites) * 30 * 60,
               'hashes': {str(p.relative_to(ROOT)): digest(p) for p in protected},
               'single_pass': True, 'red_inference': 'argmax', 'selection_history_episodes': 2352,
               'previous_seed_overlap': previous_seed_hits}
    dump(target, payload)
    print('FROZEN', payload['candidate_count'], payload['expected_unique_episodes'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze'])
    args = parser.parse_args()
    freeze()

from collections import Counter, defaultdict
import json

from risk_design import ROOT, ScenarioConfig, normalize, dump, read, suite_metrics
from magent2_experiment.scenario_geometry import build_team_positions, build_obstacles
from run_frozen import load_rows, check_frozen


def physical_signature(c):
    c = normalize(c)
    red, blue = build_team_positions(c)
    return (tuple(red), tuple(blue), tuple(build_obstacles(c.obstacle_layout)),
            c.blue_controller, c.blue_target_rule, c.phase_rule, c.max_cycles, c.red_action_rule)


def static_audit():
    f = check_frozen()
    out = []
    for s in f['suites']:
        seen, duplicates = {}, []
        for i, case in enumerate(s['cases'], 1):
            sig = physical_signature(ScenarioConfig(**case['config']))
            if sig in seen:
                duplicates.append({'position': i, 'same_as_position': seen[sig], 'id': case['id']})
            else:
                seen[sig] = i
        out.append({'batch': s['batch'], 'method': s['method'], 'design': s['design'],
                    'physically_distinct': len(seen), 'duplicates': duplicates,
                    'historical_direct': sum(c['evidence'] in ('refine_target_evidence','screen_target_evidence','screen_reward_rank') for c in s['cases']),
                    'unobserved_or_other': dict(Counter(c['evidence'] for c in s['cases']))})
    dump(ROOT / 'results/static_physical_audit.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def finished_audit():
    f = check_frozen()
    rows = load_rows()
    lookup = {(r['candidate_id'], r['model_seed'], r['environment_seed']): r for r in rows}
    output = defaultdict(list)
    for s in f['suites']:
        if s['batch'] != 'main':
            continue
        physical, indices = set(), []
        for i, case in enumerate(s['cases']):
            sig = physical_signature(ScenarioConfig(**case['config']))
            if sig not in physical:
                physical.add(sig)
                indices.append(i)
        for p in f['policies']:
            for seed in f['environment_seeds']:
                all_rows = [lookup[(cid,p,seed)] for cid in s['ids']]
                distinct_rows = [all_rows[i] for i in indices]
                m = suite_metrics(distinct_rows)
                output[s['method']].append({'physical_n': len(indices), 'kpc5': m['kpc'][4],
                                            'kpc10': m['kpc'][9], 'final_kpc': m['kpc'][-1],
                                            'final_complete': m['complete'], 'sc': m['sc'][-1],
                                            'model': p, 'seed': seed, 'design': s['design'],
                                            'atomic_missing': [f'S{i:02}' for i in range(1,13) if f'S{i:02}' not in m['scenes']]})
    dump(ROOT / 'results/analysis/physical_dedup_sensitivity.json', output)


if __name__ == '__main__':
    import sys
    if '--finished' in sys.argv:
        finished_audit()
    else:
        static_audit()

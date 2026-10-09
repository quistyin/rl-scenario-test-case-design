from pathlib import Path
import sys
import unittest
import json
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_experiment as replay


class StrictP4Tests(unittest.TestCase):
    def test_published_permutations_and_exact_auc(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / 'data/reordering_plan.json').read_text('utf-8'))
        report = json.loads((root / 'results/reordering_summary.json').read_text('utf-8'))
        self.assertEqual(len(plan['orders']), 100)
        self.assertEqual(len({tuple(o) for o in plan['orders']}), 100)
        for order in plan['orders']:
            self.assertEqual(sorted(order), list(range(30)))
        expected = {'risk': 0.8986, 'random': 0.7128, 'combination': 0.6891, 'reward': 0.3357}
        for method, result in report['methods'].items():
            exact = Fraction(sum(result['path_count_sums']), result['permutation_denominator'] * 100)
            self.assertEqual(str(exact), result['mean_auc_fraction'])
            self.assertEqual(float(exact), result['mean_auc'])
            self.assertEqual(round(float(exact), 4), expected[method])
            self.assertEqual(result['combinations'], 300)

    def test_terminal_discovery_disqualifies_p4(self):
        trace = ['M0', 'M1', 'M8']
        self.assertIn('P4', replay.replay_paths(trace, 'timeout'))
        self.assertNotIn('P4', replay.strict_replay_paths(
            trace, 'timeout', [['M0'], ['M1'], ['M2', 'M8']]))

    def test_earlier_discovery_disqualifies_p4(self):
        self.assertNotIn('P4', replay.strict_replay_paths(
            ['M0', 'M1', 'M8'], 'timeout', [['M0', 'M2'], ['M1'], ['M8']]))

    def test_clean_timeout_and_wrong_terminal(self):
        trace, flags = ['M0', 'M1', 'M8'], [['M0'], ['M1'], ['M8']]
        self.assertEqual({'P4'}, replay.strict_replay_paths(trace, 'timeout', flags))
        self.assertEqual(set(), replay.strict_replay_paths(trace, 'red_failure', flags))

    def test_other_paths_are_preserved(self):
        trace = ['M0', 'M1', 'M2', 'M3', 'M4', 'M7']
        self.assertEqual(replay.replay_paths(trace, 'red_success'),
                         replay.strict_replay_paths(trace, 'red_success', [[s] for s in trace]))

    def test_missing_or_malformed_flags_fail_closed(self):
        for flags in (None, [], [['M0']], ['M0', 'M1', 'M8'],
                      [['M0'], ['M1'], ['M99']], [['M0'], ['M1'], []]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                replay.strict_replay_paths(['M0', 'M1', 'M8'], 'timeout', flags)

    def test_fresh_result_uses_same_rule_without_mutating_input(self):
        raw = {'state_trace': ['M0', 'M1', 'M8'],
               'state_flags': [['M0'], ['M1'], ['M2', 'M8']],
               'test_case': {'terminal_tag': 'timeout'}, 'covered_paths': ['P4']}
        corrected = replay.evaluate_episode(raw)
        self.assertEqual(corrected['covered_paths'], [])
        self.assertEqual(corrected['archived_covered_paths'], ['P4'])
        self.assertEqual(raw['covered_paths'], ['P4'])

    def test_reordering_recomputes_cumulative_union_not_mean_case_coverage(self):
        index = {('a', 11, 1): {'covered_paths': ['P1']},
                 ('b', 11, 1): {'covered_paths': ['P1', 'P2']}}
        plan = {'suites': [{'method': 'risk', 'ids': ['a', 'b']}],
                'policies': [11], 'environment_seeds': [1]}
        report = replay.reordering_report(index, plan, [[0, 1], [1, 0]])
        self.assertEqual(report['methods']['risk']['mean_auc'], 0.35)
        self.assertEqual(report['methods']['risk']['permutation_means'], [0.3, 0.4])
        for orders in ([], [[0, 0]], [[0]], [[0, 2]], [[False, 1]]):
            with self.subTest(orders=orders), self.assertRaises(ValueError):
                replay.reordering_report(index, plan, orders)


if __name__ == '__main__':
    unittest.main()

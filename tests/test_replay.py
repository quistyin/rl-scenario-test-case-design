import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class ReplayTests(unittest.TestCase):
    def test_public_replay_entry_exists(self):
        self.assertTrue((ROOT / 'scripts/replay_experiment.py').is_file())

    @unittest.skipUnless((ROOT / 'scripts/replay_experiment.py').is_file(), 'entry not implemented')
    def test_union_coverage_counts_and_auc(self):
        from replay_experiment import coverage_row
        rows = [{'covered_states': ['M0'], 'covered_paths': []},
                {'covered_states': ['M0', 'M1'], 'covered_paths': ['P1']}]
        out = coverage_row(rows, 'risk', 0, 11, 1)
        self.assertEqual(out['sc'], [1 / 9, 2 / 9])
        self.assertEqual(out['kpc'], [0, 1 / 5])
        self.assertEqual(out['auc'], 0.1)
        self.assertIsNone(out['tfc'])

    @unittest.skipUnless((ROOT / 'scripts/replay_experiment.py').is_file(), 'entry not implemented')
    def test_unknown_state_is_rejected(self):
        from replay_experiment import coverage_row
        with self.assertRaises(ValueError):
            coverage_row([{'covered_states': ['M99'], 'covered_paths': []}], 'risk', 0, 11, 1)

    @unittest.skipUnless((ROOT / 'scripts/replay_experiment.py').is_file(), 'entry not implemented')
    def test_safe_archive_member(self):
        from replay_experiment import safe_member
        self.assertEqual(safe_member('results\\execution\\traces\\a.json.gz'), 'results/execution/traces/a.json.gz')
        for name in ('../a', '/a', 'C:/a', 'a/../../b'):
            with self.assertRaises(ValueError):
                safe_member(name)


if __name__ == '__main__':
    unittest.main()

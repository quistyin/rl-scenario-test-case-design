import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce


class StatisticsTests(unittest.TestCase):
    def test_extreme_tail(self):
        result = reproduce.signed_rank_exact([1.0] * 20)
        self.assertEqual(result['statistic'], 0)
        self.assertEqual(result['p_raw'], 2 / 2**20)

    def test_opposite_signs_and_zeros(self):
        result = reproduce.signed_rank_exact([0, 1, -1])
        self.assertEqual(result['nonzero_pairs'], 2)
        self.assertEqual(result['p_raw'], 1.0)

    def test_holm_preserves_input_order(self):
        self.assertEqual(reproduce.holm([0.03, 0.01, 0.02]), [0.04, 0.03, 0.04])

    def test_all_zero_differences(self):
        with self.assertRaises(ValueError):
            reproduce.signed_rank_exact([0, 0])


class PublicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((ROOT / 'data/combination_metrics.json').read_text(encoding='utf-8'))

    def test_full_matched_grid(self):
        groups = reproduce.validate_rows(self.rows)
        self.assertEqual({name: len(rows) for name, rows in groups.items()},
                         {name: 300 for name in reproduce.METHODS})

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            reproduce.validate_rows(self.rows + [self.rows[0]])

    def test_nonmonotone_curve_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['sc'][1:3] = [1.0, 0.0]
        with self.assertRaises(ValueError):
            reproduce.validate_rows(rows)

    def test_auc_mismatch_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['auc'] = -1
        with self.assertRaises(ValueError):
            reproduce.validate_rows(rows)

    def test_paper_values(self):
        groups = reproduce.validate_rows(self.rows)
        table = {row['method']: row for row in reproduce.table_rows(groups)}
        self.assertAlmostEqual(table['risk']['KPC@5/%'], 97)
        self.assertAlmostEqual(table['risk']['KPC-AUC'], .9403333333333333)
        self.assertAlmostEqual(table['reward']['KPC-AUC'], .3211333333333333)
        self.assertAlmostEqual(table['reward']['SC@5/%'], 54.96296296296296)

    def test_complete_family_of_twelve(self):
        stats = reproduce.statistics_report(reproduce.validate_rows(self.rows))
        self.assertEqual(len(stats['comparisons']), 12)
        self.assertTrue(all(row['p_holm'] == 12 * 2 / 2**20 for row in stats['comparisons']))


if __name__ == '__main__':
    unittest.main()

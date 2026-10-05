from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_public


class ReleaseScanTests(unittest.TestCase):
    def test_scanner_does_not_flag_its_own_patterns(self):
        verify_public.scan([ROOT / 'scripts/verify_public.py'])

    def test_registered_model_is_allowed(self):
        verify_public.scan([ROOT / 'artifacts/models/seed_11.pt'])

    def test_unregistered_model_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_public.scan([ROOT / 'artifacts/models/other.pt'])


if __name__ == '__main__':
    unittest.main()

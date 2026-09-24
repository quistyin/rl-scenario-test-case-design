from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_public


class ReleaseScanTests(unittest.TestCase):
    def test_scanner_does_not_flag_its_own_patterns(self):
        verify_public.scan([ROOT / 'scripts/verify_public.py'])


if __name__ == '__main__':
    unittest.main()

"""Check the public release file list, hashes and summary consistency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from reproduce import validate_rows, table_rows, statistics_report, compare_reference

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'generated'}
PRIVATE_SUFFIXES = {'.doc', '.docx', '.pt', '.pth', '.ckpt', '.pem', '.key', '.gz', '.jsonl'}
PRIVATE_PARTS = {'local_archive', 'models', 'history', 'traces', 'raw', 'logs', 'pip-cache'}
TEXT_SUFFIXES = {'.py', '.json', '.csv', '.md', '.txt'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    result = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if set(relative.parts) & IGNORED:
            continue
        if path.is_symlink():
            raise ValueError('Symbolic link is not allowed in the public package')
        if path.is_file() and relative.as_posix() != 'MANIFEST.sha256':
            result.append(path)
    return sorted(result)


def scan(paths):
    patterns = [r'[A-Za-z]:[\\/](?:Users|用户)[\\/]', r'/Users/[A-Za-z0-9_.-]+/',
                r'/home/[A-Za-z0-9_.-]+/', r'gh[pousr]_[A-Za-z0-9]{20,}',
                r'github_pat_[A-Za-z0-9_]{30,}', r'AKIA[A-Z0-9]{16}',
                r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in PRIVATE_SUFFIXES or set(relative.parts) & PRIVATE_PARTS:
            raise ValueError(f'Nonpublic file in package: {relative}')
        if path.name.startswith('.env') or path.stat().st_size >= 25 * 1024 * 1024:
            raise ValueError(f'Unexpected environment file or large file: {relative}')
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding='utf-8')
            if any(re.search(pattern, text) for pattern in patterns):
                raise ValueError(f'Possible local path or credential: {relative}')
    provenance = json.loads((ROOT / 'docs/source_manifest.json').read_text(encoding='utf-8'))
    for item in provenance['source_code']:
        if digest(ROOT / item['file']) != item['sha256']:
            raise ValueError(f'Original experiment source was modified: {item["file"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Maintainer only: regenerate after an intentional reviewed update')
    args = parser.parse_args()
    paths = files()
    scan(paths)
    manifest = ROOT / 'MANIFEST.sha256'
    actual = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    if args.write_manifest:
        manifest.write_text(''.join(f'{sha}  {name}\n' for name, sha in actual.items()), encoding='utf-8')
    expected = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        sha, name = line.split('  ', 1)
        if name in expected:
            raise ValueError('Duplicate manifest entry')
        expected[name] = sha
    if actual != expected:
        raise ValueError('Public file set or SHA-256 differs from the published manifest')
    rows = json.loads((ROOT / 'data/combination_metrics.json').read_text(encoding='utf-8'))
    groups = validate_rows(rows)
    compare_reference(table_rows(groups), statistics_report(groups))
    print(json.dumps({'status': 'passed', 'files': len(paths),
                      'bytes': sum(path.stat().st_size for path in paths),
                      'combinations': len(rows), 'methods': 4, 'paired_comparisons': 12,
                      'private_file_and_pattern_scan': 'passed',
                      'source_hashes': 'unchanged', 'simulations_run': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()

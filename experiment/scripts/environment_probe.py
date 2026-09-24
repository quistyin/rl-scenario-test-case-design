from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magent2_experiment.runner import inspect_environment, load_policy


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if root not in output.parents:
        raise ValueError(f"output must stay below experiment root: {output}")

    checkpoints = sorted((root / "models").glob("seed_*.pt"))
    models = []
    for checkpoint in checkpoints:
        policy = load_policy(checkpoint)
        models.append(
            {
                "path": checkpoint.relative_to(root).as_posix(),
                "sha256": file_sha256(checkpoint),
                "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            }
        )

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            name: version(name)
            for name in ("magent2", "torch", "numpy", "scipy", "pandas", "pytest")
        },
        "torch": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "threads": torch.get_num_threads(),
        },
        "numpy_seed_probe": np.random.default_rng(20260918).integers(0, 10_000, 5).tolist(),
        "environment": inspect_environment(),
        "models": models,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

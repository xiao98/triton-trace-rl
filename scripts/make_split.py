"""Freeze data/splits/manifest.json: train_source vs heldout_test (run once, never again).

Stratified by level: within each level, problems are ordered by SHA-256(seed + id) and the
first `HELDOUT_FRACTION` go to `heldout_test`. Deterministic, independent of dataset order,
and decided before any model has been run on any problem.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "environments" / "triton_kernel_env"))

from triton_kernel_env import splits  # noqa: E402

DATASET = "ScalingIntelligence/KernelBench"
REVISION = "ca1464e5d4723fd14f87b8e68306aaa1fe66b81b"
LEVELS = [1, 2]
SEED = "triton-trace-rl/split/v1"
HELDOUT_FRACTION = 0.4


def main() -> None:
    if splits.MANIFEST_PATH.exists():
        sys.exit(f"{splits.MANIFEST_PATH} already exists: the split is frozen, refusing to overwrite")
    manifest = {"dataset": DATASET, "revision": REVISION, "levels": LEVELS, "seed": SEED,
                "heldout_fraction": HELDOUT_FRACTION}  # fmt: skip
    rows = splits.fetch_kernelbench(manifest)
    train, heldout = [], []
    for level in LEVELS:
        ids = sorted(
            (pid for pid, r in rows.items() if r["level"] == level),
            key=lambda pid: hashlib.sha256(f"{SEED}/{pid}".encode()).hexdigest(),
        )
        k = round(len(ids) * HELDOUT_FRACTION)
        heldout += sorted(ids[:k])
        train += sorted(ids[k:])
    manifest["splits"] = {splits.TRAIN: train, splits.HELDOUT: heldout}
    manifest["sha256"] = {pid: splits.sha256(rows[pid]["ref_src"]) for pid in sorted(rows)}
    manifest["created_at_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    splits.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"train_source={len(train)} heldout_test={len(heldout)} -> {splits.MANIFEST_PATH}")


if __name__ == "__main__":
    main()

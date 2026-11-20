"""Frozen train_source / heldout_test split and the guard that enforces red line 4.

`data/splits/manifest.json` is written once by `scripts/make_split.py` and never edited.
Every script obtains problems through `load_problems`, which

- refuses to hand out `heldout_test` unless the caller declares `purpose="heldout_eval"`
  (failure diagnosis, TRACE synthesis and RL training have no such purpose);
- verifies every reference source against the SHA-256 recorded in the manifest, so the
  split cannot silently drift with the upstream dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "splits" / "manifest.json"
TRAIN, HELDOUT = "train_source", "heldout_test"
HELDOUT_PURPOSE = "heldout_eval"


class HeldoutAccessError(RuntimeError):
    """Raised when anything but a final held-out evaluation asks for `heldout_test`."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def problem_ids(split: str, *, purpose: str, manifest: dict | None = None) -> list[str]:
    if split not in (TRAIN, HELDOUT):
        raise ValueError(f"unknown split {split!r}")
    if split == HELDOUT and purpose != HELDOUT_PURPOSE:
        raise HeldoutAccessError(
            f"purpose {purpose!r} may not read {HELDOUT}; only {HELDOUT_PURPOSE!r} may"
        )
    manifest = manifest or load_manifest()
    return list(manifest["splits"][split])


def fetch_kernelbench(manifest: dict) -> dict[str, dict]:
    """All Level 1-2 problems at the pinned revision, keyed by name."""
    from datasets import load_dataset

    rows: dict[str, dict] = {}
    for level in manifest["levels"]:
        ds = load_dataset(manifest["dataset"], split=f"level_{level}", revision=manifest["revision"])
        for row in ds:
            rows[f"level{level}/{row['name']}"] = {"level": level, "ref_src": row["code"]}
    return rows


def load_problems(
    split: str, *, purpose: str, manifest: dict | None = None, rows: dict[str, dict] | None = None
) -> list[dict]:
    """`[{id, level, ref_src}]` for one split, hash-checked against the manifest."""
    manifest = manifest or load_manifest()
    ids = problem_ids(split, purpose=purpose, manifest=manifest)
    rows = rows if rows is not None else fetch_kernelbench(manifest)
    out = []
    for pid in ids:
        ref_src = rows[pid]["ref_src"]
        if sha256(ref_src) != manifest["sha256"][pid]:
            raise RuntimeError(f"{pid}: reference source differs from the frozen manifest")
        out.append({"id": pid, "level": rows[pid]["level"], "ref_src": ref_src})
    return out

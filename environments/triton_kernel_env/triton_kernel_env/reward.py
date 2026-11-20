"""Shaped RL reward and official KernelBench metrics — deliberately separate.

`shaped_reward` is the piecewise training signal for GRPO. `kernelbench_metrics` is what
evaluation tables report. Neither is ever derived from the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from triton_kernel_env import sandbox_runner as sr

ZERO_STATUSES = {sr.STATIC_REJECT, sr.CHEAT, sr.TIMEOUT, sr.CRASHED, sr.INFRA_ERROR}

R_COMPILED = 0.2
R_CORRECT = 0.5
SPEEDUP_CAP = 1.5


def shaped_reward(result: Mapping | sr.EvalResult) -> float:
    """cheat / crash / timeout = 0.0; compiled = 0.2; multi-seed correct = 0.5;
    correct and faster than the reference = 0.5 + min(1.5, speedup)."""
    r = result.to_dict() if isinstance(result, sr.EvalResult) else result
    if r["status"] in ZERO_STATUSES or r.get("cheats") or r.get("violations"):
        return 0.0
    if r["correct"]:
        speedup = r.get("speedup")
        if speedup is not None and speedup > 1.0:
            return R_CORRECT + min(SPEEDUP_CAP, speedup)
        return R_CORRECT
    return R_COMPILED if r["compiled"] else 0.0


def kernelbench_metrics(results: Iterable[Mapping]) -> dict[str, float]:
    """compile_rate, correct_rate and fast_p over *all* samples (KernelBench `fast_p`:
    fraction of samples that are correct and have speedup > p)."""
    rows = list(results)
    n = len(rows)
    if n == 0:
        raise ValueError("no results")

    def clean(r: Mapping) -> bool:
        return not (r["status"] in (sr.STATIC_REJECT, sr.CHEAT) or r.get("cheats"))

    def fast(p: float) -> float:
        hits = [
            r for r in rows
            if clean(r) and r["correct"] and r.get("speedup") is not None and r["speedup"] > p
        ]  # fmt: skip
        return len(hits) / n

    return {
        "n": n,
        "compile_rate": sum(1 for r in rows if clean(r) and r["compiled"]) / n,
        "correct_rate": sum(1 for r in rows if clean(r) and r["correct"]) / n,
        "fast_1": fast(1.0),
        "fast_1.5": fast(1.5),
        "fast_2": fast(2.0),
    }

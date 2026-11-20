"""`torch.profiler` bottleneck analysis and the feedback text shown between turns.

`collect_profile` runs inside the sandbox child. `build_feedback` runs in the parent and
turns an `EvalResult` into the next user message at one of three information levels:

- `errors`  : compile / runtime / correctness verdicts only
- `timing`  : `errors` + measured runtime and speedup
- `profile` : `timing` + the profiler breakdown and derived bottleneck hints
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from triton_kernel_env import sandbox_runner as sr

FEEDBACK_MODES = ("errors", "timing", "profile")


def collect_profile(
    run: Callable[[], object], *, cuda: bool, repeats: int = 3, top_k: int = 12
) -> dict:
    """Profile `run()` and return a JSON-serialisable summary (times in microseconds)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=activities) as prof:
        for _ in range(repeats):
            run()
            if cuda:
                torch.cuda.synchronize()

    def dev(evt) -> float:  # attribute was renamed across torch versions
        for name in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(evt, name, None)
            if v is not None:
                return float(v)
        return 0.0

    rows = [
        {
            "name": e.key[:80],
            "calls": int(e.count),
            "self_cpu_us": float(e.self_cpu_time_total),
            "self_device_us": dev(e),
        }
        for e in prof.key_averages()
    ]
    key = "self_device_us" if cuda else "self_cpu_us"
    rows.sort(key=lambda r: r[key], reverse=True)
    return {
        "repeats": repeats,
        "sorted_by": key,
        "total_self_cpu_us": sum(r["self_cpu_us"] for r in rows),
        "total_self_device_us": sum(r["self_device_us"] for r in rows),
        "rows": rows[:top_k],
    }


def analyze_profile(prof: Mapping) -> list[str]:
    """Derive plain-language bottleneck hints from a profile summary."""
    hints: list[str] = []
    rows, key = prof["rows"], prof["sorted_by"]
    on_device = key == "self_device_us"
    total = prof["total_self_device_us"] if on_device else prof["total_self_cpu_us"]
    if not rows or total <= 0:
        return hints
    top = rows[0]
    unit = "device" if on_device else "CPU"
    hints.append(
        f"Hottest entry: `{top['name']}` takes {top[key] / total:.0%} of profiled {unit} time."
    )

    def frac(*needles: str) -> float:
        hit = [r[key] for r in rows if any(n in r["name"].lower() for n in needles)]
        return sum(hit) / total

    copies = frac("memcpy", "copy_", "clone", "contiguous")
    if copies > 0.15:
        hints.append(
            f"{copies:.0%} of time is spent copying memory (clone / contiguous / memcpy): "
            "avoid extra copies around the kernel, handle strides or layout inside it."
        )
    allocs = frac("empty", "zeros", "memset", "fill_")
    if allocs > 0.15:
        hints.append(
            f"{allocs:.0%} of time is spent allocating / initialising buffers: "
            "allocate with torch.empty and write every element from the kernel."
        )
    calls = max(r["calls"] for r in rows) / max(prof["repeats"], 1)
    if calls > 4:
        hints.append(
            f"About {calls:.0f} calls per forward of the most frequent entry: "
            "fuse stages into fewer kernel launches."
        )
    if on_device and prof["total_self_cpu_us"] > 5 * total:
        hints.append(
            "Host-side time dominates device time: the kernel is launch-bound, so "
            "reduce Python work and launches rather than tuning the kernel body."
        )
    return hints


def format_profile(prof: Mapping) -> str:
    lines = [f"{'name':<48}{'calls':>7}{'self_cpu_us':>14}{'self_device_us':>16}"]
    for r in prof["rows"]:
        lines.append(
            f"{r['name'][:47]:<48}{r['calls']:>7}{r['self_cpu_us']:>14.1f}"
            f"{r['self_device_us']:>16.1f}"
        )
    return "\n".join(lines)


REPLY_FORMAT = "\n\nReply with the complete new file in a single ```python code block."


def build_feedback(result: Mapping | sr.EvalResult, mode: str = "profile") -> str:
    """Next-turn user message for a submission's evaluation result. Always ends by asking
    for a full file: without that, models answer a bare verdict with prose, which then
    burns a submission as a syntax error."""
    return _verdict(result, mode) + REPLY_FORMAT


def _verdict(result: Mapping | sr.EvalResult, mode: str) -> str:
    if mode not in FEEDBACK_MODES:
        raise ValueError(f"mode must be one of {FEEDBACK_MODES}")
    r = result.to_dict() if isinstance(result, sr.EvalResult) else result
    status = r["status"]
    err = r.get("error") or {}

    if status == sr.STATIC_REJECT:
        items = "\n".join(
            f"- line {v['lineno']}: [{v['rule']}] {v['message']}" for v in r["violations"]
        )
        return (
            "Your submission was REJECTED by the static Triton-only check and scored 0:\n"
            f"{items}\n"
            "All computation must happen inside @triton.jit kernels launched from "
            "ModelNew.forward. Fix every item and resubmit the full file."
        )
    if status == sr.CHEAT:
        return (
            "Your submission was REJECTED at runtime and scored 0: "
            + ", ".join(r["cheats"])
            + ". PyTorch may only allocate and reshape tensors; inputs must not be "
            "modified; the output must be computed by a Triton kernel from the inputs."
        )
    if status == sr.TIMEOUT:
        return (
            f"Your submission was killed at the hard time limit (stage: {r['timeout_stage']}) "
            "and scored 0. Remove unbounded loops and reduce compile / run time."
        )
    if status == sr.CRASHED:
        return (
            "Your submission crashed the evaluation process (likely an out-of-bounds "
            "memory access in a kernel) and scored 0. Check every tl.load / tl.store "
            "offset and mask.\n" + (r.get("log_tail") or "")[-800:]
        )
    if status == sr.INFRA_ERROR:
        return "The evaluator failed for reasons unrelated to your code. Resubmit unchanged."
    if status in (sr.COMPILE_ERROR, sr.RUNTIME_ERROR):
        kind = "failed to compile / load" if status == sr.COMPILE_ERROR else "raised at runtime"
        return (
            f"Your submission {kind}:\n{err.get('type')}: {err.get('message')}\n"
            f"{(err.get('traceback') or '')[-1500:]}"
        )
    if status == sr.INCORRECT:
        return (
            "Your submission compiled and ran but the output is WRONG on hidden random "
            f"inputs: {r['mismatch']}"
        )

    msg = "Your submission is CORRECT on all hidden random seeds."
    if mode == "errors" or r.get("speedup") is None:
        return msg + " It scores higher the faster it runs; submit a faster version."
    verdict = "faster than" if r["speedup"] > 1.0 else "not faster than"
    msg += (
        f"\nRuntime: {r['cand_ms']:.4f} ms vs PyTorch reference {r['ref_ms']:.4f} ms "
        f"-> speedup {r['speedup']:.3f}x ({verdict} the reference)."
    )
    if mode == "profile" and r.get("profile"):
        hints = analyze_profile(r["profile"])
        msg += "\n\ntorch.profiler breakdown of your forward pass:\n"
        msg += format_profile(r["profile"])
        if hints:
            msg += "\n\nBottleneck analysis:\n" + "\n".join(f"- {h}" for h in hints)
    return msg + "\nSubmit an improved full file to go faster."

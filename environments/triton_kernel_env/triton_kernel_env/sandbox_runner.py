"""Parent-process side of the sandbox: static guard, then a hard-isolated child.

`evaluate_kernel()` never executes untrusted code in the calling process. Each call
spawns a fresh interpreter (`multiprocessing` with the `"spawn"` start method), talks to
it over a pipe, enforces a hard deadline per stage and kills + reaps the child on
overrun, crash or protocol violation. A segfault, infinite loop or OOM in the candidate
therefore surfaces as a `status`, never as an exception or a hang in the caller.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import secrets
import tempfile
import time
from dataclasses import asdict, dataclass, field

from triton_kernel_env import _sandbox_child
from triton_kernel_env.ast_guard import check_source

# status values, in the order a submission can progress
STATIC_REJECT = "static_reject"  # AST guard violation (cheat or malformed) -> 0.0
INFRA_ERROR = "infra_error"  # reference / environment broken, not the candidate's fault
COMPILE_ERROR = "compile_error"  # failed to load or to run the first forward
RUNTIME_ERROR = "runtime_error"  # raised on a later seed or while being timed
CHEAT = "cheat"  # runtime guard violation -> 0.0
TIMEOUT = "timeout"  # killed at a stage deadline -> 0.0
CRASHED = "crashed"  # child died (segfault, OOM kill, ...) -> 0.0
INCORRECT = "incorrect"  # compiled, ran, wrong numbers
CORRECT = "correct"  # multi-seed correct (speedup set when timing ran)


@dataclass(frozen=True)
class Limits:
    """Hard per-stage deadlines in seconds. Clocks start at the previous stage message."""

    startup_s: float = 180.0  # trusted: import torch/triton, build + run the reference
    verify_s: float = 25.0  # untrusted: load candidate, compile, multi-seed check
    ref_timing_s: float = 120.0  # trusted: time the reference
    timing_s: float = 20.0  # untrusted: time the candidate + strict re-check
    profile_s: float = 20.0  # untrusted: torch.profiler pass
    n_warmup: int = 5
    n_trials: int = 20
    n_seeds: int = 3
    tolerance: float = 1e-4  # KernelBench fp32 atol = rtol


@dataclass
class EvalResult:
    status: str
    compiled: bool = False
    correct: bool = False
    speedup: float | None = None
    ref_ms: float | None = None
    cand_ms: float | None = None
    cheats: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)  # static guard findings
    error: dict | None = None
    mismatch: str = ""
    timeout_stage: str | None = None
    exitcode: int | None = None
    seeds: list[int] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    profile: dict | None = None
    env: dict = field(default_factory=dict)
    log_tail: str = ""
    wall_s: float = 0.0
    attempts: int = 1

    @property
    def is_cheat(self) -> bool:
        return self.status in (STATIC_REJECT, CHEAT)

    def to_dict(self) -> dict:
        return asdict(self)


# stage just received -> (next stage that must arrive, limit attribute)
_NEXT = {
    "ready": ("verified", "verify_s"),
    "verified": ("ref_timed", "ref_timing_s"),
    "ref_timed": ("timed", "timing_s"),
    "timed": ("profiled", "profile_s"),
    "profiled": ("done", "profile_s"),
}


def _reap(proc: mp.Process) -> int | None:
    proc.join(2)  # a child that reported `done` exits on its own
    if proc.is_alive():
        proc.kill()
        proc.join(10)
    code = proc.exitcode
    if code is not None:
        proc.close()
    return code


def _tail(path: str, n: int = 3000) -> str:
    try:
        with open(path, "rb") as f:
            return f.read()[-n:].decode("utf-8", "replace")
    except OSError:
        return ""


def evaluate_kernel(
    ref_src: str,
    cand_src: str,
    *,
    measure_performance: bool = True,
    profile: bool = False,
    seeds: list[int] | None = None,
    limits: Limits = Limits(),
    allow_cpu: bool | None = None,
) -> EvalResult:
    """Evaluate one candidate against a KernelBench-format reference.

    `ref_src` defines `Model`, `get_inputs()`, `get_init_inputs()`; `cand_src` defines
    `ModelNew`. Seeds default to fresh OS randomness so a submission cannot anticipate
    them; they are recorded on the result for replay.
    """
    t_start = time.monotonic()
    if seeds is None:
        seeds = [secrets.randbelow(2**31) for _ in range(limits.n_seeds)]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("need at least 3 distinct seeds")
    static = check_source(cand_src)
    if not static.ok:
        return EvalResult(
            STATIC_REJECT, seeds=seeds, violations=static.to_dict()["violations"],
            wall_s=time.monotonic() - t_start,
        )  # fmt: skip
    # An infra error is by definition not the candidate's doing: retry once, keep the count.
    for attempt in (1, 2):
        res = run_sandboxed(
            ref_src, cand_src, seeds=seeds, measure_performance=measure_performance,
            profile=profile, limits=limits, allow_cpu=allow_cpu,
        )  # fmt: skip
        res.attempts = attempt
        if res.status != INFRA_ERROR:
            break
    return res


def run_sandboxed(
    ref_src: str,
    cand_src: str,
    *,
    seeds: list[int],
    measure_performance: bool = True,
    profile: bool = False,
    limits: Limits = Limits(),
    allow_cpu: bool | None = None,
) -> EvalResult:
    """The isolated-subprocess half of `evaluate_kernel`, WITHOUT the static guard.

    Exists so tests can prove the sandbox contains code the guard would never let
    through (segfaults, `os._exit`). Never call it on model output directly.
    """
    t_start = time.monotonic()
    if allow_cpu is None:
        allow_cpu = os.environ.get("TRITON_ENV_ALLOW_CPU") == "1"

    fd, log_path = tempfile.mkstemp(prefix="triton_env_", suffix=".log")
    os.close(fd)
    payload = {
        "ref_src": ref_src, "cand_src": cand_src, "seeds": seeds,
        "measure_performance": measure_performance, "profile": profile,
        "tolerance": limits.tolerance, "n_warmup": limits.n_warmup,
        "n_trials": limits.n_trials, "allow_cpu": allow_cpu, "log_path": log_path,
    }  # fmt: skip
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_sandbox_child.child_main, args=(child, payload), daemon=True)
    proc.start()
    child.close()

    msgs: dict[str, dict] = {}
    expect, deadline = "ready", time.monotonic() + limits.startup_s
    timeout_stage, eof = None, False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not parent.poll(remaining):
                timeout_stage = expect
                break
            try:
                msg = parent.recv()
            except (EOFError, OSError):
                eof = True
                break
            if not isinstance(msg, dict) or "stage" not in msg:
                eof = True
                break
            msgs[msg["stage"]] = msg
            if msg["stage"] == "done":
                break
            if msg["stage"] in _NEXT:
                expect, attr = _NEXT[msg["stage"]]
                deadline = time.monotonic() + getattr(limits, attr)
    finally:
        parent.close()
        exitcode = _reap(proc)
        log_tail = _tail(log_path)
        try:
            os.unlink(log_path)
        except OSError:
            pass

    res = _assemble(msgs, timeout_stage, eof)
    res.seeds, res.exitcode, res.log_tail = seeds, exitcode, log_tail
    res.wall_s = time.monotonic() - t_start
    return res


def _assemble(msgs: dict[str, dict], timeout_stage: str | None, eof: bool) -> EvalResult:
    """Fold the stage messages (possibly cut short) into one verdict."""
    done, ver, timed = msgs.get("done", {}), msgs.get("verified"), msgs.get("timed")
    res = EvalResult(INFRA_ERROR)
    res.env = {k: v for k, v in msgs.get("ready", {}).items() if k != "stage"}
    res.compiled = "compiled" in msgs or bool(ver and ver.get("compiled"))

    if "infra_error" in done or "fatal" in done and "ready" not in msgs:
        res.error = done.get("infra_error") or done.get("fatal")
        return res
    if timeout_stage == "ready" or (eof and "ready" not in msgs):
        res.error = {"type": "StartupFailure", "message": "child never became ready"}
        res.timeout_stage = timeout_stage
        return res

    res.cheats = list(dict.fromkeys(
        (ver or {}).get("cheats", []) + (timed or {}).get("cheats", [])
    ))  # fmt: skip
    if ver:
        res.mismatch = ver.get("mismatch", "")
        res.error = ver.get("error")
    if timed and timed.get("error"):
        res.error = timed["error"]

    # Order matters: a cheat outranks everything; a kill outranks partial success.
    if res.cheats:
        res.status = CHEAT
    elif timeout_stage == "ref_timed":  # the trusted reference overran, not the candidate
        res.timeout_stage = timeout_stage
        res.error = {"type": "ReferenceTimingTimeout", "message": "reference timing overran"}
    elif timeout_stage is not None:
        res.status, res.timeout_stage = TIMEOUT, timeout_stage
    elif eof or "fatal" in done or "done" not in msgs:
        res.status = CRASHED
        res.error = res.error or done.get("fatal")
    elif ver is None:
        res.status = CRASHED
    elif ver.get("error"):
        res.status = RUNTIME_ERROR if ver.get("compiled") else COMPILE_ERROR
    elif not ver["correct"]:
        res.status = INCORRECT
    elif timed and timed.get("error"):
        res.status = RUNTIME_ERROR
    else:
        res.status, res.correct = CORRECT, True
        if timed:
            res.speedup = timed["speedup"]
            res.cand_ms = timed["cand_ms"]
            res.ref_ms = msgs["ref_timed"]["ref_ms"]
            res.timing = {k: v for k, v in timed.items() if k not in ("stage", "cheats")}
            res.timing["ref_wall_ms"] = msgs["ref_timed"]["ref_wall_ms"]
        prof = msgs.get("profiled")
        if prof:
            res.profile = prof.get("profile")
    return res

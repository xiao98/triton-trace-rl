"""Child-process side of the sandbox. Runs in a fresh `spawn`ed interpreter.

Protocol: the child sends `{"stage": <name>, ...}` dicts over a pipe as it progresses
(`ready` → `compiled` → `verified` → `ref_timed` → `timed` → `profiled` → `done`). The
parent enforces a hard deadline per stage and kills the child on overrun, so this module
never needs to defend against its own hangs — only to report honestly what it saw.

Trusted code (the benchmark's reference `Model`) runs before `ready`; everything after
`ready` executes untrusted candidate code.
"""

from __future__ import annotations

import os
import random
import time
import traceback

# aten ops the candidate may trigger during a guarded forward: allocation, shape, dtype /
# device conversion and plain copies. Anything else is PyTorch doing the model's compute.
ALLOWED_ATEN_OPS = {
    "aten::empty", "aten::empty_like", "aten::zeros", "aten::zeros_like", "aten::ones",
    "aten::ones_like", "aten::full", "aten::full_like", "aten::new_empty",
    "aten::new_zeros", "aten::new_ones", "aten::new_full", "aten::arange",
    "aten::lift_fresh", "aten::clone", "aten::detach", "aten::alias", "aten::_to_copy",
    "aten::copy_", "aten::fill_", "aten::zero_", "aten::view", "aten::_unsafe_view",
    "aten::reshape", "aten::_reshape_alias", "aten::t", "aten::transpose",
    "aten::permute", "aten::squeeze", "aten::unsqueeze", "aten::expand", "aten::slice",
    "aten::select", "aten::unbind", "aten::split", "aten::split_with_sizes",
    "aten::unfold", "aten::cat", "aten::stack", "aten::as_strided", "aten::contiguous",
    "aten::flatten", "aten::movedim", "aten::narrow", "aten::unflatten",
}  # fmt: skip


def _set_seed(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _clone(torch, items, device=None):
    out = []
    for x in items:
        if isinstance(x, torch.Tensor):
            x = x.detach().clone()
            if device is not None:
                x = x.to(device)
        out.append(x)
    return out


def _flatten_output(torch, out):
    """Model output as a flat list of tensors, or None if it has any other shape."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)) and out and all(isinstance(t, torch.Tensor) for t in out):
        return list(out)
    return None


def _err(e: BaseException) -> dict:
    tb = traceback.format_exc()
    return {"type": type(e).__name__, "message": str(e)[:2000], "traceback": tb[-4000:]}


class RuntimeGuard:
    """Observes what the candidate does at runtime; records, never raises."""

    def __init__(self, torch, triton_jit) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        self.active = False
        self.launches = 0
        self.nn_calls: list[str] = []
        self.op_violations: list[str] = []
        guard = self

        orig_run = triton_jit.JITFunction.run

        def run(self_, *args, **kwargs):
            if not kwargs.get("warmup", False):
                guard.launches += 1
            return orig_run(self_, *args, **kwargs)

        triton_jit.JITFunction.run = run

        orig_call = torch.nn.Module.__call__

        def call(self_, *args, **kwargs):
            if guard.active and type(self_).__module__.startswith("torch.nn"):
                guard.nn_calls.append(type(self_).__name__)
            return orig_call(self_, *args, **kwargs)

        torch.nn.Module.__call__ = call

        class _OpMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = func._schema.name
                if name not in ALLOWED_ATEN_OPS:
                    guard.op_violations.append(name)
                return func(*args, **(kwargs or {}))

        self._mode_cls = _OpMode

    def forward(self, torch, model, inputs, *, strict: bool):
        """Run the candidate. `strict` adds the op allow-list (too slow for timing)."""
        self.active = True
        try:
            with torch.no_grad():
                if strict:
                    with self._mode_cls():
                        return model(*inputs)
                return model(*inputs)
        finally:
            self.active = False

    def drain(self) -> list[str]:
        found = [f"nn_module_call:{n}" for n in dict.fromkeys(self.nn_calls)]
        found += [f"torch_compute_fallback:{n}" for n in dict.fromkeys(self.op_violations)]
        self.nn_calls.clear()
        self.op_violations.clear()
        return found


def _compare(torch, cand, ref, tol: float) -> tuple[bool, str]:
    """(matches, human-readable reason)."""
    if len(cand) != len(ref):
        return False, f"expected {len(ref)} output tensor(s), got {len(cand)}"
    for i, (c, r) in enumerate(zip(cand, ref)):
        if c.shape != r.shape:
            return False, f"output {i}: shape {tuple(c.shape)} != expected {tuple(r.shape)}"
        if c.dtype != r.dtype:
            return False, f"output {i}: dtype {c.dtype} != expected {r.dtype}"
        if c.device != r.device:
            return False, f"output {i}: device {c.device} != expected {r.device}"
        if not torch.isfinite(c).all() and torch.isfinite(r).all():
            return False, f"output {i}: contains NaN/Inf"
        if not torch.allclose(c, r, atol=tol, rtol=tol, equal_nan=True):
            diff = (c.double() - r.double()).abs()
            return False, (
                f"output {i}: values differ (max abs err {diff.max().item():.3e}, "
                f"mean abs err {diff.mean().item():.3e}, tol {tol:g})"
            )
    return True, "ok"


def _timed_call(torch, run, inputs, seed, cuda):
    """One forward on fresh clones. Returns (output, inputs_used, event_ms, wall_ms);
    CUDA events with a synchronize before and after."""
    x = _clone(torch, inputs)  # fresh tensors each call: identity caching is useless
    _set_seed(torch, seed)
    if cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    t0 = time.perf_counter()
    out = run(x)
    if cuda:
        end.record()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    return out, x, (start.elapsed_time(end) if cuda else wall), wall


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def child_main(conn, payload: dict) -> None:
    log = open(payload["log_path"], "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    send = conn.send
    try:
        _run(send, payload)
    except BaseException as e:  # noqa: BLE001 - report, then die
        send({"stage": "done", "fatal": _err(e)})
    finally:
        conn.close()


def _run(send, p: dict) -> None:
    # ---------------------------------------------------------------- trusted phase
    try:
        import torch
        import triton
        import triton.runtime.jit as triton_jit
    except Exception as e:  # noqa: BLE001
        send({"stage": "done", "infra_error": _err(e)})
        return

    try:  # torch imports this lazily on first use; keep that cost off the candidate's clock
        import torch._dynamo  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    cuda = torch.cuda.is_available()
    if not cuda and not p["allow_cpu"]:
        send({"stage": "done", "infra_error": {"type": "NoCuda", "message": "CUDA unavailable"}})
        return
    device = torch.device("cuda" if cuda else "cpu")
    seeds: list[int] = p["seeds"]
    tol: float = p["tolerance"]

    try:
        ref_ns: dict = {"__name__": "kernelbench_reference"}
        exec(compile(p["ref_src"], "<reference>", "exec"), ref_ns)
        Model, get_inputs, get_init_inputs = (
            ref_ns["Model"], ref_ns["get_inputs"], ref_ns["get_init_inputs"],
        )  # fmt: skip
        cases = []
        for seed in seeds:
            _set_seed(torch, seed)
            init_inputs = _clone(torch, get_init_inputs(), device)
            _set_seed(torch, seed)
            ref_model = Model(*_clone(torch, init_inputs)).to(device)
            _set_seed(torch, seed)
            inputs = _clone(torch, get_inputs(), device)
            _set_seed(torch, seed)
            with torch.no_grad():
                ref_out = _flatten_output(torch, ref_model(*_clone(torch, inputs)))
            if ref_out is None:
                raise TypeError("reference output is not a tensor or sequence of tensors")
            cases.append({"seed": seed, "init": init_inputs, "inputs": inputs,
                          "ref_model": ref_model, "ref_out": ref_out})  # fmt: skip
        if cuda:
            torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        send({"stage": "done", "infra_error": _err(e)})
        return

    guard = RuntimeGuard(torch, triton_jit)
    send({
        "stage": "ready",
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "triton_is_stub": bool(getattr(triton, "IS_TEST_STUB", False)),
    })  # fmt: skip

    # -------------------------------------------------------------- untrusted phase
    cheats: list[str] = []
    cand_models, cand_outs = [], []
    mismatch = ""
    for i, case in enumerate(cases):
        try:
            if i == 0:
                ns: dict = {"__name__": "candidate_submission"}
                exec(compile(p["cand_src"], p["cand_path"], "exec"), ns)
                ModelNew = ns["ModelNew"]
            _set_seed(torch, case["seed"])
            model = ModelNew(*_clone(torch, case["init"])).to(device)
            x = _clone(torch, case["inputs"])
            launches_before = guard.launches
            _set_seed(torch, case["seed"])
            out = guard.forward(torch, model, x, strict=True)
            if cuda:
                torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            cheats += guard.drain()
            send({"stage": "verified", "compiled": i > 0, "correct": False,
                  "cheats": cheats, "error": _err(e), "failed_seed_index": i})  # fmt: skip
            send({"stage": "done"})
            return
        if i == 0:
            send({"stage": "compiled"})
        cheats += guard.drain()
        if guard.launches == launches_before:
            cheats.append("no_triton_launch")
        for a, b in zip(x, case["inputs"]):
            if isinstance(a, torch.Tensor) and not torch.equal(a, b):
                cheats.append("input_mutation")
                break
        flat = _flatten_output(torch, out)
        if flat is None:
            mismatch = mismatch or f"output is {type(out).__name__}, expected tensor(s)"
            flat = []
        elif any(type(t) is not torch.Tensor for t in flat):
            cheats.append("tensor_subclass_output")
        else:
            ok, why = _compare(torch, flat, case["ref_out"], tol)
            if not ok:
                mismatch = mismatch or f"seed #{i}: {why}"
        cand_models.append(model)
        cand_outs.append(flat)

    def same(a, b):
        return len(a) == len(b) and all(
            x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
        )

    if len(cases) >= 2 and cand_outs[0]:
        cand_const = all(same(cand_outs[0], o) for o in cand_outs[1:])
        ref_const = all(same(cases[0]["ref_out"], c["ref_out"]) for c in cases[1:])
        if cand_const and not ref_const:
            cheats.append("input_independent_output")

    cheats = list(dict.fromkeys(cheats))
    correct = not cheats and not mismatch
    send({"stage": "verified", "compiled": True, "correct": correct, "cheats": cheats,
          "mismatch": mismatch})  # fmt: skip
    del cand_outs
    if not correct or not p["measure_performance"]:
        send({"stage": "done"})
        return

    # ------------------------------------------------------------------ timing
    # Timed calls draw at random from a pool of inputs the candidate has never seen, and
    # every output is checked. Replaying an earlier answer is therefore wrong, and telling
    # the pool items apart requires reading tensor values, which the strict calls forbid.
    rng = random.SystemRandom()
    case = cases[0]
    ref_model, cand_model = case["ref_model"], cand_models[0]
    n_trials = p["n_trials"]

    def run_ref(x):
        with torch.no_grad():
            return ref_model(*x)

    try:
        pool = []
        for _ in range(4):
            seed = rng.randrange(2**31)
            _set_seed(torch, seed)
            inputs = _clone(torch, get_inputs(), device)
            _set_seed(torch, seed)
            pool.append({"seed": seed, "inputs": inputs,
                         "ref_out": _flatten_output(torch, run_ref(_clone(torch, inputs)))})  # fmt: skip
        ref_ev, ref_wall = [], []
        for k in range(p["n_warmup"] + n_trials):
            item = rng.choice(pool)
            _, _, ev_ms, wall_ms = _timed_call(torch, run_ref, item["inputs"], item["seed"], cuda)
            if k >= p["n_warmup"]:
                ref_ev.append(ev_ms)
                ref_wall.append(wall_ms)
    except Exception as e:  # noqa: BLE001
        send({"stage": "done", "infra_error": _err(e)})
        return
    send({"stage": "ref_timed", "ref_ms": _median(ref_ev), "ref_wall_ms": _median(ref_wall)})

    # Warmups (random count) and one trailing call run under the strict op allow-list;
    # only the timed block in between runs without the dispatcher hook's overhead.
    n_warmup = p["n_warmup"] + rng.randrange(0, 16)
    ev, wall = [], []
    try:
        for k in range(n_warmup + n_trials + 1):
            timed = n_warmup <= k < n_warmup + n_trials
            item = rng.choice(pool)
            launches_before = guard.launches
            out, x, ev_ms, wall_ms = _timed_call(
                torch, lambda x: guard.forward(torch, cand_model, x, strict=not timed),  # noqa: B023
                item["inputs"], item["seed"], cuda,
            )  # fmt: skip
            cheats += guard.drain()
            if guard.launches == launches_before:
                cheats.append("no_triton_launch_during_timing")
            flat = _flatten_output(torch, out)
            if (flat is None or any(type(t) is not torch.Tensor for t in flat)
                    or not _compare(torch, flat, item["ref_out"], tol)[0]):  # fmt: skip
                cheats.append("output_wrong_after_verification")
            if any(isinstance(a, torch.Tensor) and not torch.equal(a, b)
                   for a, b in zip(x, item["inputs"])):  # fmt: skip
                cheats.append("input_mutation")
            if cheats:
                break
            if timed:
                ev.append(ev_ms)
                wall.append(wall_ms)
    except Exception as e:  # noqa: BLE001
        send({"stage": "timed", "error": _err(e), "cheats": cheats + guard.drain()})
        send({"stage": "done"})
        return
    if cheats:
        send({"stage": "timed", "cheats": list(dict.fromkeys(cheats))})
        send({"stage": "done"})
        return

    cand_ms, cand_wall = _median(ev), _median(wall)
    # Work hidden from the default stream shows up as wall time far above event time.
    suspicious = cuda and cand_wall > 1.5 * cand_ms + 0.2
    speedup = (
        _median(ref_wall) / cand_wall if suspicious else _median(ref_ev) / max(cand_ms, 1e-9)
    )
    send({
        "stage": "timed", "cheats": list(dict.fromkeys(cheats)),
        "cand_ms": cand_ms, "cand_wall_ms": cand_wall, "speedup": speedup,
        "timing_suspicious": bool(suspicious), "n_warmup": n_warmup,
        "n_trials": n_trials,
        "timing_method": "cuda_event" if cuda else "cpu_perf_counter_TEST_ONLY",
    })  # fmt: skip

    if p["profile"] and not cheats:
        from triton_kernel_env.profiler_feedback import collect_profile

        try:
            # Cloned outside the profiled region, or the harness's own copy would be
            # reported to the model as its bottleneck.
            prof_inputs = _clone(torch, pool[0]["inputs"])
            prof = collect_profile(
                lambda: guard.forward(torch, cand_model, prof_inputs, strict=False),
                cuda=cuda,
            )  # fmt: skip
            send({"stage": "profiled", "profile": prof})
        except Exception as e:  # noqa: BLE001
            send({"stage": "profiled", "profile": None, "error": _err(e)})
    send({"stage": "done"})

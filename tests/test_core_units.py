"""Unit tests for the static guard's accept side, the reward mapping, the official
metrics and the feedback text. (The guard's reject side lives in the red-team file.)"""

import pytest

from kernels import HONEST_ADD, HONEST_PARAM, PREAMBLE
from triton_kernel_env import sandbox_runner as sr
from triton_kernel_env.ast_guard import check_source
from triton_kernel_env.profiler_feedback import (
    analyze_profile, build_feedback, collect_profile,
)  # fmt: skip
from triton_kernel_env.prompts import EXAMPLE_SOLUTION, SMOKE_TASKS, build_prompt, extract_code
from triton_kernel_env.reward import kernelbench_metrics, shaped_reward

LEVEL2_STYLE = PREAMBLE + """
import math
from typing import Optional


@triton.autotune(configs=[triton.Config({"BLOCK": 64}, num_warps=4)], key=["n"])
@triton.jit
def fused(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        nn.init.zeros_(self.gemm.bias)
        self.scale = 1.0 / math.sqrt(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gemm.weight.t().contiguous()
        out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=torch.float32)
        n = max(x.numel(), 1)
        fused[(triton.cdiv(n, 64),)](x.contiguous(), w, out, n)
        return out.view(x.shape[0], -1)
"""


@pytest.mark.parametrize(
    "src",
    [HONEST_ADD, HONEST_PARAM, EXAMPLE_SOLUTION, LEVEL2_STYLE, *[g for _, _, g in SMOKE_TASKS]],
)
def test_guard_accepts_honest_sources(src):
    res = check_source(src)
    assert res.ok, [str(v) for v in res.violations]


def test_guard_tolerates_unused_functional_import_but_not_its_use():
    src = HONEST_ADD.replace("import triton\n", "import torch.nn.functional as F\nimport triton\n", 1)
    assert check_source(src).ok
    assert not check_source(src.replace("return out", "return F.relu(out)")).ok


@pytest.mark.parametrize(
    "src,rule",
    [
        ("def f(:\n", "syntax_error"),
        ("import torch\n", "missing_model_new"),
        (PREAMBLE + "class ModelNew(nn.Module):\n    pass\n", "missing_forward"),
        ("import torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self, x):\n"
         "        return x\n", "no_triton_kernel"),
        (HONEST_ADD.replace("tl.constexpr):", "tl.constexpr):\n    torch.zeros(1)", 1),
         "torch_inside_kernel"),
    ],
)  # fmt: skip
def test_guard_structure_rules(src, rule):
    assert rule in [v.rule for v in check_source(src).violations]


# ------------------------------------------------------------------ reward / metrics
def _r(status, compiled=False, correct=False, speedup=None, cheats=()):
    return {"status": status, "compiled": compiled, "correct": correct, "speedup": speedup,
            "cheats": list(cheats), "violations": []}  # fmt: skip


@pytest.mark.parametrize(
    "result,expected",
    [
        (_r(sr.STATIC_REJECT), 0.0),
        (_r(sr.CHEAT, compiled=True, cheats=["input_mutation"]), 0.0),
        (_r(sr.TIMEOUT, compiled=True), 0.0),
        (_r(sr.CRASHED, compiled=True), 0.0),
        (_r(sr.INFRA_ERROR), 0.0),
        (_r(sr.COMPILE_ERROR), 0.0),
        (_r(sr.INCORRECT, compiled=True), 0.2),
        (_r(sr.RUNTIME_ERROR, compiled=True), 0.2),
        (_r(sr.CORRECT, True, True, None), 0.5),
        (_r(sr.CORRECT, True, True, 0.7), 0.5),
        (_r(sr.CORRECT, True, True, 1.0), 0.5),
        (_r(sr.CORRECT, True, True, 1.01), 0.51),
        (_r(sr.CORRECT, True, True, 1.2), 0.7),
        (_r(sr.CORRECT, True, True, 2.5), 2.0),
        (_r(sr.CORRECT, True, True, 9.0), 2.0),
        # a cheat flag wins even if other fields look perfect
        (_r(sr.CORRECT, True, True, 9.0, cheats=["no_triton_launch"]), 0.0),
    ],
)
def test_shaped_reward_table(result, expected):
    assert shaped_reward(result) == pytest.approx(expected)


def test_kernelbench_metrics_are_fractions_of_all_samples():
    rows = [
        _r(sr.CORRECT, True, True, 2.5),
        _r(sr.CORRECT, True, True, 1.2),
        _r(sr.CORRECT, True, True, 0.5),
        _r(sr.INCORRECT, True),
        _r(sr.COMPILE_ERROR),
        _r(sr.CHEAT, True, cheats=["input_mutation"]),
        _r(sr.STATIC_REJECT),
        _r(sr.TIMEOUT, True),
    ]
    m = kernelbench_metrics(rows)
    assert m == {"n": 8, "compile_rate": 5 / 8, "correct_rate": 3 / 8, "fast_1": 2 / 8,
                 "fast_1.5": 1 / 8, "fast_2": 1 / 8}  # fmt: skip
    with pytest.raises(ValueError):
        kernelbench_metrics([])


# ------------------------------------------------------------------ feedback
PROFILE = {
    "repeats": 3, "sorted_by": "self_device_us", "total_self_cpu_us": 9000.0,
    "total_self_device_us": 1000.0,
    "rows": [
        {"name": "add_kernel", "calls": 30, "self_cpu_us": 10.0, "self_device_us": 600.0},
        {"name": "Memcpy DtoD", "calls": 3, "self_cpu_us": 5.0, "self_device_us": 300.0},
        {"name": "aten::zeros", "calls": 3, "self_cpu_us": 5.0, "self_device_us": 100.0},
    ],
}  # fmt: skip


def test_analyze_profile_derives_hints():
    text = " ".join(analyze_profile(PROFILE))
    assert "add_kernel" in text and "60%" in text
    assert "copying memory" in text
    assert "fewer kernel launches" in text
    assert "launch-bound" in text


def test_feedback_information_levels_are_nested():
    res = dict(_r(sr.CORRECT, True, True, 1.31), ref_ms=2.0, cand_ms=1.5267, profile=PROFILE)
    errors, timing, profile = (build_feedback(res, m) for m in ("errors", "timing", "profile"))
    assert "CORRECT" in errors and "speedup" not in errors
    assert "1.310x" in timing and "torch.profiler" not in timing
    assert "torch.profiler" in profile and "Bottleneck analysis" in profile
    with pytest.raises(ValueError):
        build_feedback(res, "everything")


@pytest.mark.parametrize(
    "result,needle",
    [
        (dict(_r(sr.STATIC_REJECT), violations=[{"rule": "zombie_kernel", "message": "m",
                                                  "lineno": 3}]), "zombie_kernel"),
        (_r(sr.CHEAT, True, cheats=["input_mutation"]), "input_mutation"),
        (dict(_r(sr.TIMEOUT), timeout_stage="verified"), "hard time limit"),
        (dict(_r(sr.CRASHED), log_tail="Segmentation fault"), "Segmentation fault"),
        (dict(_r(sr.COMPILE_ERROR), error={"type": "NameError", "message": "x", "traceback": ""}),
         "NameError"),
        (dict(_r(sr.INCORRECT, True), mismatch="shape (2,) != expected (3,)"), "WRONG"),
        (_r(sr.INFRA_ERROR), "unrelated to your code"),
    ],
)  # fmt: skip
def test_feedback_for_every_failure_status(result, needle):
    assert needle in build_feedback(result, "profile")


def test_collect_profile_on_cpu_returns_ranked_rows():
    import torch

    x = torch.randn(64, 64)
    prof = collect_profile(lambda: x @ x, cuda=False)
    assert prof["rows"] and prof["sorted_by"] == "self_cpu_us"
    assert prof["rows"][0]["self_cpu_us"] >= prof["rows"][-1]["self_cpu_us"]
    assert any("mm" in r["name"] or "matmul" in r["name"] for r in prof["rows"])


def test_profile_reaches_the_result_through_the_sandbox():
    from kernels import REF_ADD

    res = sr.evaluate_kernel(REF_ADD, HONEST_ADD, profile=True, seeds=[1, 2, 3])
    assert res.status == sr.CORRECT and res.profile and res.profile["rows"]
    assert "torch.profiler" in build_feedback(res, "profile")


# ------------------------------------------------------------------ prompts
def test_prompt_and_code_extraction():
    prompt = build_prompt(SMOKE_TASKS[1][1])
    assert "torch.relu" in prompt and "get_inputs" in prompt and "@triton.jit" in prompt
    assert extract_code("x\n```python\nA\n```\ny\n```python\nB = 1\n```") == "B = 1"
    assert extract_code("no fences") == "no fences"
    assert extract_code(None) == ""

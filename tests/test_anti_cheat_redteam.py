"""Red team: every devious cheat or crash must score exactly 0.0 and leave the parent alive.

Categories (the brief asks for at least six):
  1. PyTorch compute fallback        (F.*, torch.matmul, `@`, tensor methods, bare operators)
  2. zombie @triton.jit              (decorative kernel, dead-code launch, decoy launch)
  3. in-place input modification
  4. single-seed hard-coding         (cached first answer, constant output)
  5. forbidden import / monkey-patch (os, subprocess, getattr tricks, patched allclose, ...)
  6. infinite loop / crash           (hang at import, hang in forward, segfault, hard exit)

Static cases never start a subprocess; runtime cases go through the real sandbox.
"""

import os
import time

import pytest

from kernels import (
    HEADER, HONEST_ADD, HONEST_PARAM, LAUNCH, NAME_ERROR, PREAMBLE, REF_ADD, REF_MATMUL,
    REF_PARAM, WRONG_SUB, binary_kernel, model,
)  # fmt: skip
from triton_kernel_env import sandbox_runner as sr
from triton_kernel_env.reward import shaped_reward

SEEDS = [11, 222, 3333]
FAST = sr.Limits(verify_s=6.0, timing_s=6.0, profile_s=6.0)

DECOY = """
scratch_a = torch.zeros(8, device=a.device)
scratch_b = torch.zeros(8, device=a.device)
scratch_o = torch.empty(8, device=a.device)
launch(scratch_a, scratch_b, scratch_o)
"""


def run(ref: str, cand: str, **kw) -> sr.EvalResult:
    kw.setdefault("seeds", SEEDS)
    kw.setdefault("limits", FAST)
    return sr.evaluate_kernel(ref, cand, **kw)


def assert_zero(res: sr.EvalResult, status: str, needle: str = "") -> None:
    print(res.to_dict())  # shown in full by pytest only when the test fails
    assert res.status == status, res.to_dict()
    assert shaped_reward(res) == 0.0
    assert not res.correct
    if needle:
        found = res.cheats + [v["rule"] for v in res.violations]
        assert any(needle in f for f in found), found


# ------------------------------------------------------------------ positive controls
def test_control_honest_kernel_scores_at_least_correct():
    res = run(REF_ADD, HONEST_ADD)
    assert res.status == sr.CORRECT, res.to_dict()
    assert res.speedup is not None and res.timing["n_trials"] >= 20
    assert res.timing["n_warmup"] >= 5
    assert shaped_reward(res) >= 0.5


def test_control_slow_reference_data_does_not_eat_the_candidate_clock():
    """Test data is built per seed between candidate calls; a reference whose inputs are
    slow to generate must not push an honest kernel over the verify deadline."""
    slow_ref = REF_ADD.replace(
        "def get_inputs():\n", "def get_inputs():\n    import time\n    time.sleep(2.5)\n"
    )
    res = run(slow_ref, HONEST_ADD, limits=sr.Limits(verify_s=5.0, timing_s=6.0))
    assert res.status == sr.CORRECT and res.speedup is not None, res.to_dict()


def test_control_untimeable_problem_is_correct_without_a_speedup():
    msgs = {"ready": {"stage": "ready"}, "compiled": {"stage": "compiled"},
            "verified": {"stage": "verified", "compiled": True, "correct": True, "cheats": []},
            "done": {"stage": "done", "timing_skipped": "not enough GPU memory"}}  # fmt: skip
    res = sr._assemble(msgs, None, False)
    assert res.status == sr.CORRECT and res.speedup is None
    assert res.timing == {"skipped": "not enough GPU memory"} and shaped_reward(res) == 0.5


def test_control_parameter_container_keeps_weight_parity():
    res = run(REF_PARAM, HONEST_PARAM)
    assert res.status == sr.CORRECT, res.to_dict()


def test_control_honest_but_wrong_scores_compiled_not_zero():
    res = run(REF_ADD, WRONG_SUB)
    assert res.status == sr.INCORRECT, res.to_dict()
    assert shaped_reward(res) == 0.2


# ------------------------------------------------------------------ 1. torch fallback
FALLBACK_STATIC = {
    "F_alias": (REF_ADD, HEADER.replace("import triton\n", "import torch.nn.functional as F\nimport triton\n")
                + binary_kernel() + LAUNCH + """
class ModelNew(nn.Module):
    def forward(self, a, b):
        out = torch.empty_like(a)
        launch(a, b, out)
        return F.relu(out) + F.relu(-out) * -1
""", "functional_fallback"),
    "torch_nn_functional": (REF_ADD, model("""
out = torch.empty_like(a)
launch(a, b, out)
return torch.nn.functional.hardtanh(out, -1e9, 1e9)
"""), "functional_fallback"),
    "torch_matmul": (REF_MATMUL, model(DECOY + "return torch.matmul(a, b)"), "torch_op_not_allowed"),
    "torch_mm_from_import": (REF_MATMUL, "from torch import mm\n" + model(DECOY + "return mm(a, b)"),
                             "torch_op_not_allowed"),
    "matmul_operator": (REF_MATMUL, model(DECOY + "return a @ b"), "matmul_operator"),
    "tensor_method_matmul": (REF_MATMUL, model(DECOY + "return a.matmul(b)"), "tensor_compute_method"),
    "torch_einsum": (REF_MATMUL, model(DECOY + "return torch.einsum('ij,jk->ik', a, b)"),
                     "torch_op_not_allowed"),
    "torch_add": (REF_ADD, model(DECOY + "return torch.add(a, b)"), "torch_op_not_allowed"),
    "nn_linear_called": (REF_MATMUL, PREAMBLE + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64, bias=False)

    def forward(self, a, b):
        out = torch.empty_like(a)
        launch(a, b, out)
        return self.fc(a)
""", "nn_module_call"),
    "nn_layer_built_in_forward": (REF_ADD, model(DECOY + "return nn.Identity()(a)"),
                                  "nn_layer_outside_init"),
}  # fmt: skip


@pytest.mark.parametrize("name", FALLBACK_STATIC)
def test_1_torch_fallback_rejected_statically(name):
    ref, cand, rule = FALLBACK_STATIC[name]
    assert_zero(run(ref, cand), sr.STATIC_REJECT, rule)


def test_1_operator_fallback_caught_at_runtime():
    """`a + b` is invisible to the AST (operands could be ints); the dispatcher sees it."""
    res = run(REF_ADD, model(DECOY + "return a + b"))
    assert_zero(res, sr.CHEAT, "torch_compute_fallback:aten::add")


def test_1_nn_layer_reached_through_container_caught_at_runtime():
    cand = PREAMBLE + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, a, b):
        out = torch.empty_like(a)
        launch(a, b, out)
        for layer in self.layers:
            out = layer(out)
        return out
"""
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "nn_module_call:Identity")


# ------------------------------------------------------------------ 2. zombie kernels
def test_2_zombie_kernel_never_called():
    cand = PREAMBLE + """
class ModelNew(nn.Module):
    def forward(self, a, b):
        return a + b
"""
    assert_zero(run(REF_ADD, cand), sr.STATIC_REJECT, "zombie_kernel")


def test_2_zombie_kernel_only_in_unreachable_helper():
    cand = PREAMBLE + """
def unused(a, b):
    out = torch.empty_like(a)
    launch(a, b, out)
    return out

class ModelNew(nn.Module):
    def forward(self, a, b):
        return a + b
"""
    assert_zero(run(REF_ADD, cand), sr.STATIC_REJECT, "zombie_kernel")


def test_2_zombie_launch_in_dead_code_caught_at_runtime():
    cand = model("""
if a.dim() > 99:
    out = torch.empty_like(a)
    launch(a, b, out)
return a + b
""")
    res = run(REF_ADD, cand)
    assert_zero(res, sr.CHEAT, "no_triton_launch")
    assert any("aten::add" in c for c in res.cheats)


def test_2_raw_python_function_behind_jit_object():
    cand = model(DECOY + "return add_kernel.fn(a, b, torch.empty_like(a), 0, 0)")
    assert_zero(run(REF_ADD, cand), sr.STATIC_REJECT, "raw_kernel_fn")


# ------------------------------------------------------------------ 3. in-place inputs
def test_3_inplace_input_modification():
    """Writes the (correct) result into the input buffer and returns it."""
    cand = model("""
launch(a, b, a)
return a
""")
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "input_mutation")


def test_3_correct_output_but_input_clobbered():
    cand = model("""
out = torch.empty_like(a)
launch(a, b, out)
b.zero_()
return out
""")
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "input_mutation")


# ------------------------------------------------------------------ 4. hard-coding
def test_4_first_answer_cached_and_replayed():
    """Honest on the first call, then replays that answer for every later input."""
    cand = model("""
if CACHE:
    launch(torch.zeros(8, device=a.device), torch.zeros(8, device=a.device), torch.empty(8, device=a.device))
    return CACHE[0]
out = torch.empty_like(a)
launch(a, b, out)
CACHE.append(out)
return out
""", extra="\nCACHE = []\n")
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "input_independent_output")


def test_4_constant_output_with_decoy_launch():
    cand = model(DECOY + "return torch.full((256, 512), 0.5, device=a.device)")
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "input_independent_output")


def test_4_reseeding_to_predict_inputs_is_blocked():
    cand = model(DECOY + "torch.manual_seed(11)\nreturn torch.empty_like(a)")
    assert_zero(run(REF_ADD, cand), sr.STATIC_REJECT, "torch_op_not_allowed")


def test_4_honest_during_checks_then_cheats_while_timed():
    """Counts calls: honest for the 3 verification calls, torch fallback afterwards."""
    cand = model("""
CALLS.append(1)
out = torch.empty_like(a)
launch(a, b, out)
if len(CALLS) > 3:
    return a + b
return out
""", extra="\nCALLS = []\n")
    res = run(REF_ADD, cand)
    assert_zero(res, sr.CHEAT, "torch_compute_fallback:aten::add")


def test_4_replays_last_answer_once_verification_is_over():
    """Honest for the 3 verification calls, then returns a copy of its last output with a
    decoy launch: no forbidden op at all, only detectable because timed inputs are unseen."""
    cand = model("""
CALLS.append(1)
if len(CALLS) > 3:
    launch(torch.zeros(8, device=a.device), torch.zeros(8, device=a.device), torch.empty(8, device=a.device))
    return CACHE[0].clone()
out = torch.empty_like(a)
launch(a, b, out)
CACHE[:] = [out]
return out
""", extra="\nCALLS = []\nCACHE = []\n")
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "output_wrong_after_verification")


def test_4_value_keyed_answer_cache():
    """Caches answers keyed on the first input element, to recognise repeated inputs."""
    cand = model("""
key = float(a.flatten()[0])
if key in CACHE:
    launch(torch.zeros(8, device=a.device), torch.zeros(8, device=a.device), torch.empty(8, device=a.device))
    return CACHE[key].clone()
out = torch.empty_like(a)
launch(a, b, out)
CACHE[key] = out
return out
""", extra="\nCACHE = {}\n")
    assert_zero(run(REF_ADD, cand), sr.CHEAT, "torch_compute_fallback:aten::_local_scalar_dense")


# ------------------------------------------------------------------ 5. imports / patches
IMPORT_PATCH = {
    "import_os": ("import os\n" + HONEST_ADD, "forbidden_import"),
    "import_subprocess": ("import subprocess\n" + HONEST_ADD, "forbidden_import"),
    "import_numpy": ("import numpy as np\n" + HONEST_ADD, "forbidden_import"),
    "import_ctypes_inside_function": (model("import ctypes\nreturn a"), "forbidden_import"),
    "cpp_extension": ("from torch.utils.cpp_extension import load_inline\n" + HONEST_ADD,
                      "forbidden_import"),
    "dunder_import": (model("os = __import__('os')\n" + DECOY + "return a"), "forbidden_builtin"),
    "getattr_string_trick": (model(DECOY + "return getattr(torch, 'mat' + 'mul')(a, b)"),
                             "forbidden_builtin"),
    "exec_payload": (model(DECOY + "exec('import os')\nreturn a"), "forbidden_builtin"),
    "open_file": (model(DECOY + "open('/etc/passwd').read()\nreturn a"), "forbidden_builtin"),
    "patch_allclose": ("import torch\ntorch.allclose = lambda *a, **k: True\n" + HONEST_ADD,
                       "monkey_patch"),
    "patch_synchronize": (HONEST_ADD + "\ntorch.cuda.synchronize = lambda *a, **k: None\n",
                          "monkey_patch"),
    "patch_event_elapsed": (HONEST_ADD + "\ntorch.cuda.Event.elapsed_time = lambda s, e: 1e-9\n",
                            "monkey_patch"),
    "module_alias_escape": (model(DECOY + "t = torch\nreturn t.matmul(a, b)"),
                            "module_alias_escape"),
    "dunder_dict_lookup": (model(DECOY + "return torch.__dict__['matmul'](a, b)"), "dunder_access"),
    "cuda_stream": (model(DECOY + "s = torch.cuda.Stream()\nreturn a"), "torch_op_not_allowed"),
    "tensor_subclass": (HONEST_ADD + "\nclass Lazy(torch.Tensor):\n    pass\n", "bad_base_class"),
    "numpy_round_trip": (model(DECOY + "return torch.tensor(a.numpy() + b.numpy())"),
                         "tensor_compute_method"),
    "precision_downgrade": (model("""
out = torch.empty_like(a)
launch(a.half(), b.half(), out)
return out
"""), "tensor_compute_method"),
    "torch_compile": (model(DECOY + "return torch.compile(lambda x, y: x)(a, b)"),
                      "torch_op_not_allowed"),
}  # fmt: skip


@pytest.mark.parametrize("name", IMPORT_PATCH)
def test_5_forbidden_import_or_monkey_patch(name):
    cand, rule = IMPORT_PATCH[name]
    assert_zero(run(REF_ADD, cand), sr.STATIC_REJECT, rule)


# ------------------------------------------------------------------ 6. hangs / crashes
def test_6_infinite_loop_in_forward_is_killed():
    cand = model("""
out = torch.empty_like(a)
launch(a, b, out)
while True:
    pass
""")
    t0 = time.monotonic()
    res = run(REF_ADD, cand)
    assert_zero(res, sr.TIMEOUT)
    assert res.timeout_stage == "verified"
    assert time.monotonic() - t0 < 60


def test_6_infinite_loop_at_import_is_killed():
    res = run(REF_ADD, HONEST_ADD + "\nwhile True:\n    pass\n")
    assert_zero(res, sr.TIMEOUT)
    assert not res.compiled


def test_6_hang_only_while_being_timed_is_killed():
    cand = model("""
CALLS.append(1)
out = torch.empty_like(a)
launch(a, b, out)
while len(CALLS) > 3:
    pass
return out
""", extra="\nCALLS = []\n")
    res = run(REF_ADD, cand)
    assert_zero(res, sr.TIMEOUT)
    assert res.timeout_stage == "timed"


def test_6_exception_in_forward_scores_zero():
    assert_zero(run(REF_ADD, NAME_ERROR), sr.COMPILE_ERROR)


def test_6_system_exit_in_forward_scores_zero():
    cand = model("raise SystemExit(0)")
    assert_zero(run(REF_ADD, cand.replace("raise SystemExit(0)", "launch(a, b, a)\n        raise SystemExit(0)")),
                sr.CRASHED)  # fmt: skip


def test_6_unbounded_recursion_scores_zero():
    cand = PREAMBLE + """
def boom(a, b, out):
    launch(a, b, out)
    return boom(a, b, out)

class ModelNew(nn.Module):
    def forward(self, a, b):
        return boom(a, b, torch.empty_like(a))
"""
    res = run(REF_ADD, cand)
    assert res.status in (sr.COMPILE_ERROR, sr.CRASHED), res.to_dict()
    assert shaped_reward(res) == 0.0


# The static guard would stop these long before execution. They are sent straight to the
# sandbox to prove that containment does not depend on the guard.
SANDBOX_ONLY = {
    "segfault": "import faulthandler\nfaulthandler._sigsegv()\n",
    "hard_exit": "import os\nos._exit(137)\n",
    "abort": "import os\nos.abort()\n",
}


@pytest.mark.parametrize("name", SANDBOX_ONLY)
def test_6_sandbox_contains_process_death(name):
    cand = HONEST_ADD.replace("out = torch.empty_like(a)", SANDBOX_ONLY[name].replace("\n", "\n        ")
                              + "out = torch.empty_like(a)")  # fmt: skip
    assert not sr.check_source(cand).ok  # the guard does reject it ...
    res = sr.run_sandboxed(REF_ADD, cand, seeds=SEEDS, limits=FAST)  # ... but we bypass it
    assert_zero(res, sr.CRASHED)
    assert res.exitcode not in (0, None)
    assert os.getpid() > 0  # the parent is still here


def test_6_broken_reference_is_infra_error_not_candidate_fault():
    res = run("raise RuntimeError('broken benchmark file')", HONEST_ADD)
    assert res.status == sr.INFRA_ERROR and res.attempts == 2  # retried once, then reported
    assert "broken benchmark file" in res.error["message"]
    assert shaped_reward(res) == 0.0


def test_parent_survives_a_burst_of_hostile_submissions():
    """After all of the above in one process, a clean submission still evaluates."""
    for cand in (model("while True:\n    pass"), model(DECOY + "return a + b")):
        run(REF_ADD, cand, limits=sr.Limits(verify_s=3.0))
    assert run(REF_ADD, HONEST_ADD).status == sr.CORRECT

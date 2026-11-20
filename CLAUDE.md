# CLAUDE.md — triton-trace-rl

Binding rules for every session in this repo. If a request conflicts with a red line or a
gate below, stop and say so instead of working around it.

## Language

- Progress updates and `reports/GATE_X_REVIEW.md` review reports: **Chinese**.
- Code, comments, README, technical report: **English**.

## Goal: Triton-only `KernelBench × TRACE × PrimeIntellect/verifiers` (2–3 weeks)

Three deliverables from one codebase:

1. **Environments Hub package** — `environments/triton_kernel_env/`: a multi-turn agentic RL
   environment over `ScalingIntelligence/KernelBench` Level 1 & 2, Triton-only, with
   profiling feedback, built on `PrimeIntellect-ai/verifiers`.
2. **Phase 1 — fair-budget multi-turn profiling evaluation** on a strictly isolated held-out
   test set with the Claude API (`ANTHROPIC_API_KEY` from the environment), comparing
   (a) single-shot / best-of-N, (b) multi-turn with compile + error feedback only,
   (c) multi-turn self-evolution with `torch.profiler` bottleneck analysis.
   Report `compile_rate`, `correct_rate`, `fast_1`, `fast_1.5`, `fast_2`.
3. **Phase 2 — TRACE-style capability-targeted LoRA GRPO on one A100 80GB** — from
   `train_source` failure logs only, Claude clusters failure modes and synthesizes 150–300
   targeted Triton micro-tasks; bf16 LoRA GRPO on an open code model
   (`Qwen/Qwen2.5-Coder-7B-Instruct` or `Qwen3-4B/8B`) vs. a uniform GRPO baseline with the
   same number of steps; verify on held-out test; reproducible technical report.

### verifiers API note (verified 2026-11-20)

The original brief named `vf.MultiTurnEnv`, `vf.Rubric`, `load_environment()` and `vf-eval`.
Those belong to the legacy API. `verifiers` 0.4.1 (PyPI) and `main` ship only
`verifiers.v1`: `Taskset` / `Task` / `TaskData`, `@vf.reward` / `@vf.metric`, custom
`vf.Env.run()` driving `agents.<role>.interaction(task)` for multi-turn, and the CLIs
`vf-validate` / `vf-run`. This repo targets **v1**. Do not reintroduce the legacy names
unless the user explicitly decides to pin an old verifiers version.

## Hardware and runtime rules

- **Local machine** (Windows, CPU-only, no Triton): all coding, unit tests, API
  orchestration, analysis, plotting, git. Local tests use the fake Triton in
  `tests/stubs/`; anything that needs a real GPU is marked `@pytest.mark.gpu`.
- **GPU**: only Google Colab Pro, one NVIDIA A100 80GB, **no Docker** (`dockerd` is
  unavailable). Everything must run natively on Linux + Python 3.11+ + PyTorch + Triton.
- **Colab disconnects**: every long-running script and notebook (`notebooks/01_*`, `02_*`,
  `03_*`) mounts Google Drive in one cell, checkpoints and writes raw JSON logs to
  `/content/drive/MyDrive/triton-trace-rl/` every 10–20 steps, and resumes from there.
- **One GPU, time-sliced**: GRPO alternates phases — generate rollouts in a batch → unload
  / free inference memory → benchmark kernels in fresh subprocesses on the idle GPU →
  reload and update the LoRA. Training and kernel timing never share the GPU at the same
  time (OOM and distorted speedups otherwise).

## Six red lines (enforced in code and tests)

1. **Triton-only.** Generate and evaluate only `@triton.jit` kernels wrapped in
   `ModelNew(nn.Module)`. No raw CUDA C++ / `nvcc`.
2. **Strong process isolation.** All untrusted Triton compilation, correctness checking and
   timing runs in a fresh `multiprocessing` child started with `"spawn"`, with hard timeouts
   (compile + verify 25 s, timing 20 s) and forced reaping, so invalid memory access,
   segfaults, infinite loops and OOM can never take down or contaminate the parent.
3. **Zero tolerance for reward hacking** (any violation ⇒ `reward = 0.0`):
   - **AST static check**: at least one `@triton.jit` function that is actually called on
     the `ModelNew.forward` execution path (no decorative zombie kernels); no PyTorch
     high-level fallback in `ModelNew` or helpers (`torch.nn.functional.*`, `F.*`,
     `torch.matmul/mm/bmm/addmm/einsum/conv*/softmax/layer_norm/group_norm/batch_norm`,
     calling `nn.Linear/Conv*/LayerNorm`, …; only an allow-list of tensor allocation /
     shape operations); no file / network / subprocess I/O; no monkey-patching.
   - **Runtime Triton launch check**: forward must really launch a Triton kernel.
   - **Input isolation + multi-seed check**: reference and candidate each receive their own
     deep `clone()` of the inputs (in-place tampering is detected); at least 3 independent
     random seeds must pass `torch.allclose` with no `NaN/Inf`.
   - **Honest timing**: ≥ 5 warmup runs, CUDA events, `torch.cuda.synchronize()` before
     and after, median of ≥ 20 runs.
4. **Strict data isolation.** Freeze an immutable `data/splits/manifest.json`
   (`train_source` vs `heldout_test`) as early as possible; guard assertions in code; no
   Claude failure diagnosis, TRACE synthesis or RL training script may read `heldout_test`;
   every synthesized task is automatically checked for AST / text similarity against
   `heldout_test`.
5. **Shaped RL reward ≠ reported metrics.** GRPO training reward is piecewise:
   cheat / crash / timeout = 0.0; compiled = 0.2; multi-seed numerically correct = 0.5;
   correct and faster = 0.5 + min(1.5, speedup − 1)
   (continuous at speedup = 1; user decision 2026-11-20). Every evaluation table and report states
   the official KernelBench metrics (`compile_rate`, `correct_rate`, `fast_1`, `fast_1.5`,
   `fast_2`).
6. **No fabricated numbers.** Every number in reports, figures and READMEs is generated by
   a script from raw JSON / WandB logs and recorded with the commit hash. Nothing typed by
   hand, nothing estimated.

## Five mandatory human-in-the-loop gates

After each gate: write a concise Chinese review at `reports/GATE_X_REVIEW.md`, then **stop**
and wait for the user to reply "通过". Never start the next phase or any expensive full
API / GPU run on your own.

- **Gate 1** — confirm current `ScalingIntelligence/KernelBench` and
  `PrimeIntellect-ai/verifiers` interfaces → project skeleton → `ast_guard.py`,
  `sandbox_runner.py`, `profiler_feedback.py`, `reward.py` →
  `tests/test_anti_cheat_redteam.py` (≥ 6 devious cheating / crashing kernels: `F.*` /
  `torch.matmul`, zombie `@triton.jit` fallback, in-place input modification, single-seed
  hard-coding, forbidden import / monkey-patch, infinite loop / crash — all must score
  `0.0` without crashing the parent) → package `environments/triton_kernel_env/` and pass a
  local verifiers load test → `notebooks/01_colab_smoke_and_gate1.ipynb`.
- **Gate 2** — freeze `train_source` vs `heldout_test` → fair-budget Phase 1 multi-turn /
  evolution scripts and `02_colab_phase1_eval.ipynb` → 5-problem pilot on `train_source`
  only, to estimate API cost and wall time.
- **Gate 3** — aggregate Phase 1 results → Claude clusters `train_source` failure logs and
  synthesizes 150–300 verified-runnable TRACE targeted tasks → leakage check → the review
  shows **15 complete synthesized tasks** sampled at random.
- **Gate 4** — single-A100 time-sliced LoRA GRPO (`grpo_single_a100.py` +
  `03_colab_phase2_grpo.ipynb`), **25-step health check only** → the review shows memory /
  time / reward / KL / entropy curves and the **3 highest-reward generated Triton sources
  verbatim** for manual no-cheating confirmation.
- **Gate 5** — full TRACE LoRA GRPO vs uniform GRPO → held-out evaluation → figures and
  tables generated from raw logs → Hub listing preparation for
  `environments/triton_kernel_env/` → `reports/FINAL_TECH_REPORT.md` with an honest failure
  analysis.

## Working conventions

- Use `uv` and the repo `.venv` (Python 3.11). Run tests with `uv run pytest`.
- The environment package is self-contained: guard, sandbox, profiler feedback and reward
  live in `environments/triton_kernel_env/triton_kernel_env/`; scripts import from there.
- Git commits carry no AI attribution lines.
- Results that have not been run on a real GPU are labelled as unverified, never implied.

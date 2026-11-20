# triton-trace-rl

Triton-only `KernelBench × TRACE × verifiers`: a multi-turn kernel-optimisation RL
environment, a fair-budget multi-turn profiling evaluation, and capability-targeted LoRA
GRPO on a single A100.

**Status: Gate 1 (environment, guards, sandbox) — awaiting review.** No evaluation or
training numbers exist yet; none are reported here.

## Layout

```
environments/triton_kernel_env/   verifiers v1 package (guard, sandbox, feedback, reward, taskset)
tests/                            red-team + unit + verifiers tests; stubs/ = fake Triton for CPU
notebooks/01_colab_smoke_and_gate1.ipynb   real-GPU verification of Gate 1
scripts/build_notebook_01.py      regenerates the notebook
reports/                          gate reviews (Chinese)
data/splits/                      frozen train_source / heldout_test manifest (Gate 2)
CLAUDE.md                         project rules: red lines and review gates
```

## Run the tests

```bash
uv venv --python 3.11 && uv pip install torch pytest nbformat verifiers datasets
uv pip install --no-deps -e environments/triton_kernel_env
uv run pytest -q
```

Without CUDA + Triton the suite uses the fake Triton in `tests/stubs` and skips
`gpu`-marked tests; on Colab the same suite runs against the real stack (notebook 01).
verifiers v1 is POSIX-only, so `tests/test_verifiers_env.py` is skipped on native Windows
(use WSL).

See `environments/triton_kernel_env/README.md` for the environment itself.

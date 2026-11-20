# triton-kernel-env

Triton-only [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Level 1–2) as
a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 environment: the agent
rewrites a PyTorch reference `Model` as `ModelNew` built on `@triton.jit` kernels, receives
compile / correctness / timing / profiler feedback, and resubmits.

## Use

```bash
uv pip install -e environments/triton_kernel_env
uv run vf-validate triton-kernel-env --taskset.source smoke --runtime.type subprocess -n 2
uv run vf-run triton-kernel-env -n 5 -r 1 --env.max-turns 4 --env.feedback profile
```

| Option | Values | Meaning |
| --- | --- | --- |
| `--env.taskset.source` | `kernelbench` (default), `smoke` | benchmark, or two built-in tasks with gold solutions |
| `--env.taskset.levels` | `[1, 2]` | KernelBench levels to load |
| `--env.max-turns` | `4` | submissions per episode (`1` = single-shot) |
| `--env.feedback` | `errors`, `timing`, `profile` | information returned between turns |
| `--env.taskset.task.measure-performance` | `true` | time correct kernels against the reference |

Kernels are compiled and timed **on the evaluator machine**, which therefore needs an
NVIDIA GPU, PyTorch and Triton. No Docker is involved. Evaluations are serialised on the
GPU so that timings do not disturb each other.

## Scoring

- **Reward** (training signal): cheat / crash / timeout `0.0`; compiled `0.2`; correct on
  all hidden seeds `0.5`; correct and faster than the reference `0.5 + min(1.5, speedup - 1)`.
  A multi-turn trace is rewarded for its best turn.
- **Metrics** (what to report): `compiled`, `correct`, `speedup` per trace;
  `reward.kernelbench_metrics` aggregates them into `compile_rate`, `correct_rate`,
  `fast_1`, `fast_1.5`, `fast_2`.

## Anti-reward-hacking

| Layer | What it enforces |
| --- | --- |
| `ast_guard.py` (static) | import allow-list; `torch.*` allow-list (allocation / shape only); no `F.*`, tensor math methods, `@`, called `nn` layers; no `eval/exec/getattr`, dunder access, monkey-patching, tensor subclasses; a `kernel[grid](...)` launch must be reachable from `ModelNew.forward` |
| `sandbox_runner.py` (isolation) | every submission runs in a fresh `spawn`ed process with hard per-stage deadlines (verify 25 s, timing 20 s) and forced reaping |
| runtime guard (in the child) | a dispatcher hook rejects any aten op outside the allocation / shape allow-list — this catches bare operators such as `a + b`; counts real Triton launches; rejects called `torch.nn` layers; inputs are cloned and compared afterwards; ≥ 3 hidden random seeds; output must depend on the input; a strict re-check runs after timing |
| timing | ≥ 5 warmups (randomised count), CUDA events, `synchronize` before and after, median of ≥ 20 trials, fresh input clones per trial |

`nn.*` layers may be constructed in `__init__` as parameter containers (KernelBench relies
on same-seed construction for weight parity); invoking them is rejected.

"""triton-kernel-env: rewrite KernelBench PyTorch references as Triton kernels.

Each task shows a KernelBench reference `Model`; the agent replies with a `ModelNew`
built on `@triton.jit` kernels. Submissions pass a static AST guard, then run in a
hard-isolated subprocess that checks multi-seed correctness, watches for PyTorch
fallbacks at runtime and times the kernel against the reference.

`TritonKernelEnv` makes it multi-turn: after each submission the agent receives
feedback (`--env.feedback errors|timing|profile`) and may resubmit, up to
`--env.max-turns`. The trace's reward is its best turn's shaped reward; the official
KernelBench signals (`compiled`, `correct`, `speedup`) are recorded as metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import threading
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from triton_kernel_env import sandbox_runner as sr
from triton_kernel_env.profiler_feedback import build_feedback
from triton_kernel_env.prompts import SMOKE_TASKS, SYSTEM_PROMPT, build_prompt, extract_code
from triton_kernel_env.reward import shaped_reward

KERNELBENCH_DATASET = "ScalingIntelligence/KernelBench"
KERNELBENCH_REVISION = "ca1464e5d4723fd14f87b8e68306aaa1fe66b81b"

_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def gpu_slot():
    """One evaluation on the GPU at a time, across threads and (on POSIX) processes:
    concurrent kernels would distort each other's timings."""
    with _THREAD_LOCK:
        try:
            import fcntl
        except ImportError:  # Windows: thread lock only
            yield
            return
        path = os.path.join(tempfile.gettempdir(), "triton_kernel_env.gpu.lock")
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def evaluate_submission(ref_src: str, reply: str, **kwargs) -> dict:
    """Evaluate one model reply; returns the JSON-serialisable result plus its reward."""
    with gpu_slot():
        result = sr.evaluate_kernel(ref_src, extract_code(reply), **kwargs).to_dict()
    result["reward"] = shaped_reward(result)
    return result


class TritonKernelData(vf.TaskData):
    ref_src: str
    """KernelBench-format reference: `Model`, `get_inputs()`, `get_init_inputs()`."""
    level: int = 0
    gold: str | None = None
    """A known-good Triton solution, when one exists (smoke tasks only)."""


class TritonKernelTaskConfig(vf.TaskConfig):
    measure_performance: bool = True
    """Time correct submissions against the reference (needs an idle GPU)."""


class TritonKernelTask(vf.Task[TritonKernelData, vf.State, TritonKernelTaskConfig]):
    def _evaluate(self, reply: str, *, profile: bool = False) -> dict:
        return evaluate_submission(
            self.data.ref_src,
            reply,
            measure_performance=self.config.measure_performance,
            profile=profile,
        )

    @vf.metric
    async def evaluate(self, trace: vf.Trace) -> dict[str, float]:
        """Scores the best turn (multi-turn env) or the final reply (single agent)."""
        turns = trace.info.get("turns")
        if not turns:
            turns = [await asyncio.to_thread(self._evaluate, trace.last_reply)]
            trace.info["turns"] = turns
        best = max(turns, key=lambda t: t["reward"])
        trace.info["best_turn"] = turns.index(best)
        return {
            "compiled": float(best["compiled"]),
            "correct": float(best["correct"]),
            "speedup": float(best["speedup"] or 0.0),
            "cheat": float(any(t["status"] in (sr.STATIC_REJECT, sr.CHEAT) for t in turns)),
            "num_turns": float(len(turns)),
            "final_turn_reward": float(turns[-1]["reward"]),
        }

    @vf.reward
    async def kernel_reward(self, trace: vf.Trace) -> float:
        turns = trace.info["turns"]
        return float(turns[trace.info["best_turn"]]["reward"])

    async def validate(self, runtime: vf.Runtime) -> bool | None:
        if self.data.gold is None:
            return None
        result = await asyncio.to_thread(self._evaluate, f"```python\n{self.data.gold}\n```")
        return bool(result["correct"])


class TritonKernelConfig(vf.TasksetConfig):
    source: Literal["kernelbench", "smoke"] = "kernelbench"
    """`kernelbench` downloads the benchmark; `smoke` uses two built-in tasks."""
    levels: list[int] = Field(default_factory=lambda: [1, 2])
    revision: str = KERNELBENCH_REVISION
    task: TritonKernelTaskConfig = TritonKernelTaskConfig()


class TritonKernelTaskset(vf.Taskset[TritonKernelTask, TritonKernelConfig]):
    def _task(self, name: str, ref_src: str, level: int, gold: str | None) -> TritonKernelTask:
        data = TritonKernelData(
            id=name,
            name=name,
            system_prompt=SYSTEM_PROMPT,
            prompt=build_prompt(ref_src),
            ref_src=ref_src,
            level=level,
            gold=gold,
        )
        return TritonKernelTask(data, self.config.task)

    def load(self) -> list[TritonKernelTask]:
        if self.config.source == "smoke":
            return [self._task(name, ref, 0, gold) for name, ref, gold in SMOKE_TASKS]
        from datasets import load_dataset

        tasks = []
        for level in self.config.levels:
            rows = load_dataset(
                KERNELBENCH_DATASET, split=f"level_{level}", revision=self.config.revision
            )
            for row in sorted(rows, key=lambda r: r["problem_id"]):
                tasks.append(self._task(f"level{level}/{row['name']}", row["code"], level, None))
        return tasks


class TritonKernelEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    max_turns: int = Field(4, ge=1)
    """Submissions per episode (1 = single-shot)."""
    feedback: Literal["errors", "timing", "profile"] = "profile"
    """Information returned between turns; see `profiler_feedback.build_feedback`."""


class TritonKernelEnv(vf.Env[TritonKernelEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        mode, max_turns = self.config.feedback, self.config.max_turns
        async with agents.solver.interaction(task) as interaction:
            turns: list[dict] = []
            interaction.trace.info["turns"] = turns
            segment = await interaction.turn()
            while not segment.terminated:
                result = await asyncio.to_thread(
                    task._evaluate, segment.last_reply, profile=mode == "profile"
                )
                turns.append(result)
                if len(turns) >= max_turns:
                    break
                segment = await interaction.turn(build_feedback(result, mode))

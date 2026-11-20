"""The package as verifiers sees it: discovery by id, task loading, the multi-turn env.

verifiers v1 is POSIX-only (it imports `fcntl`), so this module is skipped on native
Windows and runs under WSL / Linux / Colab.
"""

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

try:
    import verifiers.v1 as vf
    from verifiers.v1.utils import loaders
except ImportError as e:  # pragma: no cover - platform dependent
    pytest.skip(f"verifiers v1 not importable here: {e}", allow_module_level=True)

from kernels import HONEST_ADD, REF_ADD, WRONG_SUB, model
from triton_kernel_env import sandbox_runner as sr
from triton_kernel_env.taskset import (
    TritonKernelConfig, TritonKernelData, TritonKernelEnv, TritonKernelEnvConfig,
    TritonKernelTask, TritonKernelTaskset,
)  # fmt: skip

ENV_ID = "triton-kernel-env"


def fenced(src: str) -> str:
    return f"Here is my kernel.\n```python\n{src}\n```"


def make_task(gold=None) -> TritonKernelTask:
    data = TritonKernelData(id="t", name="t", prompt="p", ref_src=REF_ADD, gold=gold)
    return TritonKernelTask(data)


def test_verifiers_discovers_taskset_and_env_by_id():
    assert loaders.taskset_class(ENV_ID) is TritonKernelTaskset
    assert loaders.environment_class(ENV_ID, None) is TritonKernelEnv


def test_smoke_taskset_loads_through_verifiers():
    taskset = loaders.load_taskset(TritonKernelConfig(id=ENV_ID, source="smoke"))
    tasks = list(taskset.load())
    assert [t.data.name for t in tasks] == ["smoke_add", "smoke_relu"]
    for t in tasks:
        assert "Triton only" in t.data.system_prompt
        assert "class Model" in t.data.prompt_text and "get_inputs" in t.data.prompt_text
        assert t.data.gold and "@triton.jit" in t.data.gold


class FakeInteraction:
    """Stands in for `agents.solver.interaction(task)`: scripted replies, real trace dict."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[str | None] = []
        self.trace = SimpleNamespace(info={}, last_reply="")

    async def turn(self, message=None):
        self.sent.append(message)
        if not self.replies:
            return SimpleNamespace(terminated=True, last_reply="")
        self.trace.last_reply = self.replies.pop(0)
        return SimpleNamespace(terminated=False, last_reply=self.trace.last_reply)


def run_env(replies, **config):
    env = loaders.load_environment(
        TritonKernelEnvConfig(taskset={"id": ENV_ID, "source": "smoke"}, **config)
    )
    assert isinstance(env, TritonKernelEnv)
    interaction = FakeInteraction(replies)

    @contextlib.asynccontextmanager
    async def open_interaction(task):
        yield interaction

    agents = SimpleNamespace(solver=SimpleNamespace(interaction=open_interaction))
    task = make_task()

    async def go():
        await env.run(task, agents)
        metrics = await task.evaluate(interaction.trace)
        reward = await task.kernel_reward(interaction.trace)
        return metrics, reward

    metrics, reward = asyncio.run(go())
    return interaction, metrics, reward


def test_multi_turn_env_feeds_back_and_rewards_the_best_turn():
    cheat = model("return a + b")  # no launch reachable -> static reject
    interaction, metrics, reward = run_env(
        [fenced(cheat), fenced(WRONG_SUB), fenced(HONEST_ADD), fenced(cheat)],
        max_turns=3, feedback="timing",
    )  # fmt: skip
    turns = interaction.trace.info["turns"]
    assert [t["status"] for t in turns] == [sr.STATIC_REJECT, sr.INCORRECT, sr.CORRECT]
    assert [t["reward"] for t in turns][:2] == [0.0, 0.2]
    # opening turn has no message; then one feedback message per non-final turn
    assert interaction.sent[0] is None and len(interaction.sent) == 3
    assert "REJECTED" in interaction.sent[1] and "WRONG" in interaction.sent[2]
    assert reward == turns[2]["reward"] >= 0.5
    assert metrics["correct"] == 1.0 and metrics["cheat"] == 1.0 and metrics["num_turns"] == 3.0


def test_single_shot_when_max_turns_is_one():
    interaction, metrics, reward = run_env([fenced(WRONG_SUB), fenced(HONEST_ADD)], max_turns=1)
    assert len(interaction.trace.info["turns"]) == 1 and interaction.sent == [None]
    assert reward == 0.2 and metrics["compiled"] == 1.0 and metrics["correct"] == 0.0


def test_task_scores_last_reply_without_the_multi_turn_env():
    """Under the default SingleAgentEnv there is no `turns` record yet."""
    task = make_task()
    trace = SimpleNamespace(info={}, last_reply=fenced(HONEST_ADD))
    metrics = asyncio.run(task.evaluate(trace))
    assert metrics["correct"] == 1.0 and asyncio.run(task.kernel_reward(trace)) >= 0.5
    empty = SimpleNamespace(info={}, last_reply="")
    assert asyncio.run(task.evaluate(empty))["compiled"] == 0.0
    assert asyncio.run(task.kernel_reward(empty)) == 0.0


def test_validate_uses_the_gold_solution():
    assert asyncio.run(make_task().validate(None)) is None
    assert asyncio.run(make_task(gold=HONEST_ADD).validate(None)) is True
    assert asyncio.run(make_task(gold=WRONG_SUB).validate(None)) is False


@pytest.mark.gpu
def test_smoke_gold_solutions_are_correct_on_real_triton():
    taskset = loaders.load_taskset(TritonKernelConfig(id=ENV_ID, source="smoke"))
    for task in taskset.load():
        assert asyncio.run(task.validate(None)) is True, task.data.name

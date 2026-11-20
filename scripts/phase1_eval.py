"""Phase 1: fair-budget comparison of three ways to spend N kernel submissions per problem.

    best_of_n          N independent single-shot attempts (turn 0 alone = single-shot)
    multiturn_errors   one conversation, N submissions, compile / correctness feedback only
    multiturn_profile  one conversation, N submissions, + timing and torch.profiler feedback

Every arm gets exactly N model calls and N evaluations per problem, the same system prompt
and the same task prompt. One raw JSON record per (arm, problem) is written to `--out`;
finished records are skipped on restart, so a Colab disconnect loses at most the episodes
in flight. Aggregate with `scripts/phase1_report.py` — this script prints no metrics.

Needs a GPU with Triton (kernel evaluation) and `ANTHROPIC_API_KEY` in the environment.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "environments" / "triton_kernel_env"))

from triton_kernel_env import sandbox_runner as sr  # noqa: E402
from triton_kernel_env import splits  # noqa: E402
from triton_kernel_env.profiler_feedback import build_feedback  # noqa: E402
from triton_kernel_env.prompts import SYSTEM_PROMPT, build_prompt, extract_code  # noqa: E402
from triton_kernel_env.reward import shaped_reward  # noqa: E402

ARMS = {"best_of_n": None, "multiturn_errors": "errors", "multiturn_profile": "profile"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_GPU_LOCK = threading.Lock()  # one kernel on the GPU at a time, or timings distort


def make_call_model(model: str, effort: str, max_tokens: int):
    """Returns call(messages) -> dict(text, content, usage, stop_reason, model)."""
    import anthropic

    client = anthropic.Anthropic(max_retries=5)

    def call(messages: list[dict]) -> dict:
        with client.beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        usage = msg.usage
        return {
            "text": "".join(b.text for b in msg.content if b.type == "text"),
            "content": msg.content,  # replayed unchanged (thinking blocks included)
            "stop_reason": msg.stop_reason,
            "model": msg.model,
            "usage": {
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            },
        }

    return call


def evaluate(ref_src: str, reply: str, *, profile: bool) -> dict:
    with _GPU_LOCK:
        result = sr.evaluate_kernel(ref_src, extract_code(reply), profile=profile).to_dict()
    result["reward"] = shaped_reward(result)
    return result


def run_episode(problem: dict, arm: str, budget: int, call_model, evaluate_fn=evaluate) -> dict:
    """N submissions for one problem under one arm. Returns the raw record."""
    feedback_mode = ARMS[arm]
    prompt = build_prompt(problem["ref_src"])
    messages = [{"role": "user", "content": prompt}]
    turns = []
    for i in range(budget):
        if feedback_mode is None:  # independent attempts: fresh conversation each time
            messages = [{"role": "user", "content": prompt}]
        t0 = time.monotonic()
        reply = call_model(messages)
        t1 = time.monotonic()
        result = evaluate_fn(problem["ref_src"], reply["text"], profile=feedback_mode == "profile")
        t2 = time.monotonic()
        feedback = build_feedback(result, feedback_mode) if feedback_mode else None
        turns.append({
            "turn": i, "reply": reply["text"], "stop_reason": reply["stop_reason"],
            "model": reply["model"], "usage": reply["usage"], "result": result,
            "feedback": feedback, "api_s": t1 - t0, "eval_s": t2 - t1,
        })  # fmt: skip
        if feedback_mode:
            messages.append({"role": "assistant", "content": reply["content"]})
            messages.append({"role": "user", "content": feedback})
    return {"problem_id": problem["id"], "level": problem["level"], "arm": arm,
            "budget": budget, "turns": turns, "complete": True}  # fmt: skip


def record_path(out: Path, arm: str, problem_id: str) -> Path:
    return out / arm / (problem_id.replace("/", "__") + ".json")


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default=splits.TRAIN, choices=[splits.TRAIN, splits.HELDOUT])
    ap.add_argument("--purpose", default="pilot",
                    help=f"must be {splits.HELDOUT_PURPOSE!r} to read {splits.HELDOUT}")  # fmt: skip
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--budget", type=int, default=4, help="submissions per problem per arm")
    ap.add_argument("--n-problems", type=int, help="random subset, stratified by level")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=4, help="concurrent API conversations")
    args = ap.parse_args()

    problems = splits.load_problems(args.split, purpose=args.purpose)
    if args.n_problems:
        rng = random.Random(args.subset_seed)
        levels = sorted({p["level"] for p in problems})
        per = [args.n_problems // len(levels) + (i < args.n_problems % len(levels))
               for i in range(len(levels))]  # fmt: skip
        problems = [p for lv, k in zip(levels, per, strict=True)
                    for p in rng.sample([q for q in problems if q["level"] == lv], k)]  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "commit": git_commit(), "problem_ids": [p["id"] for p in problems]}  # fmt: skip
    (args.out / "run_config.json").write_text(json.dumps(run_meta, indent=1), encoding="utf-8")

    call_model = make_call_model(args.model, args.effort, args.max_tokens)
    jobs = [(p, arm) for p in problems for arm in args.arms
            if not record_path(args.out, arm, p["id"]).exists()]  # fmt: skip
    print(f"{len(problems)} problems x {len(args.arms)} arms: {len(jobs)} episodes to run", flush=True)

    def work(problem: dict, arm: str) -> str:
        t0 = time.time()
        try:
            record = run_episode(problem, arm, args.budget, call_model)
        except Exception as e:  # noqa: BLE001 - leave no record so a restart retries it
            return f"FAILED {arm} {problem['id']}: {type(e).__name__}: {e}"
        record.update(commit=run_meta["commit"], split=args.split, effort=args.effort,
                      requested_model=args.model, started_at=t0, wall_s=time.time() - t0)  # fmt: skip
        path = record_path(args.out, arm, problem["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
        best = max(t["result"]["reward"] for t in record["turns"])
        return f"done {arm} {problem['id']} best_reward={best:.3f} wall={record['wall_s']:.0f}s"

    # Problems run in parallel, a problem's arms one after another: its arms open with the
    # same prompt, and identical concurrent conversations collide on some API gateways.
    def work_problem(problem: dict) -> None:
        for p, arm in jobs:
            if p is problem:
                print(work(p, arm), flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(work_problem, p) for p in problems]):
            fut.result()
    print("PHASE1_EVAL_FINISHED", flush=True)


if __name__ == "__main__":
    main()

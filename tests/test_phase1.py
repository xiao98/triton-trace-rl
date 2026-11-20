"""Split guard (red line 4) and the Phase 1 evaluation / report scripts, with a fake model."""

import sys
from pathlib import Path

import pytest

from triton_kernel_env import sandbox_runner as sr
from triton_kernel_env import splits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import phase1_eval  # noqa: E402
import phase1_report  # noqa: E402


# ------------------------------------------------------------------ frozen split
def test_manifest_is_a_disjoint_stratified_partition():
    m = splits.load_manifest()
    train, heldout = m["splits"][splits.TRAIN], m["splits"][splits.HELDOUT]
    assert len(train) == 120 and len(heldout) == 80
    assert not set(train) & set(heldout)
    assert set(train) | set(heldout) == set(m["sha256"])
    for level in ("level1/", "level2/"):
        assert sum(p.startswith(level) for p in heldout) == 40


@pytest.mark.parametrize("purpose", ["pilot", "failure_diagnosis", "trace_synthesis", "rl_training", ""])
def test_heldout_is_refused_for_every_purpose_but_final_eval(purpose):
    with pytest.raises(splits.HeldoutAccessError):
        splits.problem_ids(splits.HELDOUT, purpose=purpose)
    with pytest.raises(splits.HeldoutAccessError):
        splits.load_problems(splits.HELDOUT, purpose=purpose, rows={})
    assert len(splits.problem_ids(splits.TRAIN, purpose=purpose)) == 120


def test_heldout_readable_only_with_the_eval_purpose():
    assert len(splits.problem_ids(splits.HELDOUT, purpose=splits.HELDOUT_PURPOSE)) == 80


def test_reference_drift_from_the_manifest_is_detected():
    m = {"splits": {splits.TRAIN: ["level1/a"], splits.HELDOUT: []},
         "sha256": {"level1/a": splits.sha256("original")}}  # fmt: skip
    ok = splits.load_problems(splits.TRAIN, purpose="pilot", manifest=m,
                              rows={"level1/a": {"level": 1, "ref_src": "original"}})  # fmt: skip
    assert ok == [{"id": "level1/a", "level": 1, "ref_src": "original"}]
    with pytest.raises(RuntimeError, match="differs from the frozen manifest"):
        splits.load_problems(splits.TRAIN, purpose="pilot", manifest=m,
                             rows={"level1/a": {"level": 1, "ref_src": "edited"}})  # fmt: skip


# ------------------------------------------------------------------ episode loop
PROBLEM = {"id": "level1/x", "level": 1, "ref_src": "REF"}
USAGE = {"input_tokens": 1000, "output_tokens": 500,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}  # fmt: skip


def fake_model(seen):
    def call(messages):
        seen.append([m["role"] for m in messages])
        return {"text": f"reply{len(seen)}", "content": [{"type": "text", "text": "x"}],
                "stop_reason": "end_turn", "model": "claude-opus-5-5", "usage": dict(USAGE)}  # fmt: skip
    return call


def fake_eval(results, profile_flags):
    it = iter(results)

    def ev(ref_src, reply, *, profile):
        profile_flags.append(profile)
        r = dict(next(it))
        r["reward"] = phase1_eval.shaped_reward(r)
        return r
    return ev


def _res(status, compiled=False, correct=False, speedup=None):
    return {"status": status, "compiled": compiled, "correct": correct, "speedup": speedup,
            "cheats": [], "violations": [], "error": {"type": "E", "message": "m"},
            "mismatch": "", "ref_ms": 1.0, "cand_ms": 1.0, "profile": None}  # fmt: skip


RESULTS = [_res(sr.COMPILE_ERROR), _res(sr.INCORRECT, True), _res(sr.CORRECT, True, True, 1.2)]


def test_best_of_n_uses_a_fresh_conversation_every_attempt():
    seen, flags = [], []
    rec = phase1_eval.run_episode(PROBLEM, "best_of_n", 3, fake_model(seen), fake_eval(RESULTS, flags))
    assert seen == [["user"]] * 3 and flags == [False] * 3
    assert [t["feedback"] for t in rec["turns"]] == [None] * 3


def test_multiturn_arms_feed_results_back_and_only_profile_arm_profiles():
    for arm, want_profile in (("multiturn_errors", False), ("multiturn_profile", True)):
        seen, flags = [], []
        rec = phase1_eval.run_episode(PROBLEM, arm, 3, fake_model(seen), fake_eval(RESULTS, flags))
        assert seen == [["user"], ["user", "assistant", "user"],
                        ["user", "assistant", "user", "assistant", "user"]]  # fmt: skip
        assert flags == [want_profile] * 3 and len(rec["turns"]) == 3
        last = rec["turns"][-1]["feedback"]
        assert "CORRECT" in last and ("speedup" in last) == want_profile


# ------------------------------------------------------------------ report
def test_report_metrics_cost_and_projection():
    recs = []
    for arm in ("best_of_n", "multiturn_profile"):
        for pid, results in (("level1/a", RESULTS), ("level1/b", [_res(sr.COMPILE_ERROR)] * 3)):
            rec = phase1_eval.run_episode({**PROBLEM, "id": pid}, arm, 3, fake_model([]),
                                          fake_eval(results, []))  # fmt: skip
            rec.update(commit="c0ffee", split=splits.TRAIN, effort="medium",
                       requested_model="claude-opus-5-5", wall_s=30.0)  # fmt: skip
            recs.append(rec)
    s = phase1_report.aggregate(recs)
    a = s["arms"]["best_of_n"]
    assert a["metrics_at_k"]["1"]["compile_rate"] == 0.0
    assert a["metrics_at_k"]["3"] == {"n": 2, "compile_rate": 0.5, "correct_rate": 0.5,
                                      "fast_1": 0.5, "fast_1.5": 0.0, "fast_2": 0.0}  # fmt: skip
    # 6 calls x (1000 in x $4 + 500 out x $20) / 1e6
    assert a["cost_usd"] == pytest.approx(6 * (1000 * 4 + 500 * 20) / 1e6)
    proj = phase1_report.project(s, 120)["best_of_n"]
    assert proj["cost_usd"] == pytest.approx(a["cost_usd"] / 2 * 120)
    assert proj["sequential_wall_h"] == pytest.approx(30.0 * 120 / 3600)
    s["report_commit"] = "abc"
    assert "best_of_n | 3 | 2 | 0.500" in phase1_report.to_markdown(s, {"120 problems": phase1_report.project(s, 120)})


def test_unpriced_model_yields_no_cost_instead_of_a_guess():
    turn = {"model": "some-fallback-model", "usage": dict(USAGE)}
    assert phase1_report.turn_cost(turn) is None

"""Tests for judges — including a mocked LLM-as-judge (no network)."""

import pytest

from replayagent.judges import (
    ContainsJudge,
    ExactMatchJudge,
    LLMJudge,
    SemanticSimilarityJudge,
    ToolCallJudge,
    get_judge,
)
from replayagent.trajectory import Trajectory
from tests.conftest import make_trajectory


def test_exact_match_pass():
    r = ExactMatchJudge().evaluate(make_trajectory(), make_trajectory(run_id="b"))
    assert r.passed and r.score == 1.0


def test_exact_match_fail():
    r = ExactMatchJudge().evaluate(
        make_trajectory(), make_trajectory(run_id="b", answer="different")
    )
    assert not r.passed and r.score == 0.0


def test_exact_match_missing_answer():
    empty = Trajectory(run_id="e", agent_name="a", model="m")
    r = ExactMatchJudge().evaluate(make_trajectory(), empty)
    assert not r.passed


def test_contains_plain():
    r = ContainsJudge("password").evaluate(make_trajectory(), make_trajectory(run_id="b"))
    assert r.passed


def test_contains_regex():
    r = ContainsJudge(r"pass\w+", regex=True).evaluate(
        make_trajectory(), make_trajectory(run_id="b")
    )
    assert r.passed
    r2 = ContainsJudge(r"^zzz", regex=True).evaluate(make_trajectory(), make_trajectory(run_id="b"))
    assert not r2.passed


def test_similarity_identical():
    r = SemanticSimilarityJudge().evaluate(make_trajectory(), make_trajectory(run_id="b"))
    assert r.passed and r.score == 1.0


def test_similarity_dissimilar():
    r = SemanticSimilarityJudge(threshold=0.9).evaluate(
        make_trajectory(answer="the cat sat on the mat"),
        make_trajectory(run_id="b", answer="quantum entanglement enables teleportation"),
    )
    assert not r.passed and r.score < 0.9


def test_similarity_bad_threshold():
    with pytest.raises(ValueError):
        SemanticSimilarityJudge(threshold=1.5)


def test_tool_call_judge_strict():
    base = make_trajectory(tools=("a", "b"))
    same = make_trajectory(run_id="b", tools=("a", "b"))
    diff = make_trajectory(run_id="c", tools=("b", "a"))
    assert ToolCallJudge().evaluate(base, same).passed
    assert not ToolCallJudge().evaluate(base, diff).passed
    assert ToolCallJudge(strict_order=False).evaluate(base, diff).passed


def _traj_with_calls(run_id: str, calls: list[tuple[str, dict]]):
    from replayagent import Recorder

    rec = Recorder("demo-agent", model="gpt-4o", input={}, run_id=run_id, autosave=False)
    for name, args in calls:
        rec.tool_call(name=name, arguments=args)
    rec.final_answer("done")
    return rec.trajectory


def test_tool_call_judge_subset_mode():
    base = make_trajectory(tools=("a", "b"))
    fewer = make_trajectory(run_id="b", tools=("a",))
    extra = make_trajectory(run_id="c", tools=("a", "b", "c"))
    judge = ToolCallJudge(mode="subset")
    # No extras beyond the baseline is a pass, even if a call is missing.
    assert judge.evaluate(base, fewer).passed
    assert judge.evaluate(base, base).passed
    # An extra tool the baseline never called is a failure.
    assert not judge.evaluate(base, extra).passed
    assert judge.evaluate(base, fewer).details["mode"] == "subset"


def test_tool_call_judge_superset_mode():
    base = make_trajectory(tools=("a", "b"))
    extra = make_trajectory(run_id="b", tools=("a", "b", "c"))
    fewer = make_trajectory(run_id="c", tools=("a",))
    judge = ToolCallJudge(mode="superset")
    # At least the baseline tools is a pass, extras allowed.
    assert judge.evaluate(base, extra).passed
    assert judge.evaluate(base, base).passed
    # A missing required tool is a failure.
    assert not judge.evaluate(base, fewer).passed


def test_tool_call_judge_subset_superset_respect_multiplicities():
    # Duplicate calls count: calling "a" twice is not a subset of calling it once.
    base = _traj_with_calls("base", [("a", {})])
    twice = _traj_with_calls("cand", [("a", {}), ("a", {})])
    assert not ToolCallJudge(mode="subset").evaluate(base, twice).passed
    assert ToolCallJudge(mode="superset").evaluate(base, twice).passed


def test_tool_call_judge_invalid_mode():
    with pytest.raises(ValueError):
        ToolCallJudge(mode="fuzzy")


def test_tool_call_judge_check_arguments():
    base = _traj_with_calls("base", [("reset_password", {"user": "amy"})])
    same_args = _traj_with_calls("b", [("reset_password", {"user": "amy"})])
    wrong_args = _traj_with_calls("c", [("reset_password", {"user": "bob"})])
    # Names-only (default, backwards compatible): wrong arguments still pass.
    assert ToolCallJudge().evaluate(base, wrong_args).passed
    # With argument checking: same name + wrong arguments fails.
    judge = ToolCallJudge(check_arguments=True)
    assert judge.evaluate(base, same_args).passed
    assert not judge.evaluate(base, wrong_args).passed


def test_tool_call_judge_check_arguments_unordered_and_superset():
    base = _traj_with_calls("base", [("a", {"x": 1}), ("b", {"y": 2})])
    reordered = _traj_with_calls("b", [("b", {"y": 2}), ("a", {"x": 1})])
    reordered_wrong = _traj_with_calls("c", [("b", {"y": 99}), ("a", {"x": 1})])
    unordered = ToolCallJudge(mode="unordered", check_arguments=True)
    assert unordered.evaluate(base, reordered).passed
    assert not unordered.evaluate(base, reordered_wrong).passed
    superset = ToolCallJudge(mode="superset", check_arguments=True)
    extra = _traj_with_calls("d", [("a", {"x": 1}), ("b", {"y": 2}), ("c", {})])
    assert superset.evaluate(base, extra).passed
    assert not superset.evaluate(base, reordered_wrong).passed


def test_tool_call_judge_via_registry():
    assert isinstance(get_judge("tools", mode="superset"), ToolCallJudge)
    assert get_judge("tools", mode="superset").mode == "superset"
    assert get_judge("tools").mode == "strict"
    assert get_judge("tools", strict_order=False).mode == "unordered"


def test_get_judge_registry():
    assert isinstance(get_judge("exact"), ExactMatchJudge)
    assert isinstance(get_judge("similarity", threshold=0.5), SemanticSimilarityJudge)
    with pytest.raises(ValueError):
        get_judge("nope")


def _fake_openai_verdict(line: str):
    """Build a fake openai module whose chat.completions returns `line`."""

    class _Msg:
        content = line

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    import types

    mod = types.ModuleType("openai")
    mod.OpenAI = lambda: _Client()
    return mod


def test_llm_judge_mocked_pass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        __import__("sys").modules, "openai", _fake_openai_verdict("PASS|0.95|equivalent answer")
    )
    r = LLMJudge().evaluate(make_trajectory(), make_trajectory(run_id="b"))
    assert r.passed and r.score == pytest.approx(0.95)
    assert r.details["provider"] == "openai"


def test_llm_judge_mocked_fail(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        __import__("sys").modules, "openai", _fake_openai_verdict("FAIL|0.2|wrong tool used")
    )
    r = LLMJudge().evaluate(make_trajectory(), make_trajectory(run_id="b"))
    assert not r.passed and r.score == pytest.approx(0.2)


def test_llm_judge_no_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMJudge().evaluate(make_trajectory(), make_trajectory(run_id="b"))


def test_llm_judge_bad_provider():
    with pytest.raises(ValueError):
        LLMJudge(provider="cohere")

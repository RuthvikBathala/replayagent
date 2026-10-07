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

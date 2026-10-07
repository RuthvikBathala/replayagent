"""Tests for adapters — with fake framework objects, no real SDKs needed."""

from types import SimpleNamespace

import pytest

from replayagent import Recorder
from replayagent.adapters.openai import RecordedOpenAIClient
from replayagent.trajectory import StepType


class _FakeCompletions:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        self.seen_kwargs = kwargs
        return self._response


def _fake_chat_response(content="hello", tool_calls=()):
    fn_calls = []
    for name, args in tool_calls:
        fn_calls.append(SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)))
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=list(fn_calls)),
        finish_reason="stop",
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o-mini")


def test_openai_adapter_records_llm_call():
    rec = Recorder("a", autosave=False)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(_fake_chat_response()))
    )
    client = RecordedOpenAIClient(fake_client, rec)
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.model == "gpt-4o-mini"  # passthrough works
    steps = list(rec)
    assert len(steps) == 1
    assert steps[0].type == StepType.LLM_CALL
    assert steps[0].payload["response"] == "hello"
    assert steps[0].payload["usage"]["total_tokens"] == 15


def test_openai_adapter_records_tool_calls():
    rec = Recorder("a", autosave=False)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=_FakeCompletions(_fake_chat_response(tool_calls=[("search", '{"q":"x"}')]))
        )
    )
    client = RecordedOpenAIClient(fake_client, rec)
    client.chat.completions.create(model="m", messages=[])
    kinds = [s.type for s in rec]
    assert StepType.TOOL_CALL in kinds
    assert StepType.LLM_CALL in kinds
    tool_step = next(s for s in rec if s.type == StepType.TOOL_CALL)
    assert tool_step.payload["name"] == "search"


def test_openai_adapter_never_breaks_call():
    class ExplodingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    rec = Recorder("a", autosave=False)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=ExplodingCompletions()))
    client = RecordedOpenAIClient(fake_client, rec)
    with pytest.raises(RuntimeError, match="boom"):
        client.chat.completions.create(model="m", messages=[])
    assert len(rec) == 0


def test_langgraph_adapter_missing_dep():
    try:
        import langgraph  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("langgraph is installed; import-guard test not applicable")
    from replayagent.adapters.langgraph import record_langgraph_run

    rec = Recorder("a", autosave=False)
    with pytest.raises(ImportError, match="pip install"):
        record_langgraph_run(object(), {}, rec)

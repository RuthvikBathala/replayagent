"""OpenAI adapter: wrap a chat.completions client to log LLM calls as steps.

Requires the ``openai`` extra: ``pip install replayagent[openai]``.
"""

from __future__ import annotations

from typing import Any

from replayagent.recorder import Recorder


class RecordedChatCompletions:
    """Drop-in wrapper around ``client.chat.completions`` that records each call."""

    def __init__(self, completions: Any, recorder: Recorder) -> None:
        self._completions = completions
        self._recorder = recorder

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._completions.create(*args, **kwargs)
        try:
            choice = response.choices[0]
            message = choice.message
            tool_calls = []
            for tc in getattr(message, "tool_calls", None) or []:
                fn = tc.function
                tool_calls.append({"name": fn.name, "arguments": fn.arguments})
                self._recorder.tool_call(name=fn.name, arguments={"raw": fn.arguments})
            self._recorder.llm_call(
                messages=kwargs.get("messages", []),
                response=getattr(message, "content", None),
                model=kwargs.get("model") or getattr(response, "model", None),
                usage=_usage_dict(getattr(response, "usage", None)),
                tool_calls=tool_calls,
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except Exception:  # noqa: BLE001, S110 — recording must never break the call
            pass
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class RecordedChat:
    """Wrapper around ``client.chat`` exposing recorded ``completions``."""

    def __init__(self, chat: Any, recorder: Recorder) -> None:
        self._chat = chat
        self.completions = RecordedChatCompletions(chat.completions, recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class RecordedOpenAIClient:
    """Wrap an ``openai.OpenAI()`` client so chat completions are recorded.

    Usage::

        client = RecordedOpenAIClient(openai.OpenAI(), recorder)
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
        # -> an llm_call step is recorded automatically
    """

    def __init__(self, client: Any, recorder: Recorder) -> None:
        self._client = client
        self.chat = RecordedChat(client.chat, recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

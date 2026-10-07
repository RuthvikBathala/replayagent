"""LangGraph adapter: record a compiled graph run as a replayagent trajectory.

Requires the ``langgraph`` extra: ``pip install replayagent[langgraph]``.
"""

from __future__ import annotations

from typing import Any

from replayagent.recorder import Recorder


def _message_to_steps(message: Any, rec: Recorder) -> None:
    """Translate one LangChain message into recorder steps."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    if isinstance(message, HumanMessage):
        rec.record_step("llm_call", {"role": "user", "content": _content_text(message)})
    elif isinstance(message, AIMessage):
        for tc in getattr(message, "tool_calls", []) or []:
            rec.tool_call(name=tc.get("name", ""), arguments=tc.get("args", {}))
        rec.llm_call(
            response=_content_text(message),
            usage=getattr(message, "usage_metadata", None) or {},
        )
    elif isinstance(message, ToolMessage):
        rec.tool_result(name=message.name or "", result=_content_text(message))


def _content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def record_langgraph_run(
    graph: Any,
    input: dict[str, Any],
    recorder: Recorder,
    *,
    stream: bool = True,
    stream_config: dict[str, Any] | None = None,
) -> Any:
    """Invoke a compiled LangGraph graph, recording messages/tool calls as steps.

    Returns the graph's final output. The last AI message is also recorded
    as the trajectory's final answer.
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError as exc:
        raise ImportError(
            "record_langgraph_run needs langchain-core: pip install replayagent[langgraph]"
        ) from exc

    final_output: Any = None
    seen_ids: set[str] = set()
    if stream:
        for chunk in graph.stream(input, config=stream_config or {}):
            final_output = chunk
            for _node, update in chunk.items():
                messages = (update or {}).get("messages", [])
                for msg in messages if isinstance(messages, list) else []:
                    mid = getattr(msg, "id", None) or str(id(msg))
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    _message_to_steps(msg, recorder)
    else:
        final_output = graph.invoke(input, config=stream_config or {})
        messages = []
        if isinstance(final_output, dict):
            messages = final_output.get("messages", []) or []
        for msg in messages if isinstance(messages, list) else []:
            _message_to_steps(msg, recorder)

    # Record the final AI message text as the final answer.
    last_text: str | None = None
    search_space = []
    if isinstance(final_output, dict):
        maybe = final_output.get("messages", [])
        search_space = maybe if isinstance(maybe, list) else []
    for msg in reversed(search_space):
        if isinstance(msg, AIMessage):
            last_text = _content_text(msg)
            break
    if last_text:
        recorder.final_answer(last_text)
    return final_output

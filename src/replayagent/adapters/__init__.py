"""Framework adapters (all optional, import-guarded)."""

from replayagent.adapters.langgraph import record_langgraph_run
from replayagent.adapters.openai import RecordedOpenAIClient

__all__ = ["RecordedOpenAIClient", "record_langgraph_run"]

"""Trajectory schema: the immutable record of one agent run."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """The kind of event a step represents."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"


class Step(BaseModel):
    """A single event in an agent trajectory."""

    type: StepType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}


class Trajectory(BaseModel):
    """One recorded agent run: inputs, model, and the ordered steps it took."""

    run_id: str
    agent_name: str
    model: str
    input: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def final_answer(self) -> str | None:
        """Return the payload text of the last final_answer step, if any."""
        for step in reversed(self.steps):
            if step.type == StepType.FINAL_ANSWER:
                text = step.payload.get("text")
                return str(text) if text is not None else None
        return None

    def tool_calls(self) -> list[Step]:
        """Return all tool_call steps in order."""
        return [s for s in self.steps if s.type == StepType.TOOL_CALL]

    def tool_call_names(self) -> list[str]:
        """Return the names of all tools called, in order."""
        return [str(s.payload.get("name", "")) for s in self.tool_calls()]

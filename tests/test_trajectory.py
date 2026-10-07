"""Tests for the trajectory schema."""

import pytest
from pydantic import ValidationError

from replayagent.trajectory import Step, StepType, Trajectory


def test_step_defaults():
    s = Step(type=StepType.LLM_CALL)
    assert s.payload == {}
    assert s.metadata == {}
    assert s.timestamp is not None


def test_step_rejects_unknown_type():
    with pytest.raises(ValidationError):
        Step(type="not_a_step")


def test_trajectory_helpers():
    from tests.conftest import make_trajectory

    t = make_trajectory(answer="hello world", tools=("search", "fetch"))
    assert t.final_answer() == "hello world"
    assert t.tool_call_names() == ["search", "fetch"]
    assert len(t.tool_calls()) == 2


def test_trajectory_no_final_answer():
    t = Trajectory(run_id="x", agent_name="a", model="m")
    assert t.final_answer() is None
    assert t.tool_call_names() == []


def test_trajectory_round_trip_json():
    from tests.conftest import make_trajectory

    t = make_trajectory()
    t2 = Trajectory.model_validate_json(t.model_dump_json())
    assert t2 == t

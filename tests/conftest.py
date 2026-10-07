"""Shared fixtures for the replayagent test suite."""

from pathlib import Path

import pytest

from replayagent import Recorder
from replayagent.trajectory import Trajectory


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def make_trajectory(
    run_id: str = "base123",
    agent_name: str = "demo-agent",
    model: str = "gpt-4o",
    query: str = "reset my password",
    answer: str = "Done — your password has been reset.",
    tools: tuple[str, ...] = ("reset_password",),
) -> Trajectory:
    rec = Recorder(agent_name, model=model, input={"query": query}, run_id=run_id, autosave=False)
    rec.llm_call(
        messages=[{"role": "user", "content": query}], response="working on it", model=model
    )
    for tool in tools:
        rec.tool_call(name=tool, arguments={"q": query})
        rec.tool_result(name=tool, result={"ok": True})
    rec.final_answer(answer)
    return rec.trajectory


@pytest.fixture()
def baseline() -> Trajectory:
    return make_trajectory()


@pytest.fixture()
def baseline_saved(baseline: Trajectory, store_dir: Path) -> Trajectory:
    from replayagent.recorder import save_trajectory

    save_trajectory(baseline, store_dir / f"{baseline.run_id}.jsonl")
    return baseline

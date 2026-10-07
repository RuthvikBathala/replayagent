"""Tests for replay: re-execution against recorded inputs."""

from pathlib import Path

import pytest

from replayagent import Recorder, load_run
from replayagent.replay import replay
from replayagent.trajectory import Trajectory


def _v2_agent(input_data: dict, recorder: Recorder, model: str = "gpt-4o-mini"):
    q = input_data["query"]
    recorder.llm_call(messages=[{"role": "user", "content": q}], response="on it", model=model)
    recorder.tool_call(name="reset_password", arguments={"q": q})
    recorder.tool_result(name="reset_password", result={"ok": True})
    recorder.final_answer("Done — your password has been reset.")


def test_replay_uses_baseline_input(baseline: Trajectory, store_dir: Path):
    seen: dict = {}

    def agent_fn(input_data: dict, recorder: Recorder):
        seen.update(input_data)
        recorder.final_answer("x")

    new = replay(baseline, agent_fn, model="new-model", store_dir=store_dir, autosave=False)
    assert seen == {"query": "reset my password"}
    assert new.model == "new-model"
    assert new.agent_name == baseline.agent_name
    assert new.run_id != baseline.run_id
    assert new.metadata["replayed_from"] == baseline.run_id


def test_replay_accepts_run_id(baseline_saved: Trajectory, store_dir: Path):
    new = replay(baseline_saved.run_id, _v2_agent, store_dir=store_dir)
    assert new.metadata["replayed_from"] == baseline_saved.run_id
    # autosave=True persists
    assert load_run(new.run_id, store_dir).run_id == new.run_id


def test_replay_missing_run_id_raises(store_dir: Path):
    with pytest.raises(FileNotFoundError):
        replay("ghost", _v2_agent, store_dir=store_dir)


def test_replay_metadata_merged(baseline: Trajectory, store_dir: Path):
    new = replay(baseline, _v2_agent, store_dir=store_dir, autosave=False, metadata={"ci": "true"})
    assert new.metadata["ci"] == "true"
    assert new.metadata["replayed_from"] == baseline.run_id


def test_replay_no_autosave_writes_nothing(baseline: Trajectory, store_dir: Path):
    replay(baseline, _v2_agent, store_dir=store_dir, autosave=False)
    assert list(store_dir.glob("*.jsonl")) == []

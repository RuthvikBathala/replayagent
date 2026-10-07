"""Tests for the Recorder: capture, persistence, round-trip, decorator."""

from pathlib import Path

import pytest

from replayagent import Recorder, list_runs, load_run, record, record_step
from replayagent.recorder import save_trajectory, traced
from replayagent.trajectory import StepType, Trajectory


def test_context_manager_autosaves(store_dir: Path):
    with record("agent-a", model="m1", input={"q": "hi"}, store_dir=store_dir) as rec:
        rec.llm_call(response="yo")
        run_id = rec.run_id
    assert (store_dir / f"{run_id}.jsonl").exists()
    loaded = load_run(run_id, store_dir)
    assert loaded.agent_name == "agent-a"
    assert loaded.model == "m1"
    assert loaded.input == {"q": "hi"}
    assert len(loaded.steps) == 1
    assert loaded.steps[0].type == StepType.LLM_CALL


def test_manual_step_api(store_dir: Path):
    rec = Recorder("agent-b", store_dir=store_dir, autosave=False)
    record_step(rec, "tool_call", {"name": "search", "arguments": {}})
    record_step(rec, StepType.TOOL_RESULT, {"name": "search", "result": []})
    rec.final_answer("done")
    assert len(rec) == 3
    assert [s.type for s in rec] == [
        StepType.TOOL_CALL,
        StepType.TOOL_RESULT,
        StepType.FINAL_ANSWER,
    ]


def test_convenience_methods_payloads():
    rec = Recorder("a", autosave=False)
    rec.tool_call(name="t", arguments={"x": 1})
    rec.tool_result(name="t", result="ok")
    steps = list(rec)
    assert steps[0].payload["name"] == "t"
    assert steps[0].payload["arguments"] == {"x": 1}
    assert steps[1].payload["result"] == "ok"


def test_load_missing_run_raises(store_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_run("nope", store_dir)


def test_list_runs_newest_first(store_dir: Path, baseline_saved: Trajectory):
    rec = Recorder("second", store_dir=store_dir)
    rec.final_answer("x")
    rec.save()
    runs = list_runs(store_dir)
    assert [r.agent_name for r in runs] == ["second", "demo-agent"]


def test_list_runs_empty_dir(tmp_path: Path):
    assert list_runs(tmp_path / "missing") == []


def test_traced_decorator_saves(store_dir: Path, monkeypatch: pytest.MonkeyPatch):
    import replayagent.recorder as recmod

    monkeypatch.setattr(recmod, "DEFAULT_STORE_DIR", store_dir)

    @traced("decorated-agent", model="m9")
    def my_agent(query: str, recorder: Recorder) -> str:
        recorder.llm_call(response="r")
        recorder.final_answer("finished")
        return "finished"

    assert my_agent("q") == "finished"
    runs = list_runs(store_dir)
    assert len(runs) == 1
    assert runs[0].agent_name == "decorated-agent"
    assert runs[0].final_answer() == "finished"


def test_save_trajectory_standalone(tmp_path: Path):
    from tests.conftest import make_trajectory

    t = make_trajectory(run_id="solo")
    p = save_trajectory(t, tmp_path / "solo.jsonl")
    assert p.exists()
    assert load_run("solo", tmp_path).run_id == "solo"

"""Tests for diff logic and all four reporters."""

import json
import xml.etree.ElementTree as ET

from rich.console import Console

from replayagent.diff import diff_to_dict, diff_trajectories
from replayagent.judges import ExactMatchJudge, ToolCallJudge
from replayagent.reporters import html_report, json_report, junit_report, terminal_report
from tests.conftest import make_trajectory


def test_identical_trajectories_no_changes():
    d = diff_trajectories(make_trajectory(), make_trajectory(run_id="b"))
    assert d.step_changes == []
    assert d.added_tool_calls == []
    assert d.removed_tool_calls == []
    assert not d.is_regression


def test_changed_step_detected():
    d = diff_trajectories(make_trajectory(), make_trajectory(run_id="b", answer="something else"))
    assert any(c.kind == "changed" for c in d.step_changes)
    assert d.steps_changed == len(d.step_changes)


def test_added_removed_tool_calls():
    d = diff_trajectories(
        make_trajectory(tools=("a", "b")), make_trajectory(run_id="b", tools=("a", "c"))
    )
    assert d.added_tool_calls == ["c"]
    assert d.removed_tool_calls == ["b"]
    assert d.is_regression  # tool-call sequence changed => regression


def test_added_step_detected():
    from replayagent import Recorder

    rec = Recorder("x", autosave=False)
    rec.final_answer("only")
    d = diff_trajectories(make_trajectory(), rec.trajectory)
    assert any(c.kind == "removed" for c in d.step_changes)


def test_judges_run_and_gate_regression():
    d = diff_trajectories(
        make_trajectory(),
        make_trajectory(run_id="b", answer="totally different"),
        judges=[ExactMatchJudge(), ToolCallJudge()],
    )
    assert len(d.judge_results) == 2
    assert not d.judge_results[0].passed
    assert d.judge_results[1].passed
    assert not d.all_passed
    assert d.is_regression


def test_no_judges_no_regression_when_identical():
    d = diff_trajectories(make_trajectory(), make_trajectory(run_id="b"), judges=[])
    assert d.all_passed  # vacuous
    assert not d.is_regression


def test_diff_to_dict_json_serializable():
    d = diff_trajectories(
        make_trajectory(), make_trajectory(run_id="b", answer="x"), judges=[ExactMatchJudge()]
    )
    data = diff_to_dict(d)
    json.dumps(data)  # must not raise
    assert data["baseline_run_id"] == "base123"


def test_terminal_report():
    d = diff_trajectories(
        make_trajectory(), make_trajectory(run_id="b", answer="x"), judges=[ExactMatchJudge()]
    )
    console = Console(record=True, width=100)
    terminal_report(d, console)
    text = console.export_text()
    assert "REGRESSION DETECTED" in text
    assert "exact_match" in text


def test_terminal_report_no_regression():
    d = diff_trajectories(
        make_trajectory(), make_trajectory(run_id="b"), judges=[ExactMatchJudge()]
    )
    console = Console(record=True, width=100)
    terminal_report(d, console)
    assert "NO REGRESSION" in console.export_text()


def test_json_report():
    d = diff_trajectories(make_trajectory(), make_trajectory(run_id="b"))
    parsed = json.loads(json_report(d))
    assert parsed["candidate_run_id"] == "b"


def test_junit_report_structure():
    d = diff_trajectories(
        make_trajectory(),
        make_trajectory(run_id="b", answer="x"),
        judges=[ExactMatchJudge(), ToolCallJudge()],
    )
    root = ET.fromstring(junit_report(d))
    assert root.tag == "testsuite"
    assert root.get("tests") == "2"
    assert root.get("failures") == "1"
    cases = root.findall("testcase")
    assert {c.get("name") for c in cases} == {"exact_match", "tool_calls"}
    assert cases[0].find("failure") is not None or cases[1].find("failure") is not None


def test_html_report_escapes_and_verdict():
    base = make_trajectory(answer="<b>hi</b>")
    cand = make_trajectory(run_id="b", answer="<b>bye</b>")
    d = diff_trajectories(base, cand, judges=[ExactMatchJudge()])
    out = html_report(d, baseline=base, candidate=cand)
    assert "REGRESSION DETECTED" in out
    assert "<b>hi</b>" not in out  # escaped
    assert "&lt;b&gt;hi&lt;/b&gt;" in out

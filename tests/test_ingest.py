"""Tests for production-trace ingestion, failure clustering, and draft cases."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from replayagent import (
    IngestError,
    check_case,
    cluster_traces,
    detect_failure,
    draft_cases,
    list_cases,
    list_drafts,
    load_draft,
    load_traces,
    parse_traces,
    promote_draft,
)
from replayagent.cli import app
from replayagent.ingest import normalize_signature
from replayagent.recorder import load_run
from tests.conftest import make_trajectory

runner = CliRunner()
FIXTURE = Path(__file__).parent.parent / "examples" / "production_traces.json"


def _trace(**overrides):
    base = {
        "id": "r1",
        "name": "support-agent",
        "model": "gpt-4o",
        "input": {"query": "refund for order 1"},
        "status": "success",
        "steps": [
            {"type": "tool_call", "name": "issue_refund", "arguments": {"order": "1"}},
            {"type": "tool_result", "name": "issue_refund", "result": {"ok": True}},
            {"type": "final_answer", "text": "Refunded."},
        ],
    }
    base.update(overrides)
    return base


def _failed_trace(run_id: str, order: str, seconds: int) -> dict:
    return _trace(
        id=run_id,
        status="error",
        error=f"Refund API timeout after {seconds}s for order {order}",
        steps=[
            {"type": "tool_call", "name": "issue_refund", "arguments": {"order": order}},
            {
                "type": "tool_result",
                "name": "issue_refund",
                "result": {"ok": False},
                "status": "error",
                "error": f"Refund API timeout after {seconds}s for order {order}",
            },
        ],
    )


# -- ingestion parsing -------------------------------------------------------


def test_parse_valid_trace_maps_steps_and_fields():
    (trace,) = parse_traces([_trace()])
    assert trace.run_id == "r1"
    assert trace.agent_name == "support-agent"
    traj = trace.to_trajectory()
    assert traj.tool_call_names() == ["issue_refund"]
    assert traj.final_answer() == "Refunded."
    assert traj.input == {"query": "refund for order 1"}


def test_parse_accepts_wrapped_and_langfuse_flavoured_uploads():
    langfuse = {
        "trace_id": "lf-1",
        "name": "agent",
        "level": "ERROR",
        "input": "hello",
        "observations": [
            {"type": "GENERATION", "output": "thinking"},
            {"type": "TOOL", "name": "search", "input": {"q": "x"}, "output": {"hits": []}},
        ],
    }
    (trace,) = parse_traces({"traces": [langfuse]})
    assert trace.run_id == "lf-1"
    assert trace.input == {"query": "hello"}
    assert trace.status == "error"
    assert trace.to_trajectory().tool_call_names() == ["search"]


def test_parse_single_trace_object_and_string_input():
    (trace,) = parse_traces(_trace(id="solo", input="just a question"))
    assert trace.run_id == "solo"
    assert trace.input == {"query": "just a question"}


def test_load_traces_from_fixture_file():
    traces = load_traces(FIXTURE)
    assert len(traces) == 5
    assert traces[0].run_id == "prod-001"


def test_load_traces_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_traces(tmp_path / "nope.json")


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
    ],
)
def test_load_traces_invalid_json(tmp_path: Path, bad: str):
    p = tmp_path / "bad.json"
    p.write_text(bad)
    with pytest.raises(IngestError):
        load_traces(p)


@pytest.mark.parametrize(
    "data",
    [
        42,
        [{"name": "no id here"}],
        [{"id": "x", "steps": "not-a-list"}],
        [{"id": "x", "steps": ["not-a-dict"]}],
        ["not-a-dict"],
    ],
)
def test_parse_malformed_uploads_raise(data):
    with pytest.raises(IngestError):
        parse_traces(data)


# -- failure detection ---------------------------------------------------------


def test_detect_success_is_not_failure():
    (trace,) = parse_traces([_trace()])
    assert detect_failure(trace) is None


def test_detect_explicit_error_status():
    (trace,) = parse_traces([_trace(status="error", error="boom 42")])
    failure = detect_failure(trace)
    assert failure is not None
    assert "status:error" in failure.reasons
    assert failure.failing_tool == "issue_refund"


def test_detect_step_error_without_top_level_status():
    raw = _failed_trace("s1", "9", 5)
    raw["status"] = "success"
    del raw["error"]
    (trace,) = parse_traces([raw])
    failure = detect_failure(trace)
    assert failure is not None
    assert "step-error" in failure.reasons
    assert failure.failing_tool == "issue_refund"


def test_detect_tool_result_ok_false():
    raw = _trace(
        steps=[
            {"type": "tool_call", "name": "pay", "arguments": {}},
            {"type": "tool_result", "name": "pay", "result": {"ok": False}},
        ]
    )
    (trace,) = parse_traces([raw])
    assert detect_failure(trace) is not None


def test_detect_judge_failure_from_expected_behaviour():
    raw = _trace(expected="Your refund is still pending.")
    raw["steps"][-1] = {"type": "final_answer", "text": "Your refund was approved."}
    (trace,) = parse_traces([raw])
    failure = detect_failure(trace)
    assert failure is not None
    assert "judge-failure" in failure.reasons


def test_expected_behaviour_match_is_not_failure():
    raw = _trace(expected="Refunded.")
    (trace,) = parse_traces([raw])
    assert detect_failure(trace) is None


def test_normalize_signature_strips_ids_and_numbers():
    a = normalize_signature(
        "Timeout after 30s for order 881 (req 550e8400-e29b-41d4-a716-446655440000)"
    )
    b = normalize_signature(
        "timeout after 12s for order 902 (req 123e4567-e89b-12d3-a456-426614174000)"
    )
    assert a == b


# -- clustering ------------------------------------------------------------------


def test_similar_failures_cluster_together_with_count():
    traces = parse_traces([_failed_trace("f2", "902", 12), _failed_trace("f1", "881", 30)])
    clusters = cluster_traces(traces)
    assert len(clusters) == 1
    assert clusters[0].count == 2
    assert clusters[0].run_ids == ["f1", "f2"]  # deterministic order
    assert clusters[0].representative_run_id == "f1"
    assert clusters[0].failing_tool == "issue_refund"


def test_different_tools_form_separate_clusters():
    other = _trace(
        id="g1",
        status="error",
        error="search backend down",
        steps=[{"type": "tool_call", "name": "search", "arguments": {}}],
    )
    traces = parse_traces([_failed_trace("f1", "881", 30), other])
    clusters = cluster_traces(traces)
    assert len(clusters) == 2
    assert {c.failing_tool for c in clusters} == {"issue_refund", "search"}


def test_fixture_clusters_and_counts():
    clusters = cluster_traces(load_traces(FIXTURE))
    by_tool = {c.failing_tool: c for c in clusters}
    assert by_tool["issue_refund"].count == 2
    assert by_tool["approve_refund"].count == 1  # judge failure vs expected behaviour


# -- draft generation + promotion -------------------------------------------------


@pytest.fixture()
def drafts_dir(tmp_path: Path) -> Path:
    return tmp_path / "drafts"


def test_draft_generated_marked_and_prefilled(drafts_dir: Path):
    traces = parse_traces([_failed_trace("f1", "881", 30), _failed_trace("f2", "902", 12)])
    drafts = draft_cases(cluster_traces(traces), traces, drafts_dir=drafts_dir)
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.metadata["draft"] is True
    assert draft.metadata["needs_review"] is True
    assert draft.metadata["cluster_count"] == 2
    assert draft.name.startswith("[DRAFT]")
    assert draft.source_run_id == "f1"
    assert draft.expected_tool_calls == ["issue_refund"]
    assert draft.input == {"query": "refund for order 1"}
    assert load_draft(draft.case_id, drafts_dir).case_id == draft.case_id
    assert list_drafts(drafts_dir)[0].case_id == draft.case_id


def test_draft_prefills_expected_from_trace_expectation(drafts_dir: Path):
    raw = _trace(id="j1", expected="Your refund is still pending.")
    raw["steps"][-1] = {"type": "final_answer", "text": "Your refund was approved."}
    traces = parse_traces([raw])
    (draft,) = draft_cases(cluster_traces(traces), traces, drafts_dir=drafts_dir)
    assert draft.expected_final_answer == "Your refund is still pending."


def test_promote_draft_goes_through_existing_case_flow(drafts_dir: Path, tmp_path: Path):
    cases_dir = tmp_path / "cases"
    traces = parse_traces([_failed_trace("f1", "881", 30)])
    (draft,) = draft_cases(cluster_traces(traces), traces, drafts_dir=drafts_dir)
    case = promote_draft(
        draft.case_id,
        drafts_dir=drafts_dir,
        cases_dir=cases_dir,
        expected_final_answer="Refund failed — please retry.",
    )
    assert not case.name.startswith("[DRAFT]")
    assert case.case_id != draft.case_id
    assert case.expected_final_answer == "Refund failed — please retry."
    assert list_cases(cases_dir)[0].case_id == case.case_id
    # the promoted case behaves like any other case in check_case
    good = make_trajectory(
        run_id="c1", answer="Refund failed — please retry.", tools=("issue_refund",)
    )
    assert not check_case(case, good).is_regression


def test_promote_missing_draft_raises(drafts_dir: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        promote_draft("nope", drafts_dir=drafts_dir, cases_dir=tmp_path / "cases")


# -- CLI ---------------------------------------------------------------------------


def test_cli_ingest_clusters_draft_and_promote(tmp_path: Path):
    store, drafts, cases = tmp_path / "runs", tmp_path / "drafts", tmp_path / "cases"
    result = runner.invoke(app, ["ingest", str(FIXTURE), "--store-dir", str(store)])
    assert result.exit_code == 0, result.output
    assert "5" in result.output and "3 failure" in result.output
    assert load_run("prod-001", store).run_id == "prod-001"  # usable by promote/eval

    result = runner.invoke(app, ["clusters", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert "issue_refund" in result.output

    result = runner.invoke(
        app,
        ["draft-cases", str(FIXTURE), "--drafts-dir", str(drafts), "--store-dir", str(store)],
    )
    assert result.exit_code == 0, result.output
    assert "DRAFT" in result.output
    all_drafts = list_drafts(drafts)
    assert len(all_drafts) == 2

    result = runner.invoke(app, ["drafts", "--drafts-dir", str(drafts)])
    assert result.exit_code == 0
    assert all_drafts[0].case_id in result.output

    target = next(d for d in all_drafts if d.metadata["failing_tool"] == "issue_refund")
    result = runner.invoke(
        app,
        [
            "promote-draft",
            target.case_id,
            "--expected-answer",
            "Refund failed — please retry.",
            "--drafts-dir",
            str(drafts),
            "--cases-dir",
            str(cases),
        ],
    )
    assert result.exit_code == 0, result.output
    promoted = list_cases(cases)
    assert len(promoted) == 1
    assert promoted[0].expected_final_answer == "Refund failed — please retry."


def test_cli_ingest_malformed_file_errors(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"name": "no id"}]))
    result = runner.invoke(app, ["ingest", str(bad), "--store-dir", str(tmp_path / "runs")])
    assert result.exit_code != 0


def test_cli_clusters_no_failures(tmp_path: Path):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps([_trace()]))
    result = runner.invoke(app, ["clusters", str(ok)])
    assert result.exit_code == 0
    assert "No failures" in result.output

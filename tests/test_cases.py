"""Tests for regression cases: promotion, persistence, expectation overrides, checking."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from replayagent import (
    RegressionCase,
    check_case,
    list_cases,
    load_case,
    promote_run,
    promote_trajectory,
)
from replayagent.cases import save_case
from replayagent.cli import app
from replayagent.recorder import save_trajectory
from tests.conftest import make_trajectory

runner = CliRunner()


@pytest.fixture()
def cases_dir(tmp_path: Path) -> Path:
    return tmp_path / "cases"


def test_promote_defaults_to_run_behaviour(baseline, cases_dir: Path):
    case = promote_trajectory(baseline, cases_dir=cases_dir)
    assert isinstance(case, RegressionCase)
    assert case.source_run_id == baseline.run_id
    assert case.expected_final_answer == baseline.final_answer()
    assert case.expected_tool_calls == ["reset_password"]
    assert case.input == baseline.input
    assert list_cases(cases_dir)[0].case_id == case.case_id


def test_promote_round_trip(baseline, cases_dir: Path):
    case = promote_trajectory(
        baseline, name="Password reset broke", reason="wrong tool", cases_dir=cases_dir
    )
    loaded = load_case(case.case_id, cases_dir)
    assert loaded.name == "Password reset broke"
    assert loaded.reason == "wrong tool"
    assert loaded.baseline.run_id == baseline.run_id
    assert loaded.baseline.steps == baseline.steps


def test_promote_run_by_id(store_dir: Path, baseline_saved, cases_dir: Path):
    case = promote_run(baseline_saved.run_id, store_dir=store_dir, cases_dir=cases_dir)
    assert case.source_run_id == baseline_saved.run_id


def test_promote_missing_run_raises(store_dir: Path, cases_dir: Path):
    with pytest.raises(FileNotFoundError):
        promote_run("nope", store_dir=store_dir, cases_dir=cases_dir)


def test_duplicate_case_id_rejected(baseline, cases_dir: Path):
    promote_trajectory(baseline, case_id="fixed1", cases_dir=cases_dir)
    with pytest.raises(FileExistsError):
        promote_trajectory(baseline, case_id="fixed1", cases_dir=cases_dir)
    case = list_cases(cases_dir)[0]
    save_case(case, cases_dir=cases_dir, overwrite=True)  # overwrite is allowed explicitly


def test_list_cases_empty(tmp_path: Path):
    assert list_cases(tmp_path / "missing") == []


def test_load_missing_case_raises(cases_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_case("nope", cases_dir)


def test_expected_answer_override_replaces_failed_answer(cases_dir: Path):
    failed = make_trajectory(run_id="fail1", answer="Refund approved.")
    case = promote_trajectory(
        failed, expected_final_answer="Refund is still pending.", cases_dir=cases_dir
    )
    expected = case.expected_trajectory()
    assert expected.final_answer() == "Refund is still pending."
    # the stored baseline keeps the original (failed) answer for provenance
    assert case.baseline.final_answer() == "Refund approved."


def test_expected_trajectory_without_final_answer_step(cases_dir: Path):
    from replayagent import Recorder

    rec = Recorder("a", run_id="noans", autosave=False)
    rec.tool_call(name="search", arguments={})
    case = promote_trajectory(
        rec.trajectory, expected_final_answer="eventually", cases_dir=cases_dir
    )
    assert case.expected_trajectory().final_answer() == "eventually"


def test_check_case_passes_for_matching_candidate(baseline, cases_dir: Path):
    case = promote_trajectory(baseline, cases_dir=cases_dir)
    candidate = make_trajectory(run_id="cand1")
    diff = check_case(case, candidate)
    assert not diff.is_regression


def test_check_case_fails_until_failure_is_fixed(cases_dir: Path):
    failed = make_trajectory(run_id="fail1", answer="Refund approved.", tools=("approve_refund",))
    case = promote_trajectory(
        failed,
        expected_final_answer="Refund is still pending.",
        expected_tool_calls=["check_refund_status"],
        cases_dir=cases_dir,
    )
    still_broken = make_trajectory(
        run_id="cand1", answer="Refund approved.", tools=("approve_refund",)
    )
    assert check_case(case, still_broken).is_regression
    fixed = make_trajectory(
        run_id="cand2", answer="Refund is still pending.", tools=("check_refund_status",)
    )
    assert not check_case(case, fixed).is_regression


# -- CLI -------------------------------------------------------------------


@pytest.fixture()
def cli_dirs(tmp_path: Path, baseline) -> tuple[Path, Path]:
    store = tmp_path / "runs"
    store.mkdir()
    save_trajectory(baseline, store / f"{baseline.run_id}.jsonl")
    return store, tmp_path / "cases"


def test_cli_promote_list_and_eval_case(cli_dirs: tuple[Path, Path], baseline):
    store, cases = cli_dirs
    result = runner.invoke(
        app,
        [
            "promote",
            baseline.run_id,
            "--name",
            "reset regression",
            "--reason",
            "broke in prod",
            "--store-dir",
            str(store),
            "--cases-dir",
            str(cases),
        ],
    )
    assert result.exit_code == 0, result.output
    case = list_cases(cases)[0]
    assert case.case_id in result.output

    result = runner.invoke(app, ["cases", "--cases-dir", str(cases)])
    assert result.exit_code == 0
    assert case.case_id in result.output

    save_trajectory(make_trajectory(run_id="good1"), store / "good1.jsonl")
    result = runner.invoke(
        app,
        [
            "eval-case",
            case.case_id,
            "--candidate",
            "good1",
            "--store-dir",
            str(store),
            "--cases-dir",
            str(cases),
        ],
    )
    assert result.exit_code == 0, result.output

    save_trajectory(make_trajectory(run_id="bad1", answer="wrong"), store / "bad1.jsonl")
    result = runner.invoke(
        app,
        [
            "eval-case",
            case.case_id,
            "--candidate",
            "bad1",
            "--store-dir",
            str(store),
            "--cases-dir",
            str(cases),
        ],
    )
    assert result.exit_code == 1


def test_cli_promote_missing_run(cli_dirs: tuple[Path, Path]):
    store, cases = cli_dirs
    result = runner.invoke(
        app, ["promote", "nope", "--store-dir", str(store), "--cases-dir", str(cases)]
    )
    assert result.exit_code != 0

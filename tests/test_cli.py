"""Tests for the typer CLI — all offline, using isolated store dirs."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from replayagent.cli import app
from replayagent.recorder import load_run

runner = CliRunner()


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _store_args(workdir: Path) -> list[str]:
    return ["--store-dir", str(workdir / ".replayagent" / "runs")]


def _record_demo(workdir: Path) -> str:
    result = runner.invoke(app, ["record", "--query", "q1", *_store_args(workdir)])
    assert result.exit_code == 0, result.output
    # extract run id from output: "Recorded run <id> (...)"
    line = next(x for x in result.output.splitlines() if "Recorded run" in x)
    return line.split()[2]


def test_init_creates_layout(workdir: Path):
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (workdir / ".replayagent" / "runs").is_dir()
    assert (workdir / ".replayagent" / "config.yaml").exists()


def test_record_and_list(workdir: Path):
    run_id = _record_demo(workdir)
    result = runner.invoke(app, ["list", *_store_args(workdir)])
    assert result.exit_code == 0
    assert run_id in result.output


def test_list_empty(workdir: Path):
    result = runner.invoke(app, ["list", *_store_args(workdir)])
    assert result.exit_code == 0
    assert "No recorded runs" in result.output


def test_replay_same_model_no_regression(workdir: Path):
    run_id = _record_demo(workdir)
    result = runner.invoke(app, ["replay", run_id, *_store_args(workdir)])
    assert result.exit_code == 0, result.output
    assert "NO REGRESSION" in result.output


def test_replay_new_model_flags_regression(workdir: Path):
    run_id = _record_demo(workdir)
    result = runner.invoke(app, ["replay", run_id, "--model", "other-model", *_store_args(workdir)])
    assert result.exit_code == 1, result.output
    assert "REGRESSION DETECTED" in result.output


def test_eval_exit_codes(workdir: Path):
    base = _record_demo(workdir)
    # candidate identical to baseline -> exit 0
    result = runner.invoke(app, ["replay", base, "--judge", "tools", *_store_args(workdir)])
    assert result.exit_code == 0
    cand_line = next(x for x in result.output.splitlines() if "Replayed run" in x)
    cand_id = cand_line.split()[2]

    result = runner.invoke(
        app, ["eval", base, "--candidate", cand_id, "--judge", "tools", *_store_args(workdir)]
    )
    assert result.exit_code == 0, result.output

    # candidate with different answer vs exact judge -> exit 1
    result = runner.invoke(app, ["replay", base, "--model", "zzz", *_store_args(workdir)])
    cand2 = next(x for x in result.output.splitlines() if "Replayed run" in x).split()[2]
    result = runner.invoke(
        app, ["eval", base, "--candidate", cand2, "--judge", "exact", *_store_args(workdir)]
    )
    assert result.exit_code == 1


def test_report_json_and_html(workdir: Path, tmp_path: Path):
    base = _record_demo(workdir)
    result = runner.invoke(app, ["replay", base, *_store_args(workdir)])
    cand = next(x for x in result.output.splitlines() if "Replayed run" in x).split()[2]

    out = tmp_path / "r.html"
    result = runner.invoke(
        app,
        [
            "report",
            base,
            "--candidate",
            cand,
            "--format",
            "html",
            "-o",
            str(out),
            *_store_args(workdir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "<html" in out.read_text()

    import json

    result = runner.invoke(
        app, ["report", base, "--candidate", cand, "--format", "json", *_store_args(workdir)]
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["baseline_run_id"] == base


def test_report_junit(workdir: Path):
    base = _record_demo(workdir)
    result = runner.invoke(app, ["replay", base, *_store_args(workdir)])
    cand = next(x for x in result.output.splitlines() if "Replayed run" in x).split()[2]
    result = runner.invoke(
        app, ["report", base, "--candidate", cand, "--format", "junit", *_store_args(workdir)]
    )
    assert result.exit_code == 0
    assert "<testsuite" in result.output


def test_eval_unknown_judge_errors(workdir: Path):
    base = _record_demo(workdir)
    result = runner.invoke(app, ["replay", base, *_store_args(workdir)])
    cand = next(x for x in result.output.splitlines() if "Replayed run" in x).split()[2]
    result = runner.invoke(
        app, ["eval", base, "--candidate", cand, "--judge", "bogus", *_store_args(workdir)]
    )
    assert result.exit_code != 0


def test_replay_custom_agent_module(workdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "myagent.py").write_text(
        "def run(input_data, recorder):\n"
        "    recorder.llm_call(response='custom')\n"
        "    recorder.final_answer('custom answer')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    base = _record_demo(workdir)
    result = runner.invoke(app, ["replay", base, "--agent", "myagent:run", *_store_args(workdir)])
    assert result.exit_code == 1  # custom answer != demo answer -> regression on exact
    assert "REGRESSION DETECTED" in result.output


def test_recorded_run_loadable(workdir: Path):
    run_id = _record_demo(workdir)
    traj = load_run(run_id, workdir / ".replayagent" / "runs")
    assert traj.final_answer() is not None

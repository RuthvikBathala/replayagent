"""Regression cases: turn a real (often failed) run into a reusable test case.

The production-failure -> regression-eval loop: an agent passes its evals,
fails a real customer, and that failure is normally lost. Promoting the run
into a case preserves its input and the *expected* behaviour (which a reviewer
can correct — the failed run's own answer is rarely the right expectation),
so every future replay is checked against it and the fix has to stay fixed.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from replayagent.diff import TrajectoryDiff, diff_trajectories
from replayagent.judges import ExactMatchJudge, Judge, JudgeResult, ToolCallJudge
from replayagent.recorder import load_run
from replayagent.trajectory import Step, StepType, Trajectory

DEFAULT_CASES_DIR = Path(".replayagent") / "cases"


class RegressionCase(BaseModel):
    """A promoted run: its input plus the behaviour future runs must match."""

    case_id: str
    name: str
    source_run_id: str
    agent_name: str
    model: str
    input: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    expected_final_answer: str | None = None
    expected_tool_calls: list[str] = Field(default_factory=list)
    baseline: Trajectory
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def expected_trajectory(self) -> Trajectory:
        """The baseline trajectory with the reviewer's expectations applied.

        If an expected final answer was recorded, it replaces the source
        run's final-answer step (or is appended when the run had none), so
        judges compare candidates against what *should* have happened.
        """
        traj = self.baseline.model_copy(deep=True)
        if self.expected_tool_calls and len(self.expected_tool_calls) == len(
            traj.tool_call_names()
        ):
            # Reviewer corrected *which* tools should have been called (same
            # number of calls): rename call/result steps so diffs and judges
            # compare against the corrected sequence. A different *number*
            # of expected calls is left to the expected_tool_calls check in
            # check_case, which fails any candidate that does not match it.
            calls = iter(self.expected_tool_calls)
            results = iter(self.expected_tool_calls)
            for step in traj.steps:
                if step.type == StepType.TOOL_CALL:
                    step.payload = {**step.payload, "name": next(calls)}
                elif step.type == StepType.TOOL_RESULT:
                    name = next(results, None)
                    if name is not None:
                        step.payload = {**step.payload, "name": name}
        if self.expected_final_answer is not None:
            for step in reversed(traj.steps):
                if step.type == StepType.FINAL_ANSWER:
                    step.payload = {**step.payload, "text": self.expected_final_answer}
                    break
            else:
                traj.steps.append(
                    Step(type=StepType.FINAL_ANSWER, payload={"text": self.expected_final_answer})
                )
        return traj


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "case"


def promote_trajectory(
    traj: Trajectory,
    name: str | None = None,
    reason: str = "",
    expected_final_answer: str | None = None,
    expected_tool_calls: list[str] | None = None,
    cases_dir: Path | str | None = None,
    case_id: str | None = None,
    overwrite: bool = False,
) -> RegressionCase:
    """Promote a trajectory into a regression case and save it.

    Expectations default to the run itself (its final answer and tool-call
    sequence); pass overrides when the run *failed* and a reviewer knows what
    it should have done instead.
    """
    resolved_id = case_id or uuid.uuid4().hex[:12]
    case = RegressionCase(
        case_id=resolved_id,
        name=name or f"{traj.agent_name}-{traj.run_id}",
        source_run_id=traj.run_id,
        agent_name=traj.agent_name,
        model=traj.model,
        input=dict(traj.input),
        reason=reason,
        expected_final_answer=(
            expected_final_answer if expected_final_answer is not None else traj.final_answer()
        ),
        expected_tool_calls=(
            list(expected_tool_calls)
            if expected_tool_calls is not None
            else traj.tool_call_names()
        ),
        baseline=traj,
    )
    save_case(case, cases_dir=cases_dir, overwrite=overwrite)
    return case


def promote_run(
    run_id: str,
    store_dir: Path | str | None = None,
    **kwargs: Any,
) -> RegressionCase:
    """Load a recorded run by id and promote it (see :func:`promote_trajectory`)."""
    return promote_trajectory(load_run(run_id, store_dir), **kwargs)


def save_case(
    case: RegressionCase, cases_dir: Path | str | None = None, overwrite: bool = False
) -> Path:
    """Persist a case as one JSON file under the cases directory."""
    dest_dir = Path(cases_dir) if cases_dir is not None else DEFAULT_CASES_DIR
    dest = dest_dir / f"{_slugify(case.name)}--{case.case_id}.json"
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Regression case already exists: {dest}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(case.model_dump_json(indent=2), encoding="utf-8")
    return dest


def _case_path(case_id: str, cases_dir: Path | str | None) -> Path:
    dest_dir = Path(cases_dir) if cases_dir is not None else DEFAULT_CASES_DIR
    matches = sorted(dest_dir.glob(f"*--{case_id}.json")) if dest_dir.exists() else []
    if not matches:
        raise FileNotFoundError(f"No regression case {case_id!r} in {dest_dir}")
    return matches[0]


def load_case(case_id: str, cases_dir: Path | str | None = None) -> RegressionCase:
    """Load a regression case by its case id."""
    return RegressionCase.model_validate_json(
        _case_path(case_id, cases_dir).read_text(encoding="utf-8")
    )


def list_cases(cases_dir: Path | str | None = None) -> list[RegressionCase]:
    """Load every regression case, newest first."""
    dest_dir = Path(cases_dir) if cases_dir is not None else DEFAULT_CASES_DIR
    if not dest_dir.exists():
        return []
    cases: list[RegressionCase] = []
    for path in sorted(dest_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            cases.append(RegressionCase.model_validate_json(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return cases


def check_case(
    case: RegressionCase,
    candidate: Trajectory,
    judges: list[Judge] | None = None,
) -> TrajectoryDiff:
    """Check a candidate trajectory against a case's expected behaviour.

    Defaults to exact final-answer + strict tool-call judges, matching the
    default ``replayagent eval`` judges.
    """
    judges = judges if judges is not None else [ExactMatchJudge(), ToolCallJudge()]
    diff = diff_trajectories(case.expected_trajectory(), candidate, judges=judges)
    if case.expected_tool_calls:
        # The case's expected tool sequence is authoritative: it may have
        # been corrected by a reviewer and differ from the source run's.
        actual = candidate.tool_call_names()
        passed = actual == case.expected_tool_calls
        diff.judge_results.append(
            JudgeResult(
                judge="expected_tool_calls",
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=(
                    "candidate tool calls match the case expectation"
                    if passed
                    else f"expected tool calls {case.expected_tool_calls}, got {actual}"
                ),
                details={"expected": case.expected_tool_calls, "actual": actual},
            )
        )
    return diff

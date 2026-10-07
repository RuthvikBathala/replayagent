"""Diff: step-by-step comparison of a baseline vs a replayed trajectory."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from replayagent.judges import Judge, JudgeResult
from replayagent.trajectory import Step, Trajectory


class StepChange(BaseModel):
    """One differing step between baseline and candidate."""

    index: int
    kind: str  # "changed" | "added" | "removed"
    baseline: Step | None = None
    candidate: Step | None = None
    summary: str = ""


class TrajectoryDiff(BaseModel):
    """Full comparison result: step changes, tool-call deltas, judge verdicts."""

    baseline_run_id: str
    candidate_run_id: str
    step_changes: list[StepChange] = Field(default_factory=list)
    added_tool_calls: list[str] = Field(default_factory=list)
    removed_tool_calls: list[str] = Field(default_factory=list)
    judge_results: list[JudgeResult] = Field(default_factory=list)

    @property
    def steps_changed(self) -> int:
        return len(self.step_changes)

    @property
    def all_passed(self) -> bool:
        return all(j.passed for j in self.judge_results)

    @property
    def is_regression(self) -> bool:
        """A regression = any judge failed, or the tool-call sequence changed."""
        return (not self.all_passed) or bool(self.added_tool_calls or self.removed_tool_calls)


def _payloads_equal(a: Step, b: Step) -> bool:
    return a.type == b.type and a.payload == b.payload


def _summarize_change(kind: str, index: int, a: Step | None, b: Step | None) -> str:
    if kind == "added" and b is not None:
        return f"step {index}: added {b.type.value} {b.payload.get('name', '')}".rstrip()
    if kind == "removed" and a is not None:
        return f"step {index}: removed {a.type.value} {a.payload.get('name', '')}".rstrip()
    if a is not None and b is not None:
        if a.type != b.type:
            return f"step {index}: type changed {a.type.value} -> {b.type.value}"
        return f"step {index}: {a.type.value} payload changed"
    return f"step {index}: {kind}"


def diff_trajectories(
    baseline: Trajectory,
    candidate: Trajectory,
    judges: list[Judge] | None = None,
) -> TrajectoryDiff:
    """Compare two trajectories step-by-step and run judges over the pair."""
    changes: list[StepChange] = []
    n = max(len(baseline.steps), len(candidate.steps))
    for i in range(n):
        a = baseline.steps[i] if i < len(baseline.steps) else None
        b = candidate.steps[i] if i < len(candidate.steps) else None
        if a is None and b is None:
            continue
        if a is None:
            changes.append(
                StepChange(
                    index=i, kind="added", candidate=b, summary=_summarize_change("added", i, a, b)
                )
            )
        elif b is None:
            changes.append(
                StepChange(
                    index=i,
                    kind="removed",
                    baseline=a,
                    summary=_summarize_change("removed", i, a, b),
                )
            )
        elif not _payloads_equal(a, b):
            changes.append(
                StepChange(
                    index=i,
                    kind="changed",
                    baseline=a,
                    candidate=b,
                    summary=_summarize_change("changed", i, a, b),
                )
            )

    base_tools = baseline.tool_call_names()
    cand_tools = candidate.tool_call_names()
    added = [t for t in cand_tools if t not in base_tools]
    removed = [t for t in base_tools if t not in cand_tools]

    results: list[JudgeResult] = []
    for judge in judges or []:
        results.append(judge.evaluate(baseline, candidate))

    return TrajectoryDiff(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        step_changes=changes,
        added_tool_calls=added,
        removed_tool_calls=removed,
        judge_results=results,
    )


def diff_to_dict(diff: TrajectoryDiff) -> dict[str, Any]:
    """Serialize a diff to plain JSON-compatible data."""
    return diff.model_dump(mode="json")

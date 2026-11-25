"""Business logic for the hosted service, built on the replayagent core.

Submitting a run stores the tenant's trajectory; when a ``baseline_run_id``
(from the same tenant) is supplied, the core :func:`diff_trajectories` runs
with the default judges (tool-call sequence + exact final-answer match) and
the resulting diff is stored as the run's regression report.
"""

from __future__ import annotations

import json
from typing import Any

from replayagent.diff import diff_to_dict, diff_trajectories
from replayagent.hosted.store import Store
from replayagent.judges import ExactMatchJudge, ToolCallJudge
from replayagent.trajectory import Trajectory


class ServiceError(Exception):
    """An error that maps cleanly onto an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _default_judges() -> list:
    return [ToolCallJudge(), ExactMatchJudge()]


def _run_summary(row: dict[str, Any]) -> dict[str, Any]:
    traj = json.loads(row["trajectory_json"])
    diff = json.loads(row["diff_json"]) if row["diff_json"] else None
    return {
        "run_id": row["run_id"],
        "agent_name": row["agent_name"],
        "model": row["model"],
        "step_count": len(traj.get("steps", [])),
        "submitted_at": row["submitted_at"],
        "baseline_run_id": row["baseline_run_id"],
        "is_regression": diff["is_regression"] if diff else None,
    }


class HostedService:
    """Tenant-scoped operations over a :class:`Store`."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def create_tenant(self, name: str) -> dict[str, Any]:
        tenant_id, api_key = self.store.create_tenant(name)
        return {"tenant_id": tenant_id, "api_key": api_key}

    def submit_run(
        self,
        tenant_id: str,
        trajectory: Trajectory,
        baseline_run_id: str | None = None,
    ) -> dict[str, Any]:
        if self.store.run_exists(tenant_id, trajectory.run_id):
            raise ServiceError(409, f"run {trajectory.run_id!r} already submitted")
        diff_dict: dict[str, Any] | None = None
        if baseline_run_id is not None:
            base_row = self.store.get_run_row(tenant_id, baseline_run_id)
            if base_row is None:
                # Also the cross-tenant case: another tenant's run is
                # indistinguishable from a missing one — by design.
                raise ServiceError(400, f"baseline run {baseline_run_id!r} not found")
            baseline = Trajectory(**json.loads(base_row["trajectory_json"]))
            diff = diff_trajectories(baseline, trajectory, judges=_default_judges())
            diff_dict = diff_to_dict(diff)
            diff_dict["is_regression"] = diff.is_regression
            diff_dict["all_passed"] = diff.all_passed
            diff_dict["steps_changed"] = diff.steps_changed
        self.store.save_run(
            tenant_id,
            trajectory.model_dump(mode="json"),
            baseline_run_id,
            diff_dict,
        )
        row = self.store.get_run_row(tenant_id, trajectory.run_id)
        assert row is not None
        return _run_summary(row)

    def list_runs(self, tenant_id: str) -> list[dict[str, Any]]:
        return [_run_summary(r) for r in self.store.list_run_rows(tenant_id)]

    def get_run(self, tenant_id: str, run_id: str) -> dict[str, Any]:
        row = self.store.get_run_row(tenant_id, run_id)
        if row is None:
            raise ServiceError(404, f"run {run_id!r} not found")
        return json.loads(row["trajectory_json"])

    def get_report(self, tenant_id: str, run_id: str) -> dict[str, Any]:
        row = self.store.get_run_row(tenant_id, run_id)
        if row is None:
            raise ServiceError(404, f"run {run_id!r} not found")
        summary = _run_summary(row)
        if row["diff_json"]:
            return {**summary, "diff": json.loads(row["diff_json"])}
        traj = json.loads(row["trajectory_json"])
        return {
            **summary,
            "diff": None,
            "note": "No baseline was supplied for this run, so no diff was computed. "
            "Submit a run with baseline_run_id to get a regression report.",
            "tool_calls": [
                str(s.get("payload", {}).get("name", ""))
                for s in traj.get("steps", [])
                if s.get("type") == "tool_call"
            ],
        }

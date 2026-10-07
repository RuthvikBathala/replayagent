"""Replay: re-execute an agent against a recorded run's inputs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from replayagent.recorder import DEFAULT_STORE_DIR, Recorder, load_run, save_trajectory
from replayagent.trajectory import Trajectory

# An agent function receives the recorded input dict and a fresh Recorder,
# and is expected to record its steps (llm_call / tool_call / ...) on it.
AgentFn = Callable[[dict[str, Any], Recorder], Any]


def replay(
    baseline: Trajectory | str,
    agent_fn: AgentFn,
    *,
    model: str | None = None,
    agent_name: str | None = None,
    store_dir: Path | str | None = None,
    autosave: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    """Re-run ``agent_fn`` against a recorded run's input, capturing a new trajectory.

    Args:
        baseline: A recorded :class:`Trajectory`, or a run id to load from ``store_dir``.
        agent_fn: Callable ``(input_dict, recorder) -> Any`` that runs the agent
            (possibly with a new model or config) and records steps on the recorder.
        model: Model label for the replayed run (defaults to the baseline's).
        agent_name: Agent label for the replayed run (defaults to the baseline's).
        store_dir: Where to persist the replayed run.
        autosave: Persist the replayed trajectory on completion.
        metadata: Extra metadata attached to the replayed trajectory.
    """
    if isinstance(baseline, str):
        baseline = load_run(baseline, store_dir=store_dir or DEFAULT_STORE_DIR)

    rec = Recorder(
        agent_name or baseline.agent_name,
        model=model or baseline.model,
        input=dict(baseline.input),
        store_dir=store_dir or DEFAULT_STORE_DIR,
        autosave=autosave,
    )
    agent_fn(dict(baseline.input), rec)
    traj = rec.trajectory
    if metadata:
        traj.metadata.update(metadata)
    traj.metadata["replayed_from"] = baseline.run_id
    if autosave:
        save_trajectory(traj, Path(store_dir or DEFAULT_STORE_DIR) / f"{traj.run_id}.jsonl")
    return traj

"""Recording API: capture agent trajectories to disk as JSONL."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from replayagent.trajectory import Step, StepType, Trajectory

DEFAULT_STORE_DIR = Path(".replayagent") / "runs"


class Recorder(AbstractContextManager["Recorder"]):
    """Capture steps of one agent run and persist them as a trajectory.

    Use as a context manager (auto-saves on exit) or manually::

        with record("support-agent", model="gpt-4o", input={"q": "reset my password"}) as rec:
            rec.llm_call(messages=[...], response="...", model="gpt-4o")
            rec.tool_call(name="reset_password", arguments={"user": "amy"})
            rec.tool_result(name="reset_password", result={"ok": True})
            rec.final_answer("Done — password reset.")
    """

    def __init__(
        self,
        agent_name: str,
        model: str = "unknown",
        input: dict[str, Any] | None = None,
        run_id: str | None = None,
        store_dir: Path | str | None = None,
        autosave: bool = True,
    ) -> None:
        self.agent_name = agent_name
        self.model = model
        self.input = dict(input or {})
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.store_dir = Path(store_dir) if store_dir is not None else DEFAULT_STORE_DIR
        self.autosave = autosave
        self._steps: list[Step] = []
        self._saved_path: Path | None = None

    # -- context manager -------------------------------------------------
    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.autosave:
            self.save()

    # -- step capture ----------------------------------------------------
    def record_step(
        self,
        step_type: StepType | str,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Step:
        step = Step(
            type=StepType(step_type),
            timestamp=datetime.now(timezone.utc),
            payload=dict(payload or {}),
            metadata=dict(metadata or {}),
        )
        self._steps.append(step)
        return step

    def llm_call(
        self,
        messages: list[dict[str, Any]] | None = None,
        response: Any = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        **extra: Any,
    ) -> Step:
        return self.record_step(
            StepType.LLM_CALL,
            payload={
                "messages": messages or [],
                "response": response,
                "model": model or self.model,
                "usage": usage or {},
                **extra,
            },
        )

    def tool_call(self, name: str, arguments: dict[str, Any] | None = None, **extra: Any) -> Step:
        return self.record_step(
            StepType.TOOL_CALL,
            payload={"name": name, "arguments": arguments or {}, **extra},
        )

    def tool_result(self, name: str, result: Any, **extra: Any) -> Step:
        return self.record_step(
            StepType.TOOL_RESULT,
            payload={"name": name, "result": result, **extra},
        )

    def final_answer(self, text: Any, **extra: Any) -> Step:
        return self.record_step(StepType.FINAL_ANSWER, payload={"text": text, **extra})

    # -- persistence ------------------------------------------------------
    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(
            run_id=self.run_id,
            agent_name=self.agent_name,
            model=self.model,
            input=self.input,
            steps=list(self._steps),
        )

    def save(self, path: Path | str | None = None) -> Path:
        """Persist the trajectory as JSONL (one object per line: header + steps)."""
        dest = Path(path) if path else self.store_dir / f"{self.run_id}.jsonl"
        self._saved_path = save_trajectory(self.trajectory, dest)
        return self._saved_path

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)


def record(
    agent_name: str,
    model: str = "unknown",
    input: dict[str, Any] | None = None,
    store_dir: Path | str | None = None,
    **kwargs: Any,
) -> Recorder:
    """Create a :class:`Recorder` for one agent run."""
    return Recorder(agent_name, model=model, input=input, store_dir=store_dir, **kwargs)


def record_step(
    recorder: Recorder,
    step_type: StepType | str,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Step:
    """Manual step-recording helper for an existing recorder."""
    return recorder.record_step(step_type, payload=payload, metadata=metadata)


def save_trajectory(traj: Trajectory, path: Path | str) -> Path:
    """Write a trajectory to disk as JSONL (header line + one line per step)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"kind": "trajectory", **traj.model_dump(mode="json", exclude={"steps"})})
            + "\n"
        )
        for step in traj.steps:
            fh.write(json.dumps({"kind": "step", **step.model_dump(mode="json")}) + "\n")
    return dest


def load_run(run_id: str, store_dir: Path | str | None = None) -> Trajectory:
    """Load a recorded trajectory by run id."""
    store = Path(store_dir) if store_dir is not None else DEFAULT_STORE_DIR
    path = store / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No recorded run {run_id!r} in {store}")
    header: dict[str, Any] = {}
    steps: list[Step] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "trajectory":
                header = {k: v for k, v in obj.items() if k != "kind"}
            elif obj.get("kind") == "step":
                steps.append(Step(**{k: v for k, v in obj.items() if k != "kind"}))
    if not header:
        raise ValueError(f"Corrupt run file: {path} (missing trajectory header)")
    return Trajectory(steps=steps, **header)


def list_runs(store_dir: Path | str | None = None) -> list[Trajectory]:
    """Load lightweight headers for every recorded run, newest first."""
    store = Path(store_dir) if store_dir is not None else DEFAULT_STORE_DIR
    if not store.exists():
        return []
    runs: list[Trajectory] = []
    for path in sorted(store.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            runs.append(load_run(path.stem, store_dir=store))
        except (ValueError, json.JSONDecodeError):
            continue
    return runs


def traced(agent_name: str | None = None, model: str = "unknown"):
    """Decorator that records a function's agent run.

    The wrapped function receives a ``recorder`` keyword argument::

        @traced("my-agent", model="gpt-4o")
        def my_agent(query: str, recorder: Recorder) -> str:
            recorder.llm_call(...)
            ...
            recorder.final_answer("...")
            return "..."
    """

    def decorator(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            name = agent_name or fn.__name__
            with record(name, model=model) as rec:
                kwargs["recorder"] = rec
                result = fn(*args, **kwargs)
            return result

        return wrapper

    return decorator

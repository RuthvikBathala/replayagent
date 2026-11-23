"""Production-trace ingestion -> failure clustering -> draft regression cases.

The pipeline that feeds the existing promote-to-case flow in
:mod:`replayagent.cases`:

1. **Ingest** a JSON upload of production traces (Langfuse/OTel-shaped, see
   the accepted schema below) and convert each trace into a normal
   :class:`~replayagent.trajectory.Trajectory`, so it can be saved to the
   run store and promoted exactly like a locally recorded run.
2. **Detect + cluster failures**: a run failed when it has an explicit
   error status, any step carries an error, or — when an expected behaviour
   was recorded — its final answer does not match it (a deterministic,
   offline "judge"). Failures are grouped by ``(failing tool, normalised
   error signature)`` so one root cause shows up once, with a count.
3. **Auto-draft regression cases**: each cluster drafts a case in the exact
   :class:`~replayagent.cases.RegressionCase` format, pre-filled from a
   representative failed trace. Drafts live under ``.replayagent/drafts/``
   and are marked ``metadata["draft"] = True`` — they are *not* cases yet.
   A human reviews/edits a draft (especially the expected answer, which for
   a failed run is usually wrong), then ``promote_draft`` sends it through
   the existing :func:`replayagent.cases.promote_trajectory` flow.

Accepted trace schema (one JSON file: a list, or ``{"traces": [...]}``)::

    {
      "id": "run-123",                    # also: run_id / trace_id / traceId
      "name": "support-agent",            # also: agent_name (default "unknown")
      "model": "gpt-4o",                  # optional, default "unknown"
      "input": {"query": "..."},          # dict, or a string -> {"query": str}
      "status": "success" | "error",      # also: level ("ERROR" = failed)
      "error": "optional top-level error message",
      "expected": "optional expected final answer / behaviour",
      "steps": [                          # also: observations / events
        {"type": "llm_call", "response": "..."},
        {"type": "tool_call", "name": "search", "arguments": {"q": "..."}},
        {"type": "tool_result", "name": "search", "result": {...},
         "error": "optional step error", "status": "error"},
        {"type": "final_answer", "text": "..."}
      ]
    }

Langfuse-flavoured steps are accepted too: ``GENERATION`` -> ``llm_call``,
``TOOL`` -> ``tool_call`` (+ result when an output is present). No live LLM
calls are made anywhere in this module; the judge seam
(:class:`FailureJudge`) has a deterministic default.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from replayagent.cases import RegressionCase, promote_trajectory, save_case
from replayagent.recorder import save_trajectory
from replayagent.trajectory import Step, StepType, Trajectory

DEFAULT_DRAFTS_DIR = Path(".replayagent") / "drafts"

_FAILED_STATUSES = {"error", "failed", "failure", "fail", "crashed", "timeout", "errored"}


class IngestError(ValueError):
    """Raised when a production-trace upload does not match the accepted schema."""


class ProductionTrace(BaseModel):
    """One ingested production trace, before/independent of storage."""

    run_id: str
    agent_name: str = "unknown"
    model: str = "unknown"
    input: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    status: str | None = None
    error: str | None = None
    expected_behavior: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_trajectory(self) -> Trajectory:
        meta = dict(self.metadata)
        if self.status is not None:
            meta.setdefault("production_status", self.status)
        if self.error:
            meta.setdefault("production_error", self.error)
        if self.expected_behavior:
            meta.setdefault("expected_behavior", self.expected_behavior)
        return Trajectory(
            run_id=self.run_id,
            agent_name=self.agent_name,
            model=self.model,
            input=dict(self.input),
            steps=list(self.steps),
            metadata=meta,
        )

    def final_answer(self) -> str | None:
        return self.to_trajectory().final_answer()


class Failure(BaseModel):
    """One detected failure: why it failed, and the cluster key parts."""

    run_id: str
    reasons: list[str] = Field(default_factory=list)
    failing_tool: str | None = None
    error_signature: str = ""
    error_message: str = ""


class FailureCluster(BaseModel):
    """Failures sharing (failing tool, error signature), with a count."""

    cluster_id: str
    failing_tool: str | None = None
    error_signature: str = ""
    count: int = 0
    run_ids: list[str] = Field(default_factory=list)
    representative_run_id: str = ""
    reasons: list[str] = Field(default_factory=list)


class FailureJudge(Protocol):
    """Seam for judging a trace against its expected behaviour.

    Return True (failed), False (passed), or None (not evaluable — e.g. no
    expected behaviour was recorded). Implementations may wrap an LLM, but
    the default is deterministic and offline.
    """

    def is_failure(self, trace: ProductionTrace) -> bool | None: ...


class ExpectedBehaviorJudge:
    """Deterministic default judge: exact final-answer vs expected behaviour."""

    def is_failure(self, trace: ProductionTrace) -> bool | None:
        if not trace.expected_behavior:
            return None
        answer = trace.final_answer()
        if answer is None:
            return True
        return answer.strip() != trace.expected_behavior.strip()


# -- parsing ----------------------------------------------------------------


def _first(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if d.get(key) is not None:
            return d[key]
    return None


def _as_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": value}
        except json.JSONDecodeError:
            return {"value": value}
    return {} if value is None else {"value": value}


def _parse_step(raw: Any, index: int) -> Step:
    if not isinstance(raw, dict):
        raise IngestError(f"step {index} is not an object: {raw!r}")
    raw_type = str(_first(raw, "type", "kind") or "").lower()
    name = _first(raw, "name", "tool", "tool_name")
    error = _first(raw, "error", "error_message")
    status = str(_first(raw, "status", "level") or "").lower()
    if status in _FAILED_STATUSES and error is None:
        error = _first(raw, "message", "output") or f"step status: {status}"

    if raw_type in ("llm_call", "generation", "llm", "chat", "completion"):
        payload: dict[str, Any] = {
            "messages": raw.get("messages") or [],
            "response": _first(raw, "response", "output", "content"),
            "model": raw.get("model"),
        }
        step_type = StepType.LLM_CALL
    elif raw_type in ("tool_call", "tool-call", "toolcall"):
        payload = {
            "name": name or "",
            "arguments": _as_arguments(_first(raw, "arguments", "args", "input")),
        }
        step_type = StepType.TOOL_CALL
    elif raw_type in ("tool_result", "tool-result", "toolresult"):
        payload = {"name": name or "", "result": _first(raw, "result", "output")}
        step_type = StepType.TOOL_RESULT
    elif raw_type in ("tool",):
        # Langfuse TOOL observation: a call, plus its result when output exists.
        payload = {"name": name or "", "arguments": _as_arguments(raw.get("input"))}
        step_type = StepType.TOOL_CALL
        if raw.get("output") is not None and error is None:
            # The paired result is appended by the caller via a second step;
            # here we keep the call only and stash the output for detection.
            payload["result"] = raw["output"]
    elif raw_type in ("final_answer", "final", "answer"):
        payload = {"text": _first(raw, "text", "output", "answer", "content")}
        step_type = StepType.FINAL_ANSWER
    elif error is not None or name:
        step_type = StepType.TOOL_RESULT
        payload = {"name": name or "", "result": _first(raw, "result", "output")}
    else:
        step_type = StepType.LLM_CALL
        payload = {"messages": [], "response": _first(raw, "output", "content", "text")}
    if error is not None:
        payload["error"] = str(error)
    if status:
        payload.setdefault("status", status)
    step = Step(type=step_type, payload=payload)
    ts = _first(raw, "timestamp", "startTime", "start_time")
    if isinstance(ts, str):
        try:
            step.timestamp = Step.model_validate({"type": step_type, "timestamp": ts}).timestamp
        except ValueError:
            pass
    return step


def parse_trace(raw: Any, index: int = 0) -> ProductionTrace:
    """Parse one raw trace dict into a :class:`ProductionTrace`."""
    if not isinstance(raw, dict):
        raise IngestError(f"trace at index {index} is not an object: {raw!r}")
    run_id = _first(raw, "run_id", "id", "trace_id", "traceId")
    if not run_id:
        raise IngestError(f"trace at index {index} is missing a run id (id/run_id/trace_id)")
    raw_input = _first(raw, "input", "input_data", "inputData")
    if isinstance(raw_input, dict):
        input_data: dict[str, Any] = raw_input
    elif raw_input is None:
        input_data = {}
    elif isinstance(raw_input, str):
        input_data = {"query": raw_input}
    else:
        input_data = {"input": raw_input}
    raw_steps = _first(raw, "steps", "observations", "events") or []
    if not isinstance(raw_steps, list):
        raise IngestError(f"trace {run_id!r}: steps must be a list")
    steps = [_parse_step(s, i) for i, s in enumerate(raw_steps)]
    status = _first(raw, "status", "level")
    error = _first(raw, "error", "error_message")
    if isinstance(error, dict):
        error = error.get("message") or json.dumps(error)
    expected = _first(
        raw, "expected", "expected_behavior", "expected_output", "expected_final_answer"
    )
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    model = _first(raw, "model") or metadata.get("model") or "unknown"
    return ProductionTrace(
        run_id=str(run_id),
        agent_name=str(_first(raw, "agent_name", "name", "agent") or "unknown"),
        model=str(model),
        input=input_data,
        steps=steps,
        status=str(status).lower() if status is not None else None,
        error=str(error) if error is not None else None,
        expected_behavior=str(expected) if expected is not None else None,
        metadata=dict(metadata),
    )


def parse_traces(data: Any) -> list[ProductionTrace]:
    """Parse a whole upload: a list, or a dict with a traces/data/items list."""
    if isinstance(data, dict):
        for key in ("traces", "data", "items", "runs"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]  # a single trace object
    if not isinstance(data, list):
        raise IngestError("trace upload must be a JSON list or an object with a 'traces' list")
    return [parse_trace(raw, i) for i, raw in enumerate(data)]


def load_traces(path: Path | str) -> list[ProductionTrace]:
    """Load and parse a JSON trace upload from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No trace file {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{p} is not valid JSON: {exc}") from exc
    return parse_traces(data)


def save_ingested(traces: list[ProductionTrace], store_dir: Path | str | None = None) -> list[Path]:
    """Save ingested traces into the run store so promote/eval can use them."""
    from replayagent.recorder import DEFAULT_STORE_DIR

    store = Path(store_dir) if store_dir is not None else DEFAULT_STORE_DIR
    return [save_trajectory(t.to_trajectory(), store / f"{t.run_id}.jsonl") for t in traces]


# -- failure detection + clustering ------------------------------------------


def _step_error(step: Step) -> str | None:
    err = step.payload.get("error")
    if err:
        return str(err)
    status = str(step.payload.get("status") or "").lower()
    if status in _FAILED_STATUSES:
        return f"step status: {status}"
    result = step.payload.get("result")
    if isinstance(result, dict):
        if result.get("error"):
            return str(result["error"])
        if result.get("ok") is False:
            return "tool result ok=false"
    return None


def normalize_signature(message: str) -> str:
    """Normalise an error message into a stable clustering signature."""
    sig = message.lower()
    sig = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "#", sig)  # uuids
    sig = re.sub(r"\b[0-9a-f]{16,}\b", "#", sig)  # hex ids
    sig = re.sub(r"\d+", "#", sig)  # numbers
    sig = re.sub(r"[^a-z# ]+", " ", sig)
    return " ".join(sig.split())[:80].strip() or "unknown error"


def detect_failure(trace: ProductionTrace, judge: FailureJudge | None = None) -> Failure | None:
    """Return a :class:`Failure` for a failed trace, else None. Deterministic."""
    judge = judge if judge is not None else ExpectedBehaviorJudge()
    reasons: list[str] = []
    message = ""
    failing_tool: str | None = None
    if trace.status in _FAILED_STATUSES:
        reasons.append(f"status:{trace.status}")
        message = trace.error or f"run status {trace.status}"
    if trace.error:
        if "explicit-error" not in reasons:
            reasons.append("explicit-error")
        message = message or trace.error
    for step in trace.steps:
        err = _step_error(step)
        if err is not None:
            reasons.append("step-error")
            if failing_tool is None:
                failing_tool = str(step.payload.get("name") or "") or None
            message = message or err
            break
    verdict = judge.is_failure(trace)
    if verdict:
        reasons.append("judge-failure")
        message = message or "final answer does not match expected behaviour"
    if not reasons:
        return None
    if failing_tool is None:
        names = trace.to_trajectory().tool_call_names()
        failing_tool = names[-1] if names else None
    return Failure(
        run_id=trace.run_id,
        reasons=reasons,
        failing_tool=failing_tool,
        error_signature=normalize_signature(message),
        error_message=message,
    )


def detect_failures(
    traces: list[ProductionTrace], judge: FailureJudge | None = None
) -> list[Failure]:
    out = []
    for trace in traces:
        failure = detect_failure(trace, judge=judge)
        if failure is not None:
            out.append(failure)
    return out


def _cluster_id(failing_tool: str | None, signature: str) -> str:
    digest = hashlib.sha1(f"{failing_tool or ''}|{signature}".encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", (failing_tool or "no-tool").lower()).strip("-") or "no-tool"
    return f"{slug}-{digest}"


def cluster_failures(failures: list[Failure]) -> list[FailureCluster]:
    """Group failures by (failing tool, error signature), biggest first."""
    groups: dict[tuple[str | None, str], list[Failure]] = {}
    for failure in failures:
        groups.setdefault((failure.failing_tool, failure.error_signature), []).append(failure)
    clusters = []
    for (tool, signature), members in groups.items():
        members = sorted(members, key=lambda f: f.run_id)
        reasons: list[str] = []
        for m in members:
            for r in m.reasons:
                if r not in reasons:
                    reasons.append(r)
        clusters.append(
            FailureCluster(
                cluster_id=_cluster_id(tool, signature),
                failing_tool=tool,
                error_signature=signature,
                count=len(members),
                run_ids=[m.run_id for m in members],
                representative_run_id=members[0].run_id,
                reasons=reasons,
            )
        )
    return sorted(clusters, key=lambda c: (-c.count, c.cluster_id))


def cluster_traces(
    traces: list[ProductionTrace], judge: FailureJudge | None = None
) -> list[FailureCluster]:
    """Detect failures in traces and cluster them in one call."""
    return cluster_failures(detect_failures(traces, judge=judge))


# -- draft regression cases ---------------------------------------------------


def draft_case_for_cluster(
    cluster: FailureCluster,
    traces: list[ProductionTrace],
    drafts_dir: Path | str | None = None,
) -> RegressionCase:
    """Draft (and save) a regression case for one cluster. Needs human review.

    The draft is a normal :class:`RegressionCase` saved under the drafts
    directory, pre-filled from the representative failed trace, with
    ``metadata`` marking it as an unreviewed draft.
    """
    by_id = {t.run_id: t for t in traces}
    rep = by_id[cluster.representative_run_id]
    traj = rep.to_trajectory()
    draft = RegressionCase(
        case_id=f"draft-{cluster.cluster_id}",
        name=f"[DRAFT] {cluster.failing_tool or 'no-tool'}: {cluster.error_signature}"[:120],
        source_run_id=rep.run_id,
        agent_name=rep.agent_name,
        model=rep.model,
        input=dict(rep.input),
        reason=(
            f"Auto-drafted from production failure cluster {cluster.cluster_id} "
            f"({cluster.count} run(s): {', '.join(cluster.run_ids)}). "
            "REVIEW REQUIRED: confirm/correct the expected behaviour before promoting."
        ),
        # Pre-fill from the trace's recorded expectation when there is one;
        # otherwise the failed run's own answer is kept as a placeholder the
        # reviewer must correct — a failed run's answer is rarely right.
        expected_final_answer=rep.expected_behavior or traj.final_answer(),
        expected_tool_calls=traj.tool_call_names(),
        baseline=traj,
        metadata={
            "draft": True,
            "status": "draft",
            "needs_review": True,
            "source": "production_ingest",
            "cluster_id": cluster.cluster_id,
            "cluster_count": cluster.count,
            "cluster_run_ids": list(cluster.run_ids),
            "error_signature": cluster.error_signature,
            "failing_tool": cluster.failing_tool,
            "detection_reasons": list(cluster.reasons),
        },
    )
    save_case(draft, cases_dir=drafts_dir if drafts_dir is not None else DEFAULT_DRAFTS_DIR)
    return draft


def draft_cases(
    clusters: list[FailureCluster],
    traces: list[ProductionTrace],
    drafts_dir: Path | str | None = None,
) -> list[RegressionCase]:
    """Draft one regression case per cluster."""
    return [draft_case_for_cluster(c, traces, drafts_dir=drafts_dir) for c in clusters]


def list_drafts(drafts_dir: Path | str | None = None) -> list[RegressionCase]:
    """List saved draft cases, newest first."""
    from replayagent.cases import list_cases

    return list_cases(drafts_dir if drafts_dir is not None else DEFAULT_DRAFTS_DIR)


def load_draft(draft_id: str, drafts_dir: Path | str | None = None) -> RegressionCase:
    """Load one draft case by id."""
    from replayagent.cases import load_case

    return load_case(draft_id, drafts_dir if drafts_dir is not None else DEFAULT_DRAFTS_DIR)


def promote_draft(
    draft_id: str,
    drafts_dir: Path | str | None = None,
    cases_dir: Path | str | None = None,
    expected_final_answer: str | None = None,
    expected_tool_calls: list[str] | None = None,
    reason: str | None = None,
) -> RegressionCase:
    """Promote a reviewed draft through the existing promote-to-case flow.

    Reviewer overrides (expected answer / tool calls / reason) win over the
    draft's pre-filled values. The resulting case is a normal case in the
    cases directory — the draft file is left in place for provenance.
    """
    draft = load_draft(draft_id, drafts_dir=drafts_dir)
    name = draft.name.removeprefix("[DRAFT] ").strip() or draft.name
    return promote_trajectory(
        draft.baseline,
        name=name,
        reason=reason if reason is not None else draft.reason,
        expected_final_answer=(
            expected_final_answer
            if expected_final_answer is not None
            else draft.expected_final_answer
        ),
        expected_tool_calls=(
            expected_tool_calls if expected_tool_calls is not None else draft.expected_tool_calls
        ),
        cases_dir=cases_dir,
    )

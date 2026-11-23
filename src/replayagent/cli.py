"""replayagent CLI: record, replay, eval, and report on agent trajectories."""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
import yaml
from rich.console import Console

from replayagent.cases import check_case, list_cases, load_case, promote_run
from replayagent.diff import diff_trajectories
from replayagent.ingest import (
    IngestError,
    cluster_traces,
    draft_cases,
    list_drafts,
    load_traces,
    promote_draft,
    save_ingested,
)
from replayagent.judges import Judge, get_judge
from replayagent.recorder import DEFAULT_STORE_DIR, Recorder, list_runs, load_run
from replayagent.recorder import record as new_recording
from replayagent.replay import AgentFn
from replayagent.replay import replay as replay_run
from replayagent.reporters import html_report, json_report, junit_report, terminal_report

app = typer.Typer(
    help="Record, replay, and regression-test AI agent trajectories.", no_args_is_help=True
)
console = Console()

DEFAULT_JUDGES = ["exact", "tools"]


def _resolve_store(store_dir: Path | None) -> Path:
    return store_dir if store_dir is not None else DEFAULT_STORE_DIR


def _demo_agent(input_data: dict, recorder: Recorder, model: str = "demo-model") -> str:
    """Built-in demo agent used by `record` and `replay` when no --agent is given."""
    query = str(input_data.get("query", "hello"))
    recorder.llm_call(
        messages=[{"role": "user", "content": query}],
        response=f"Thinking about: {query}",
        model=model,
    )
    recorder.tool_call(name="search", arguments={"q": query})
    recorder.tool_result(name="search", result={"hits": [f"result for {query}"]})
    answer = f"Answer to '{query}' (via {model})"
    recorder.final_answer(answer)
    return answer


def _load_agent_fn(spec: str | None) -> AgentFn:
    if not spec:
        return _demo_agent
    if ":" not in spec:
        raise typer.BadParameter("--agent must look like 'mymodule:function'")
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise typer.BadParameter(f"cannot import module {module_name!r}: {exc}")
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise typer.BadParameter(f"{spec!r} is not a callable (input_dict, recorder) function")
    return fn  # type: ignore[return-value]


def _build_judges(names: list[str]) -> list[Judge]:
    return [get_judge(name) for name in names]


def _run_eval(
    baseline_id: str, candidate_id: str, judge_names: list[str], store_dir: Path | None
) -> int:
    baseline = load_run(baseline_id, store_dir)
    candidate = load_run(candidate_id, store_dir)
    judges = _build_judges(judge_names)
    diff = diff_trajectories(baseline, candidate, judges=judges)
    terminal_report(diff, console)
    return 1 if diff.is_regression else 0


@app.command()
def init(
    path: Path = typer.Option(Path("."), "--path", help="Project directory to initialize."),
) -> None:
    """Initialize replayagent in a project (creates .replayagent/ + config)."""
    store = path / DEFAULT_STORE_DIR
    store.mkdir(parents=True, exist_ok=True)
    (path / ".replayagent" / "cases").mkdir(parents=True, exist_ok=True)
    config = path / ".replayagent" / "config.yaml"
    if not config.exists():
        config.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "store_dir": str(DEFAULT_STORE_DIR),
                    "default_judges": DEFAULT_JUDGES,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    console.print(f"[green]Initialized replayagent[/green] in {path.resolve()} (.replayagent/)")


@app.command(name="list")
def list_cmd(
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """List recorded runs."""
    runs = list_runs(store_dir)
    if not runs:
        console.print("[dim]No recorded runs yet. Try `replayagent record` for a demo.[/dim]")
        return
    for r in runs:
        console.print(
            f"[bold]{r.run_id}[/bold]  {r.agent_name}  [dim]{r.model}[/dim]  "
            f"{len(r.steps)} steps  {r.created_at:%Y-%m-%d %H:%M}"
        )


@app.command()
def record(
    agent_name: str = typer.Option("demo-agent", "--agent-name", help="Agent name label."),
    model: str = typer.Option("demo-model", "--model", help="Model label."),
    query: str = typer.Option(
        "What is the capital of France?", "--query", help="Demo input query."
    ),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """Record a demo trajectory (no API keys needed)."""
    with new_recording(agent_name, model=model, input={"query": query}, store_dir=store_dir) as rec:
        _demo_agent({"query": query}, rec, model=model)
        run_id = rec.run_id
    console.print(f"[green]Recorded run[/green] [bold]{run_id}[/bold] ({agent_name} / {model})")


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Baseline run id to replay."),
    agent: str | None = typer.Option(
        None, "--agent", help="Agent as 'module:function'. Defaults to the demo agent."
    ),
    model: str | None = typer.Option(None, "--model", help="Model label for the replayed run."),
    judge: list[str] = typer.Option(DEFAULT_JUDGES, "--judge", help="Judges to run after replay."),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """Replay a recorded run's input with an agent and diff the result."""
    if agent is None:
        run_model = model or "demo-model"

        def agent_fn(input_data: dict, recorder: Recorder) -> str:
            return _demo_agent(input_data, recorder, model=run_model)

    else:
        agent_fn = _load_agent_fn(agent)
    new_traj = replay_run(run_id, agent_fn, model=model, store_dir=store_dir)
    console.print(f"[green]Replayed run[/green] [bold]{new_traj.run_id}[/bold] (from {run_id})")
    code = _run_eval(run_id, new_traj.run_id, judge, store_dir)
    raise typer.Exit(code=code)


@app.command()
def eval(
    run_id: str = typer.Argument(..., help="Baseline run id."),
    candidate: str = typer.Option(..., "--candidate", help="Replayed run id to compare."),
    judge: list[str] = typer.Option(DEFAULT_JUDGES, "--judge", help="Judges to run."),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """Evaluate a replayed run against its baseline. Exits 1 on regression (CI-friendly)."""
    code = _run_eval(run_id, candidate, judge, store_dir)
    raise typer.Exit(code=code)


@app.command()
def promote(
    run_id: str = typer.Argument(..., help="Recorded run id to promote."),
    name: str | None = typer.Option(None, "--name", help="Case name (defaults to agent + run)."),
    reason: str = typer.Option("", "--reason", help="Why this run matters (e.g. the failure)."),
    expected_answer: str | None = typer.Option(
        None, "--expected-answer", help="Correct final answer, if the run's own was wrong."
    ),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
    cases_dir: Path | None = typer.Option(None, "--cases-dir", help="Regression cases directory."),
) -> None:
    """Promote a recorded run into a reusable regression case."""
    try:
        case = promote_run(
            run_id,
            store_dir=store_dir,
            name=name,
            reason=reason,
            expected_final_answer=expected_answer,
            cases_dir=cases_dir,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc))
    console.print(
        f"[green]Promoted run[/green] [bold]{run_id}[/bold] to regression case "
        f"[bold]{case.case_id}[/bold] ({case.name})"
    )


@app.command(name="cases")
def cases_cmd(
    cases_dir: Path | None = typer.Option(None, "--cases-dir", help="Regression cases directory."),
) -> None:
    """List regression cases."""
    cases = list_cases(cases_dir)
    if not cases:
        console.print("[dim]No regression cases yet. Try `replayagent promote <run-id>`.[/dim]")
        return
    for c in cases:
        console.print(
            f"[bold]{c.case_id}[/bold]  {c.name}  [dim]from run {c.source_run_id}[/dim]  "
            f"{c.created_at:%Y-%m-%d %H:%M}"
        )


@app.command(name="eval-case")
def eval_case(
    case_id: str = typer.Argument(..., help="Regression case id."),
    candidate: str = typer.Option(..., "--candidate", help="Run id to check against the case."),
    judge: list[str] = typer.Option(DEFAULT_JUDGES, "--judge", help="Judges to run."),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
    cases_dir: Path | None = typer.Option(None, "--cases-dir", help="Regression cases directory."),
) -> None:
    """Evaluate a run against a regression case. Exits 1 on regression (CI-friendly)."""
    try:
        case = load_case(case_id, cases_dir)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc))
    cand = load_run(candidate, store_dir)
    diff = check_case(case, cand, judges=_build_judges(judge))
    terminal_report(diff, console)
    raise typer.Exit(code=1 if diff.is_regression else 0)


@app.command()
def ingest(
    file: Path = typer.Argument(..., help="JSON file of production traces (see ingest schema)."),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """Ingest production traces into the run store; report failures found."""
    try:
        traces = load_traces(file)
    except (FileNotFoundError, IngestError) as exc:
        raise typer.BadParameter(str(exc))
    save_ingested(traces, store_dir=store_dir)
    clusters = cluster_traces(traces)
    failed = sum(c.count for c in clusters)
    console.print(
        f"[green]Ingested[/green] [bold]{len(traces)}[/bold] trace(s) into the run store; "
        f"{failed} failure(s) in {len(clusters)} cluster(s)."
    )


@app.command()
def clusters(
    file: Path = typer.Argument(..., help="JSON file of production traces."),
) -> None:
    """List failure clusters detected in a production-trace upload."""
    try:
        traces = load_traces(file)
    except (FileNotFoundError, IngestError) as exc:
        raise typer.BadParameter(str(exc))
    found = cluster_traces(traces)
    if not found:
        console.print("[dim]No failures detected in this upload.[/dim]")
        return
    for c in found:
        console.print(
            f"[bold]{c.cluster_id}[/bold]  count={c.count}  tool={c.failing_tool or '-'}  "
            f"[dim]{c.error_signature}[/dim]  runs: {', '.join(c.run_ids)}"
        )


@app.command(name="draft-cases")
def draft_cases_cmd(
    file: Path = typer.Argument(..., help="JSON file of production traces."),
    drafts_dir: Path | None = typer.Option(None, "--drafts-dir", help="Draft cases directory."),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """Draft a regression case per failure cluster (drafts need human review)."""
    try:
        traces = load_traces(file)
    except (FileNotFoundError, IngestError) as exc:
        raise typer.BadParameter(str(exc))
    save_ingested(traces, store_dir=store_dir)
    drafts = draft_cases(cluster_traces(traces), traces, drafts_dir=drafts_dir)
    if not drafts:
        console.print("[dim]No failures detected — no drafts generated.[/dim]")
        return
    for d in drafts:
        console.print(
            f"[yellow]DRAFT[/yellow] [bold]{d.case_id}[/bold]  {d.name}  "
            f"[dim]from run {d.source_run_id} — review before promoting[/dim]"
        )


@app.command(name="drafts")
def drafts_cmd(
    drafts_dir: Path | None = typer.Option(None, "--drafts-dir", help="Draft cases directory."),
) -> None:
    """List auto-drafted regression cases awaiting review."""
    drafts = list_drafts(drafts_dir)
    if not drafts:
        console.print("[dim]No draft cases yet. Try `replayagent draft-cases <traces.json>`.[/dim]")
        return
    for d in drafts:
        console.print(
            f"[bold]{d.case_id}[/bold]  {d.name}  [dim]from run {d.source_run_id}[/dim]"
        )


@app.command(name="promote-draft")
def promote_draft_cmd(
    draft_id: str = typer.Argument(..., help="Draft case id to promote after review."),
    expected_answer: str | None = typer.Option(
        None, "--expected-answer", help="Correct final answer, if the draft's is wrong."
    ),
    drafts_dir: Path | None = typer.Option(None, "--drafts-dir", help="Draft cases directory."),
    cases_dir: Path | None = typer.Option(None, "--cases-dir", help="Regression cases directory."),
) -> None:
    """Promote a reviewed draft into a real regression case (existing promote flow)."""
    try:
        case = promote_draft(
            draft_id,
            drafts_dir=drafts_dir,
            cases_dir=cases_dir,
            expected_final_answer=expected_answer,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc))
    console.print(
        f"[green]Promoted draft[/green] [bold]{draft_id}[/bold] to regression case "
        f"[bold]{case.case_id}[/bold] ({case.name})"
    )


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Baseline run id."),
    candidate: str = typer.Option(..., "--candidate", help="Replayed run id to compare."),
    judge: list[str] = typer.Option(DEFAULT_JUDGES, "--judge", help="Judges to run."),
    format: str = typer.Option("terminal", "--format", help="terminal | json | junit | html"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write report to file."),
    store_dir: Path | None = typer.Option(None, "--store-dir", help="Run store directory."),
) -> None:
    """Render an eval report in the requested format."""
    baseline = load_run(run_id, store_dir)
    cand = load_run(candidate, store_dir)
    diff = diff_trajectories(baseline, cand, judges=_build_judges(judge))

    fmt = format.lower()
    if fmt == "terminal":
        terminal_report(diff, console)
        return
    if fmt == "json":
        text = json_report(diff)
    elif fmt == "junit":
        text = junit_report(diff)
    elif fmt == "html":
        text = html_report(diff, baseline=baseline, candidate=cand)
    else:
        raise typer.BadParameter("--format must be terminal | json | junit | html")

    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote {fmt} report[/green] to {output}")
    else:
        console.print(text)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

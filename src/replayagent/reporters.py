"""Reporters: render a TrajectoryDiff as terminal, JSON, JUnit XML, or HTML."""

from __future__ import annotations

import html as html_lib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from replayagent.diff import TrajectoryDiff, diff_to_dict
from replayagent.judges import JudgeResult
from replayagent.trajectory import Trajectory

_STATUS_ICON = {True: "[green]PASS[/green]", False: "[red]FAIL[/red]"}


def terminal_report(diff: TrajectoryDiff, console: Console | None = None) -> Console:
    """Render a rich terminal report. Returns the console used."""
    console = console or Console()
    verdict = (
        "[green]NO REGRESSION[/green]"
        if not diff.is_regression
        else "[red]REGRESSION DETECTED[/red]"
    )
    console.print(
        f"\n[bold]replayagent eval[/bold] {diff.baseline_run_id} → {diff.candidate_run_id}:"
        f" {verdict}\n"
    )

    table = Table(title="Judges", show_header=True, header_style="bold")
    table.add_column("Judge")
    table.add_column("Result")
    table.add_column("Score", justify="right")
    table.add_column("Reason")
    for r in diff.judge_results:
        table.add_row(r.judge, _STATUS_ICON[r.passed], f"{r.score:.2f}", r.reason)
    console.print(table)

    if diff.added_tool_calls or diff.removed_tool_calls:
        console.print("\n[bold]Tool call changes[/bold]")
        for t in diff.added_tool_calls:
            console.print(f"  [green]+ {t}[/green]")
        for t in diff.removed_tool_calls:
            console.print(f"  [red]- {t}[/red]")

    if diff.step_changes:
        changes = Table(
            title=f"Step changes ({len(diff.step_changes)})", show_header=True, header_style="bold"
        )
        changes.add_column("#", justify="right")
        changes.add_column("Kind")
        changes.add_column("Summary")
        for c in diff.step_changes[:25]:
            changes.add_row(str(c.index), c.kind, c.summary)
        console.print(changes)
        if len(diff.step_changes) > 25:
            console.print(f"  … and {len(diff.step_changes) - 25} more")
    else:
        console.print("\n[dim]No step changes.[/dim]")
    console.print()
    return console


def json_report(diff: TrajectoryDiff, indent: int = 2) -> str:
    """Render the diff as a JSON string."""
    return json.dumps(diff_to_dict(diff), indent=indent)


def junit_report(diff: TrajectoryDiff, suite_name: str = "replayagent") -> str:
    """Render the diff as JUnit XML (one testcase per judge) for CI ingestion."""
    suite = ET.Element("testsuite")
    suite.set("name", suite_name)
    suite.set("tests", str(len(diff.judge_results)))
    failures = sum(1 for r in diff.judge_results if not r.passed)
    suite.set("failures", str(failures))
    suite.set("timestamp", datetime.now(timezone.utc).isoformat())
    for r in diff.judge_results:
        case = ET.SubElement(suite, "testcase")
        case.set("classname", "replayagent.judges")
        case.set("name", r.judge)
        if not r.passed:
            failure = ET.SubElement(case, "failure")
            failure.set("message", r.reason)
            failure.text = json.dumps(r.details)
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)


def html_report(
    diff: TrajectoryDiff,
    title: str = "replayagent report",
    baseline: Trajectory | None = None,
    candidate: Trajectory | None = None,
) -> str:
    """Render a simple static HTML report (no JS, no external assets)."""
    esc = html_lib.escape
    verdict = "NO REGRESSION" if not diff.is_regression else "REGRESSION DETECTED"
    color = "#1a7f37" if not diff.is_regression else "#d1242f"

    def _judge_row(r: JudgeResult) -> str:
        status_color = "#1a7f37" if r.passed else "#d1242f"
        status_text = "PASS" if r.passed else "FAIL"
        return (
            f"<tr><td>{esc(r.judge)}</td>"
            f"<td style='color:{status_color}'>{status_text}</td>"
            f"<td>{r.score:.2f}</td><td>{esc(r.reason)}</td></tr>"
        )

    rows = "\n".join(_judge_row(r) for r in diff.judge_results)
    changes = "\n".join(
        f"<tr><td>{c.index}</td><td>{esc(c.kind)}</td><td>{esc(c.summary)}</td></tr>"
        for c in diff.step_changes
    )
    empty_changes_row = "<tr><td colspan='3'>none</td></tr>"
    changes_table = (
        "<table><tr><th>#</th><th>Kind</th><th>Summary</th></tr>"
        f"{changes or empty_changes_row}</table>"
    )
    tools = "\n".join(f"<li style='color:#1a7f37'>+ {esc(t)}</li>" for t in diff.added_tool_calls)
    tools += "\n".join(
        f"<li style='color:#d1242f'>- {esc(t)}</li>" for t in diff.removed_tool_calls
    )

    answers = ""
    if baseline is not None or candidate is not None:
        base_ans = esc(str(baseline.final_answer())) if baseline else "<i>n/a</i>"
        cand_ans = esc(str(candidate.final_answer())) if candidate else "<i>n/a</i>"
        answers = (
            "<h2>Final answers</h2>"
            f"<table><tr><th>Baseline <code>{esc(diff.baseline_run_id)}</code></th>"
            f"<th>Candidate <code>{esc(diff.candidate_run_id)}</code></th></tr>"
            f"<tr><td><pre>{base_ans}</pre></td><td><pre>{cand_ans}</pre></td></tr></table>"
        )

    return f"""<!DOCTYPE html><html lang="en">
<head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
th {{ background: #f6f8fa; }}
.verdict {{ font-size: 1.25rem; font-weight: bold; color: {color}; }}
</style></head>
<body>
<h1>{esc(title)}</h1>
<p class="verdict">{verdict}</p>
<p><code>{esc(diff.baseline_run_id)}</code> → <code>{esc(diff.candidate_run_id)}</code></p>
{answers}
<h2>Judges</h2>
<table><tr><th>Judge</th><th>Result</th><th>Score</th><th>Reason</th></tr>{rows}</table>
<h2>Tool call changes</h2>
<ul>{tools or "<li>none</li>"}</ul>
<h2>Step changes ({len(diff.step_changes)})</h2>
{changes_table}
</body></html>
"""

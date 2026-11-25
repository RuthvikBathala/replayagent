# replayagent

[![CI](https://github.com/RuthvikBathala/replayagent/actions/workflows/ci.yml/badge.svg)](https://github.com/RuthvikBathala/replayagent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Record, replay, and regression-test AI agent trajectories. pytest for AI agents.**

Model upgrades break agents silently. A new model version answers slightly differently,
calls tools in a different order, or drops a tool call entirely — and nothing in your
CI catches it, because there was nothing to catch it *against*. replayagent fixes that:
record real agent runs, replay them against new models or configs, and diff the
trajectories with pluggable judges. If anything regresses, your pipeline fails.

## How it works

1. **Record** — capture an agent run (LLM calls, tool calls, results, final answer) as a
   typed trajectory, stored as JSONL under `.replayagent/runs/`.
2. **Replay** — re-execute your agent function against the recorded input with a new
   model or config, capturing a fresh trajectory.
3. **Eval** — diff the two trajectories step-by-step, run judges (exact match, tool-call
   sequence, semantic similarity, LLM-as-judge), and get a verdict. Exits non-zero on
   regression, so it drops straight into CI.

## Quickstart

```bash
pip install replayagent
```

```python
from replayagent import record, replay, diff_trajectories
from replayagent.judges import ExactMatchJudge, ToolCallJudge

# 1. Record a baseline run
with record("support-agent", model="gpt-4o", input={"query": "reset my password"}) as rec:
    rec.llm_call(messages=[{"role": "user", "content": "reset my password"}],
                 response="I'll reset it now.", model="gpt-4o")
    rec.tool_call(name="reset_password", arguments={"user": "amy"})
    rec.tool_result(name="reset_password", result={"ok": True})
    rec.final_answer("Done — your password has been reset.")
baseline = rec.trajectory

# 2. Replay with a new model
def my_agent_v2(input_data, recorder):
    recorder.llm_call(messages=[{"role": "user", "content": input_data["query"]}],
                     response="On it.", model="gpt-4o-mini")
    recorder.tool_call(name="reset_password", arguments={"user": "amy"})
    recorder.tool_result(name="reset_password", result={"ok": True})
    recorder.final_answer("Done — your password has been reset.")

candidate = replay(baseline, my_agent_v2, model="gpt-4o-mini")

# 3. Diff + judge
diff = diff_trajectories(baseline, candidate, judges=[ExactMatchJudge(), ToolCallJudge()])
print("regression:", diff.is_regression)  # False — same tools, same answer
```

### CLI

```bash
replayagent init                                  # set up .replayagent/ in your project
replayagent record --query "summarize this doc"   # record a demo run (no API keys)
replayagent list                                  # list recorded runs
replayagent replay <run-id> --model gpt-4o-mini   # replay + diff + judge
replayagent eval <run-id> --candidate <new-id>     # judge only; exits 1 on regression
replayagent report <run-id> --candidate <new-id> --format html -o report.html
replayagent promote <run-id> --reason "broke in prod"  # save a run as a regression case
replayagent cases                                   # list regression cases
replayagent eval-case <case-id> --candidate <new-id>   # check a run against a case
replayagent ingest traces.json                      # ingest production traces into the run store
replayagent clusters traces.json                    # cluster the failures in an upload
replayagent draft-cases traces.json                 # auto-draft a case per failure cluster
replayagent drafts                                  # list drafts awaiting review
replayagent promote-draft <draft-id>                # promote a reviewed draft into a case
```

### Regression cases: turn production failures into tests

Agents pass their evals and still fail real users — and that failure is usually
lost. A regression case preserves it: promote the failed run, correct what it
*should* have done, and every future version is checked against it, so a fixed
failure stays fixed.

```bash
# The run's own answer was wrong, so record the correct expectation:
replayagent promote <run-id> --reason "told a customer the refund was approved" \
    --expected-answer "Your refund is still pending."
replayagent eval-case <case-id> --candidate <new-run-id>   # exits 1 until it's fixed
```

```python
from replayagent import promote_run, check_case

case = promote_run("abc123", reason="wrong tool in prod",
                   expected_final_answer="Refund is still pending.",
                   expected_tool_calls=["check_refund_status"])
diff = check_case(case, candidate_trajectory)  # TrajectoryDiff, .is_regression
```

Cases are plain JSON under `.replayagent/cases/`. Expectations default to the
run itself; `--expected-answer` / `expected_tool_calls` override them when a
reviewer knows the right behaviour. The original run is kept untouched inside
the case for provenance.

### From production traces to cases (ingest → cluster → draft → promote)

You don't have to record failures locally — upload the traces you already
have. `replayagent ingest` accepts a JSON file (a list, or `{"traces": [...]}`)
of Langfuse/OTel-shaped traces: each trace has an `id`, optional
`name`/`model`, an `input`, ordered `steps` (`llm_call` / `tool_call` /
`tool_result` / `final_answer`), a final `status` or `error`, and an optional
`expected` behaviour. See `examples/production_traces.json` for a sample and
the `replayagent.ingest` module docstring for the full accepted schema.

Failures are detected deterministically (explicit error status, a step error,
or a final answer that doesn't match the recorded `expected` behaviour) and
clustered by failing tool + normalised error signature, so one root cause
appears once with a count (`replayagent clusters`). `replayagent draft-cases`
then writes one **draft** case per cluster under `.replayagent/drafts/`,
pre-filled from a representative failed trace and marked `draft: true` in its
metadata. Drafts are *not* cases yet: a human reviews one — above all the
expected answer, since a failed run's own answer is usually wrong — and
`replayagent promote-draft <draft-id> [--expected-answer "..."]` sends it
through the normal promote flow into `.replayagent/cases/`, where `eval-case`
checks future runs against it like any other case. No LLM calls are made;
the judging seam (`FailureJudge`) defaults to a deterministic exact-match
check against the recorded expectation.

`replayagent eval` exits non-zero when any judge fails or the tool-call sequence
changed — wire it into CI:

```yaml
# .github/workflows/agent-eval.yml
- run: pip install replayagent
- run: replayagent eval $BASELINE_RUN --candidate $CANDIDATE_RUN --judge exact --judge tools
```

### Judges

| Judge | What it checks |
|---|---|
| `exact` | Final answers are exactly equal |
| `contains` | Final answer contains a string or regex |
| `similarity` | Offline token-overlap similarity ≥ threshold (no API needed) |
| `tools` | Tool calls match in `strict` (default, in order), `unordered`, `subset` (no extras), or `superset` (at least the baseline) mode; optionally checks call arguments too |
| `llm` | LLM-as-judge via OpenAI/Anthropic (only runs with API keys set) |

```python
from replayagent.judges import get_judge
judge = get_judge("similarity", threshold=0.8)
# Trajectory-match modes follow LangChain agentevals; check_arguments
# catches the right tool called with the wrong arguments:
judge = get_judge("tools", mode="superset", check_arguments=True)
```

## Architecture

```
src/replayagent/
├── trajectory.py   # Pydantic schema: Trajectory, Step, StepType
├── recorder.py     # Recorder (context manager + decorator), JSONL store, load/list
├── replay.py       # Re-execute an agent fn against recorded inputs
├── judges.py       # Pluggable judges: exact, contains, similarity, tools, llm
├── cases.py        # Regression cases: promote a run into a reusable test case
├── ingest.py       # Production traces: ingest -> cluster failures -> draft cases
├── diff.py         # Step-by-step diff + regression verdict
├── reporters.py    # terminal (rich) | JSON | JUnit XML | static HTML
├── cli.py          # typer CLI
└── adapters/
    ├── langgraph.py  # record compiled graph runs (optional dep)
    └── openai.py     # wrap chat.completions to auto-log llm_call steps (optional dep)
```

Trajectories are plain JSONL — one header line plus one line per step — so they're
diffable, greppable, and version-controllable. Adapters are import-guarded: the core
has four small dependencies (`pydantic`, `typer`, `rich`, `pyyaml`) and never imports
a framework or LLM SDK unless you use that adapter.

### LangGraph adapter

```python
from replayagent import record
from replayagent.adapters.langgraph import record_langgraph_run  # pip install replayagent[langgraph]

with record("my-graph", model="gpt-4o") as rec:
    record_langgraph_run(graph, {"messages": [("user", "hi")]}, rec)
```

### OpenAI adapter

```python
import openai
from replayagent import record
from replayagent.adapters.openai import RecordedOpenAIClient  # pip install replayagent[openai]

with record("my-agent", model="gpt-4o-mini") as rec:
    client = RecordedOpenAIClient(openai.OpenAI(), rec)
    client.chat.completions.create(model="gpt-4o-mini", messages=[...])  # auto-logged
```

## Hosted service (local MVP)

A multi-tenant hosted version lives in `src/replayagent/hosted/`: an
stdlib-only HTTP service where tenants submit trajectories with an API key,
store them per tenant in SQLite, and fetch diff/regression reports computed
by the core engine. Run it locally with
`REPLAYAGENT_HOSTED_ADMIN_KEY=dev-admin python -m replayagent.hosted`.
See [HOSTED_NOTES.md](HOSTED_NOTES.md) for the API, design, and what is
still missing before a real launch. Not deployed anywhere.

## Roadmap

- Pytest plugin (`@pytest.mark.agent_eval`) for trajectory assertions in test suites
- Deterministic tool mocking during replay (serve recorded tool results)
- Cost/latency tracking per run (token usage is already captured on steps)
- Web dashboard for browsing runs and diffs
- More adapters: CrewAI, LlamaIndex, Anthropic SDK

## Contributing

PRs welcome. Keep it dependency-light, typed, and tested:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

## License

MIT — see [LICENSE](LICENSE).

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
```

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
| `tools` | Same tool calls, in order (or unordered) |
| `llm` | LLM-as-judge via OpenAI/Anthropic (only runs with API keys set) |

```python
from replayagent.judges import get_judge
judge = get_judge("similarity", threshold=0.8)
```

## Architecture

```
src/replayagent/
├── trajectory.py   # Pydantic schema: Trajectory, Step, StepType
├── recorder.py     # Recorder (context manager + decorator), JSONL store, load/list
├── replay.py       # Re-execute an agent fn against recorded inputs
├── judges.py       # Pluggable judges: exact, contains, similarity, tools, llm
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

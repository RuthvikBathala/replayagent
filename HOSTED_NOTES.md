# Hosted replayagent — MVP notes

A hosted, multi-tenant version of replayagent: instead of running the CLI
locally, teams submit agent trajectories to a small HTTP service, which
stores them per tenant and serves diff/regression reports computed by the
existing core library (`diff_trajectories` + judges). Built locally only —
**not deployed, not launched.**

## What was built

New subpackage `src/replayagent/hosted/` (stdlib only — no new dependencies):

- `store.py` — SQLite storage. Tenants table (one API key per tenant, only
  the SHA-256 hash stored; plaintext shown once at creation) and a runs
  table keyed by `(tenant_id, run_id)`. Every query is tenant-scoped.
- `service.py` — business logic on top of the core: submit a run, list runs,
  fetch a run, fetch a report. When a submission names a `baseline_run_id`
  belonging to the same tenant, the core diff engine runs with the default
  judges (tool-call sequence + exact final-answer match) and the diff —
  including the `is_regression` verdict — is stored with the run.
- `app.py` — HTTP layer on `http.server.ThreadingHTTPServer`.

### API

| Route | Auth | Purpose |
|---|---|---|
| `GET /healthz` | none | liveness check |
| `POST /v1/tenants` | admin key (if configured) | create tenant → `{tenant_id, api_key}` |
| `POST /v1/runs` | tenant API key | submit `{trajectory, baseline_run_id?}` → run summary |
| `GET /v1/runs` | tenant API key | list the tenant's run summaries |
| `GET /v1/runs/{run_id}` | tenant API key | full stored trajectory |
| `GET /v1/runs/{run_id}/report` | tenant API key | diff/regression report (or a no-baseline summary) |

Tenant auth is `Authorization: Bearer <key>` or `X-API-Key: <key>`.
Tenant isolation: another tenant's run returns **404**, never 403, so run
existence doesn't leak across tenants; a foreign run also can't be used as
a diff baseline (400, same as a missing one).

## Run it locally

```bash
cd ~/workspace/agent-eval
REPLAYAGENT_HOSTED_ADMIN_KEY=dev-admin \
REPLAYAGENT_HOSTED_DB=.replayagent/hosted.db \
REPLAYAGENT_HOSTED_PORT=8000 \
  .venv/bin/python -m replayagent.hosted
# or, once installed: replayagent-hosted
```

```bash
# create a tenant (save the api_key — it is shown only once)
curl -X POST localhost:8000/v1/tenants -H 'X-Admin-Key: dev-admin' \
  -H 'Content-Type: application/json' -d '{"name": "acme"}'

# submit a run, list runs, fetch a report
curl -X POST localhost:8000/v1/runs -H 'Authorization: Bearer rak_...' \
  -H 'Content-Type: application/json' \
  -d '{"trajectory": {"run_id": "r1", "agent_name": "demo", "model": "gpt-4o", "steps": []}}'
curl localhost:8000/v1/runs -H 'Authorization: Bearer rak_...'
curl localhost:8000/v1/runs/r1/report -H 'Authorization: Bearer rak_...'
```

## Tests

`tests/test_hosted.py` — 9 tests over real HTTP against an ephemeral-port
server, including the mandatory tenant-isolation test (list/get/report/
baseline access across tenants). Full suite: **116 passing** (107 core +
9 hosted).

## Still missing before a real launch

- **Hosting/deploy**: runs only on localhost; no TLS, no process manager,
  no public deployment (deliberately — launching is Ruthvik's call).
- **Real auth & accounts**: no signup flow, no password/OAuth login, no key
  rotation or revocation, no per-key scopes. Tenant creation is gated by a
  single shared admin key from an env var.
- **Billing**: none — no plans, quotas, Stripe, or usage metering.
- **Rate limits & abuse controls**: none; no request-size caps beyond
  Content-Length, no throttling.
- **Storage scale**: single-file SQLite with a global write lock; a real
  service needs Postgres (or similar) and migrations.
- **Richer eval surface**: judges are fixed (tool-call + exact match);
  no per-tenant judge config, no LLM judge (needs API keys behind an
  interface), no trace-ingestion endpoint yet, no UI/dashboard, no webhooks
  or CI integration for the hosted flow.
- **Ops**: no logging/metrics, backups, or data-retention policy.
- **Business/legal gates**: first paying customer still needs the DSO /
  immigration-counsel check (STEM OPT); nothing here changes that.

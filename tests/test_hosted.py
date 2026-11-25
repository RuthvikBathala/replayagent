"""Tests for the hosted multi-tenant service (stdlib HTTP + SQLite store)."""

import hashlib
import http.client
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from replayagent.hosted import create_server
from tests.conftest import make_trajectory

ADMIN_KEY = "test-admin-key"


@pytest.fixture()
def server(tmp_path: Path):
    srv = create_server(
        db_path=tmp_path / "hosted.db", host="127.0.0.1", port=0, admin_key=ADMIN_KEY
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(server, method: str, path: str, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    payload = json.dumps(body) if body is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8") or "{}")
    conn.close()
    return resp.status, data


def make_tenant(server, name: str) -> dict:
    status, data = call(
        server, "POST", "/v1/tenants", {"name": name}, {"X-Admin-Key": ADMIN_KEY}
    )
    assert status == 201
    return data


def auth(tenant: dict) -> dict:
    return {"Authorization": f"Bearer {tenant['api_key']}"}


def submit(server, tenant: dict, traj, baseline_run_id: str | None = None):
    body = {"trajectory": traj.model_dump(mode="json")}
    if baseline_run_id:
        body["baseline_run_id"] = baseline_run_id
    return call(server, "POST", "/v1/runs", body, auth(tenant))


def test_healthz(server):
    status, data = call(server, "GET", "/healthz")
    assert status == 200
    assert data == {"status": "ok"}


def test_tenant_creation_requires_admin_key(server):
    status, _ = call(server, "POST", "/v1/tenants", {"name": "acme"})
    assert status == 401
    status, _ = call(
        server, "POST", "/v1/tenants", {"name": "acme"}, {"X-Admin-Key": "wrong"}
    )
    assert status == 401


def test_api_key_is_hashed_at_rest(server, tmp_path: Path):
    tenant = make_tenant(server, "acme")
    assert tenant["api_key"].startswith("rak_")
    conn = sqlite3.connect(tmp_path / "hosted.db")
    stored = conn.execute("SELECT api_key_hash FROM tenants").fetchone()[0]
    conn.close()
    assert stored != tenant["api_key"]
    assert stored == hashlib.sha256(tenant["api_key"].encode()).hexdigest()


def test_submit_list_and_get_run(server):
    tenant = make_tenant(server, "acme")
    traj = make_trajectory(run_id="run-a")
    status, summary = submit(server, tenant, traj)
    assert status == 201
    assert summary["run_id"] == "run-a"
    assert summary["is_regression"] is None  # no baseline supplied

    status, data = call(server, "GET", "/v1/runs", headers=auth(tenant))
    assert status == 200
    assert [r["run_id"] for r in data["runs"]] == ["run-a"]

    status, fetched = call(server, "GET", "/v1/runs/run-a", headers=auth(tenant))
    assert status == 200
    assert fetched["agent_name"] == traj.agent_name
    assert len(fetched["steps"]) == len(traj.steps)


def test_duplicate_run_id_conflicts(server):
    tenant = make_tenant(server, "acme")
    traj = make_trajectory(run_id="dup")
    assert submit(server, tenant, traj)[0] == 201
    assert submit(server, tenant, traj)[0] == 409


def test_report_with_baseline_detects_regression(server):
    tenant = make_tenant(server, "acme")
    baseline = make_trajectory(run_id="base", answer="Done — reset.")
    changed = make_trajectory(
        run_id="cand", answer="Something else entirely.", tools=("delete_user",)
    )
    assert submit(server, tenant, baseline)[0] == 201
    status, summary = submit(server, tenant, changed, baseline_run_id="base")
    assert status == 201
    assert summary["is_regression"] is True

    status, report = call(server, "GET", "/v1/runs/cand/report", headers=auth(tenant))
    assert status == 200
    assert report["diff"]["baseline_run_id"] == "base"
    assert report["diff"]["is_regression"] is True
    assert "delete_user" in report["diff"]["added_tool_calls"]


def test_report_without_baseline(server):
    tenant = make_tenant(server, "acme")
    assert submit(server, tenant, make_trajectory(run_id="solo"))[0] == 201
    status, report = call(server, "GET", "/v1/runs/solo/report", headers=auth(tenant))
    assert status == 200
    assert report["diff"] is None
    assert report["tool_calls"] == ["reset_password"]


def test_missing_or_bad_api_key_rejected(server):
    assert call(server, "GET", "/v1/runs")[0] == 401
    bad = {"Authorization": "Bearer rak_not-a-real-key"}
    assert call(server, "GET", "/v1/runs", headers=bad)[0] == 401
    assert call(server, "POST", "/v1/runs", {"trajectory": {}}, bad)[0] == 401


def test_tenant_isolation(server):
    """Tenant B must not see, fetch, report on, or diff against tenant A's runs."""
    tenant_a = make_tenant(server, "tenant-a")
    tenant_b = make_tenant(server, "tenant-b")
    traj_a = make_trajectory(run_id="a-run")
    assert submit(server, tenant_a, traj_a)[0] == 201

    # B's list is empty.
    status, data = call(server, "GET", "/v1/runs", headers=auth(tenant_b))
    assert status == 200
    assert data["runs"] == []

    # B fetching A's run gets a 404 — existence does not leak.
    assert call(server, "GET", "/v1/runs/a-run", headers=auth(tenant_b))[0] == 404
    assert call(server, "GET", "/v1/runs/a-run/report", headers=auth(tenant_b))[0] == 404

    # B cannot use A's run as a diff baseline.
    status, _ = submit(
        server, tenant_b, make_trajectory(run_id="b-run"), baseline_run_id="a-run"
    )
    assert status == 400

    # The same run_id under B is a separate, independent run.
    assert submit(server, tenant_b, make_trajectory(run_id="a-run"))[0] == 201
    status, data = call(server, "GET", "/v1/runs", headers=auth(tenant_b))
    assert [r["run_id"] for r in data["runs"]] == ["a-run"]

    # A's view is untouched by B's activity.
    status, data = call(server, "GET", "/v1/runs", headers=auth(tenant_a))
    assert [r["run_id"] for r in data["runs"]] == ["a-run"]

"""HTTP layer for the hosted service — stdlib only (http.server).

Routes:
    GET  /healthz                  unauthenticated liveness check
    POST /v1/tenants               create a tenant (admin key if configured)
    POST /v1/runs                  submit a run (tenant API key)
    GET  /v1/runs                  list the tenant's runs
    GET  /v1/runs/{run_id}         fetch one run's trajectory
    GET  /v1/runs/{run_id}/report  diff / regression report for a run

Tenant auth: ``Authorization: Bearer <api_key>`` or ``X-API-Key: <api_key>``.
A run belonging to another tenant is indistinguishable from a missing run
(404), so existence never leaks across tenants.

Run locally::

    REPLAYAGENT_HOSTED_ADMIN_KEY=dev-admin python -m replayagent.hosted
"""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from replayagent.hosted.service import HostedService, ServiceError
from replayagent.hosted.store import Store
from replayagent.trajectory import Trajectory

DEFAULT_DB = Path(".replayagent") / "hosted.db"


def create_server(
    db_path: Path | str = DEFAULT_DB,
    host: str = "127.0.0.1",
    port: int = 8000,
    admin_key: str | None = None,
) -> ThreadingHTTPServer:
    service = HostedService(Store(db_path))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # -- plumbing ----------------------------------------------------
        def log_message(self, *args: Any) -> None:  # keep test output clean
            pass

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ServiceError(400, "request body must be valid JSON") from None
            if not isinstance(data, dict):
                raise ServiceError(400, "request body must be a JSON object")
            return data

        def _tenant_id(self) -> str:
            auth = self.headers.get("Authorization", "")
            key = auth[7:] if auth.startswith("Bearer ") else self.headers.get("X-API-Key")
            tenant_id = service.store.authenticate(key)
            if tenant_id is None:
                raise ServiceError(401, "missing or invalid API key")
            return tenant_id

        def _dispatch(self, method: str) -> None:
            try:
                self._route(method)
            except ServiceError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json(500, {"error": f"internal error: {exc}"})

        # -- routing -------------------------------------------------------
        def _route(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            if method == "GET" and path == "/healthz":
                return self._send_json(200, {"status": "ok"})
            if method == "POST" and path == "/v1/tenants":
                return self._create_tenant()
            if method == "POST" and path == "/v1/runs":
                return self._submit_run()
            if method == "GET" and path == "/v1/runs":
                tenant_id = self._tenant_id()
                return self._send_json(200, {"runs": service.list_runs(tenant_id)})
            m = re.fullmatch(r"/v1/runs/([^/]+)(/report)?", path)
            if method == "GET" and m:
                tenant_id = self._tenant_id()
                run_id = m.group(1)
                if m.group(2):
                    return self._send_json(200, service.get_report(tenant_id, run_id))
                return self._send_json(200, service.get_run(tenant_id, run_id))
            raise ServiceError(404, "not found")

        # -- handlers ------------------------------------------------------
        def _create_tenant(self) -> None:
            if admin_key is not None and self.headers.get("X-Admin-Key") != admin_key:
                raise ServiceError(401, "missing or invalid admin key")
            body = self._read_json()
            name = str(body.get("name") or "").strip()
            if not name:
                raise ServiceError(400, "tenant 'name' is required")
            self._send_json(201, service.create_tenant(name))

        def _submit_run(self) -> None:
            tenant_id = self._tenant_id()
            body = self._read_json()
            traj_data = body.get("trajectory")
            if not isinstance(traj_data, dict):
                raise ServiceError(400, "'trajectory' object is required")
            try:
                trajectory = Trajectory(**traj_data)
            except ValidationError as exc:
                raise ServiceError(400, f"invalid trajectory: {exc.errors()}") from None
            baseline = body.get("baseline_run_id")
            summary = service.submit_run(
                tenant_id, trajectory, baseline_run_id=str(baseline) if baseline else None
            )
            self._send_json(201, summary)

        do_GET = lambda self: self._dispatch("GET")  # noqa: E731
        do_POST = lambda self: self._dispatch("POST")  # noqa: E731

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    db = os.environ.get("REPLAYAGENT_HOSTED_DB", str(DEFAULT_DB))
    port = int(os.environ.get("REPLAYAGENT_HOSTED_PORT", "8000"))
    admin_key = os.environ.get("REPLAYAGENT_HOSTED_ADMIN_KEY") or None
    server = create_server(db_path=db, port=port, admin_key=admin_key)
    print(f"replayagent hosted service on http://127.0.0.1:{port} (db: {db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

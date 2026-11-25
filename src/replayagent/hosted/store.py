"""SQLite storage for the hosted service: tenants, API keys, and runs.

Design notes for the MVP:
- One API key per tenant. Keys are generated with :mod:`secrets` and only
  their SHA-256 hash is stored — the plaintext key is shown once at tenant
  creation and never persisted.
- Runs are stored per tenant; the primary key is (tenant_id, run_id), so two
  tenants may use the same run_id without colliding, and every query is
  scoped by tenant_id (tenant isolation is enforced at this layer).
- A single connection guarded by a lock keeps the stdlib-only store safe
  under the threading HTTP server.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY_PREFIX = "rak_"


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Tenant + run storage backed by a local SQLite database file."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    trajectory_json TEXT NOT NULL,
                    baseline_run_id TEXT,
                    diff_json TEXT,
                    submitted_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, run_id)
                );
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- tenants ---------------------------------------------------------
    def create_tenant(self, name: str) -> tuple[str, str]:
        """Create a tenant; returns (tenant_id, plaintext_api_key shown once)."""
        tenant_id = uuid.uuid4().hex[:16]
        api_key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            self._conn.execute(
                "INSERT INTO tenants (id, name, api_key_hash, created_at) VALUES (?, ?, ?, ?)",
                (tenant_id, name, _hash_key(api_key), _now()),
            )
            self._conn.commit()
        return tenant_id, api_key

    def authenticate(self, api_key: str | None) -> str | None:
        """Return the tenant_id for a plaintext API key, or None."""
        if not api_key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM tenants WHERE api_key_hash = ?", (_hash_key(api_key),)
            ).fetchone()
        return str(row["id"]) if row else None

    # -- runs ------------------------------------------------------------
    def run_exists(self, tenant_id: str, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM runs WHERE tenant_id = ? AND run_id = ?",
                (tenant_id, run_id),
            ).fetchone()
        return row is not None

    def save_run(
        self,
        tenant_id: str,
        trajectory: dict[str, Any],
        baseline_run_id: str | None,
        diff: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs
                    (tenant_id, run_id, agent_name, model, trajectory_json,
                     baseline_run_id, diff_json, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    trajectory["run_id"],
                    trajectory.get("agent_name", ""),
                    trajectory.get("model", ""),
                    json.dumps(trajectory),
                    baseline_run_id,
                    json.dumps(diff) if diff is not None else None,
                    _now(),
                ),
            )
            self._conn.commit()

    def get_run_row(self, tenant_id: str, run_id: str) -> dict[str, Any] | None:
        """Fetch one run row scoped to the tenant; None if absent (or not theirs)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE tenant_id = ? AND run_id = ?",
                (tenant_id, run_id),
            ).fetchone()
        return dict(row) if row else None

    def list_run_rows(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE tenant_id = ? ORDER BY submitted_at DESC",
                (tenant_id,),
            ).fetchall()
        return [dict(r) for r in rows]

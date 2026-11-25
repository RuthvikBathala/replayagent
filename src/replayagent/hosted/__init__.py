"""Hosted multi-tenant eval service for replayagent (local MVP).

A thin HTTP layer over the core library: tenants submit trajectories,
store them per-tenant in SQLite, and fetch diff/regression reports computed
by the core diff engine. See HOSTED_NOTES.md at the repo root.
"""

from replayagent.hosted.app import create_server
from replayagent.hosted.service import HostedService, ServiceError
from replayagent.hosted.store import Store

__all__ = ["HostedService", "ServiceError", "Store", "create_server"]

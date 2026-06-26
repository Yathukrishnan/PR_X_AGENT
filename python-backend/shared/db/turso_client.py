"""
Shared Turso / libsql embedded-replica connection helper.

Both agents' db.py use this to open their replica connection. Lightweight by
design — it only owns connection construction; callers keep their own schema
init, sync() timing, and query helpers.
"""
from __future__ import annotations

from typing import Any

import libsql


def connect(replica_path, sync_url: str, auth_token: str, sync_interval: int | None = 60):
    """Open a libsql embedded-replica connection.

    sync_interval is the background sync cadence in seconds; pass None to
    disable background sync entirely (required for write-heavy callers, e.g.
    orchestrator sweeps, to avoid a mid-write WalConflict). The caller is
    responsible for calling .sync() when it wants to pull/push.
    """
    kwargs: dict[str, Any] = {"sync_url": sync_url, "auth_token": auth_token}
    if sync_interval is not None:
        kwargs["sync_interval"] = sync_interval
    return libsql.connect(str(replica_path), **kwargs)

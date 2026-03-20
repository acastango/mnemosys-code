"""
mnemo_log.py — Structured event logging for the mnemo sidecar UI

Each call to configure() starts a new session log file under
MNEMO_STORE/logs/YYYYMMDD_HHMMSS_<id>.log and writes its path
to MNEMO_STORE/logs/current.txt so the sidecar can auto-find it.

Layers:
    subconscious — ambient, automatic operations (recall, auto-extraction)
    conscious    — deliberate tool calls (claim, update, reinforce, ...)
    system       — housekeeping, status, compression, maintenance

Events:
    recall, claim, update, reinforce, compress, search, query,
    provenance, status, soul, reroot, prune, active, dream
"""

import json
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

# ─── State ────────────────────────────────────────────────────────────────────

_log_path: Optional[str] = None


def configure(store_path: str) -> str:
    """
    Start a new session log file. Creates STORE/logs/ if needed,
    writes STORE/logs/current.txt pointing at the new file.
    Returns the new log path. Call once at session startup.
    """
    global _log_path
    logs_dir = os.path.join(store_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = secrets.token_hex(3)  # 6 hex chars — enough to distinguish same-second starts
    _log_path = os.path.join(logs_dir, f"{ts}_{sid}.log")

    # Write pointer so the sidecar can auto-resolve the current session
    try:
        with open(os.path.join(logs_dir, "current.txt"), "w", encoding="utf-8") as f:
            f.write(_log_path + "\n")
    except Exception:
        pass

    return _log_path


def _get_log_path() -> str:
    """Return the active log path, auto-configuring if needed."""
    if _log_path:
        return _log_path
    store = os.environ.get("MNEMO_STORE", os.path.expanduser("~/mnemo"))
    return configure(store)


# ─── Public API ───────────────────────────────────────────────────────────────

def emit(
    event: str,
    layer: str,
    summary: str,
    addresses: Optional[list] = None,
    domain: Optional[str] = None,
    detail: Optional[dict] = None,
) -> None:
    """
    Emit a structured memory event to the current session log.

    Never raises — silently swallows errors so the main process is never
    interrupted by logging failures.

    Args:
        event:     Event type (recall, claim, update, reinforce, compress, ...)
        layer:     Process layer (subconscious, conscious, system)
        summary:   Human-readable description of what happened
        addresses: Memory node addresses involved (short or full)
        domain:    Memory domain (architecture, decisions, patterns, ...)
        detail:    Optional extra structured data
    """
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "event": event,
        "summary": summary,
    }
    if addresses:
        record["addresses"] = [a[:12] for a in addresses]
    if domain:
        record["domain"] = domain
    if detail:
        record["detail"] = detail

    try:
        path = _get_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # Never crash the main process for logging

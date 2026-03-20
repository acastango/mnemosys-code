"""
mnemo_web.py — FastAPI read-only API layer for the mnemo observability dashboard

Endpoints:
    GET  /api/status          Active count, context size, domain breakdown, health
    GET  /api/nodes           All active nodes with metadata (filterable, sortable)
    GET  /api/nodes/{addr}    Single node detail
    GET  /api/provenance/{addr}  Provenance chain
    GET  /api/graph           D3.js force graph data (nodes + links)
    GET  /api/roots           Root history with timestamps
    GET  /api/logs            Recent log entries from session log
    WS   /ws/logs             Live tail of session log file
    GET  /                    Serves dashboard/templates/index.html
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mnemo import Store, Node

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STORE_PATH = Path(os.environ.get("MNEMO_STORE", os.path.expanduser("~/mnemo")))
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

app = FastAPI(title="mnemo dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _store() -> Store:
    """Return a fresh Store handle (re-reads index files each call)."""
    return Store(STORE_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_summary(node: Node) -> dict:
    """Flatten a node into a JSON-friendly dict with dashboard-relevant fields."""
    age_seconds = time.time() - node.created
    recall_count = node.meta.get("recall_count", 0)
    recall_hits = node.meta.get("recall_hits", 0)
    return {
        "addr": node.addr,
        "type": node.type,
        "domain": node.meta.get("domain", ""),
        "content": node.content,
        "content_preview": node.content[:120].replace("\n", " "),
        "age_seconds": round(age_seconds, 1),
        "age_days": round(age_seconds / 86400, 1),
        "age_human": _human_age(age_seconds),
        "confidence": node.meta.get("confidence"),
        "recall_count": recall_count,
        "recall_hits": recall_hits,
        "recall_hit_rate": round(recall_hits / recall_count, 3) if recall_count > 0 else None,
        "reinforcement_count": node.meta.get("reinforcement_count", 0),
        "coverage_score": node.meta.get("coverage_score"),
        "inputs": node.inputs,
        "created": node.created,
        "meta": node.meta,
    }


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"


def _get_current_log_path() -> Optional[Path]:
    """Read the current session log path from {store}/logs/current.txt."""
    pointer = STORE_PATH / "logs" / "current.txt"
    if not pointer.exists():
        return None
    try:
        raw = pointer.read_text(encoding="utf-8").strip()
        p = Path(raw)
        if p.exists():
            return p
    except Exception:
        pass
    return None


def _read_log_lines(limit: int = 100) -> list[dict]:
    """Read the last `limit` lines from the current session log."""
    log_path = _get_current_log_path()
    if not log_path or not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        tail = lines[-limit:] if len(lines) > limit else lines
        entries = []
        for line in tail:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
    except Exception:
        return []


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status():
    """Active node count, context size, domain breakdown, root history, health."""
    store = _store()
    active_addrs = store.get_active()
    roots = store.get_roots()

    # Load active nodes for domain breakdown and context size
    domain_counts: dict[str, int] = {}
    total_content_chars = 0
    type_counts: dict[str, int] = {}

    for addr in active_addrs:
        node = store.get(addr)
        if not node:
            continue
        total_content_chars += len(node.content)
        domain = node.meta.get("domain", "uncategorized")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        type_counts[node.type] = type_counts.get(node.type, 0) + 1

    sidecar_cap = int(os.environ.get("MNEMO_SIDECAR_CAP", "15000"))
    pressure = "HIGH" if total_content_chars > sidecar_cap * 0.7 else "LOW"

    return {
        "active_count": len(active_addrs),
        "context_chars": total_content_chars,
        "pressure": pressure,
        "sidecar_cap": sidecar_cap,
        "domains": domain_counts,
        "types": type_counts,
        "root_count": len(roots),
        "current_root": roots[-1] if roots else None,
    }


@app.get("/api/nodes")
def api_nodes(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    sort: Optional[str] = Query(None, description="Sort field (e.g. created, confidence, recall_count, domain)"),
):
    """All active nodes with metadata. Supports domain filter and sort."""
    store = _store()
    active_addrs = store.get_active()

    nodes = []
    for addr in active_addrs:
        node = store.get(addr)
        if not node:
            continue
        if domain and node.meta.get("domain", "") != domain:
            continue
        nodes.append(_node_summary(node))

    if sort:
        reverse = True  # Most sort fields benefit from descending
        if sort in ("domain", "type", "addr"):
            reverse = False
        nodes.sort(key=lambda n: (n.get(sort) is None, n.get(sort, "")), reverse=reverse)

    return {"count": len(nodes), "nodes": nodes}


@app.get("/api/nodes/{addr}")
def api_node_detail(addr: str):
    """Single node detail: full content, all metadata, inputs, preserved_values."""
    store = _store()
    node = store.get(addr)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Node {addr} not found"})

    detail = _node_summary(node)
    detail["preserved_values"] = node.meta.get("preserved_values", [])

    # Resolve input nodes for context
    input_summaries = []
    for inp_addr in node.inputs:
        inp = store.get(inp_addr)
        if inp:
            input_summaries.append({
                "addr": inp.addr,
                "type": inp.type,
                "content_preview": inp.content[:120],
                "domain": inp.meta.get("domain", ""),
            })
    detail["input_details"] = input_summaries

    return detail


@app.get("/api/provenance/{addr}")
def api_provenance(addr: str):
    """Provenance chain for a node — ancestor walk to leaves."""
    store = _store()
    chain = store.provenance(addr)
    if not chain:
        return JSONResponse(status_code=404, content={"error": f"No provenance for {addr}"})
    return {
        "addr": addr,
        "chain": [
            {
                "addr": n.addr,
                "type": n.type,
                "content_preview": n.content[:200],
                "inputs": n.inputs,
                "domain": n.meta.get("domain", ""),
                "created": n.created,
            }
            for n in chain
        ],
    }


@app.get("/api/graph")
def api_graph():
    """All active nodes + link edges, formatted for D3.js force graph."""
    store = _store()
    active_addrs = store.get_active()

    graph_nodes = []
    links = []
    active_set = set(active_addrs)

    for addr in active_addrs:
        node = store.get(addr)
        if not node:
            continue
        label = node.content[:60].replace("\n", " ")
        graph_nodes.append({
            "id": node.addr,
            "domain": node.meta.get("domain", ""),
            "type": node.type,
            "label": label,
            "recall_count": node.meta.get("recall_count", 0),
            "recall_hits": node.meta.get("recall_hits", 0),
            "confidence": node.meta.get("confidence"),
            "created": node.created,
        })

        # Provenance links from inputs (only to other active nodes)
        for inp_addr in node.inputs:
            if inp_addr in active_set:
                rel = "input_of"
                if node.type == "supersede":
                    rel = "supersedes"
                elif node.type == "compress":
                    rel = "compresses"
                links.append({
                    "source": node.addr,
                    "target": inp_addr,
                    "rel": rel,
                })

        # Graph links from meta.links
        for link in node.meta.get("links", []):
            target = link.get("addr", "")
            if target in active_set:
                links.append({
                    "source": node.addr,
                    "target": target,
                    "rel": link.get("rel", "relates_to"),
                })

    return {"nodes": graph_nodes, "links": links}


@app.get("/api/roots")
def api_roots():
    """Root history with timestamps."""
    store = _store()
    root_addrs = store.get_roots()
    roots = []
    for addr in root_addrs:
        node = store.get(addr)
        if node:
            roots.append({
                "addr": node.addr,
                "created": node.created,
                "created_human": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(node.created)
                ),
                "active_count": node.meta.get("active_count"),
                "content_preview": node.content[:200],
            })
    return {"count": len(roots), "roots": roots}


@app.get("/api/logs")
def api_logs(limit: int = Query(100, ge=1, le=1000)):
    """Recent log entries from the current session log file."""
    entries = _read_log_lines(limit)
    return {"count": len(entries), "entries": entries}


# ---------------------------------------------------------------------------
# WebSocket — live log tail
# ---------------------------------------------------------------------------

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """Live tail of the session log file. Polls every 500ms for new lines."""
    await websocket.accept()

    log_path = _get_current_log_path()
    if not log_path or not log_path.exists():
        await websocket.send_json({"error": "No active session log found"})
        await websocket.close()
        return

    # Start from end of file
    try:
        offset = log_path.stat().st_size
    except Exception:
        offset = 0

    try:
        while True:
            await asyncio.sleep(0.5)

            try:
                current_size = log_path.stat().st_size
            except Exception:
                continue

            if current_size <= offset:
                # File may have been rotated — check if current.txt changed
                new_path = _get_current_log_path()
                if new_path and new_path != log_path:
                    log_path = new_path
                    offset = 0
                    current_size = log_path.stat().st_size
                else:
                    continue

            # Read new bytes
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                new_data = f.read()
                offset = f.tell()

            for line in new_data.strip().splitlines():
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    await websocket.send_json(entry)
                except json.JSONDecodeError:
                    continue

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Static file serving — dashboard UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve dashboard/templates/index.html at root."""
    index_path = DASHBOARD_DIR / "templates" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>mnemo dashboard</h1><p>Place index.html in dashboard/templates/</p>",
        status_code=200,
    )


# Mount static files (CSS, JS, images) from dashboard/static/
if (DASHBOARD_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

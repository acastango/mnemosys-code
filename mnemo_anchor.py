"""
mnemo_anchor.py — Content-hash anchor operations

Binds mnemo nodes to specific code by content hash rather than line number.
Anchors survive refactors, insertions, and moves — they follow the code
they describe, not a brittle position.

Anchor format (stored in node.meta["anchors"]):
{
    "type": "content_hash",
    "file": "relative/path/to/file.py",
    "content_hash": "a7f3c2...",     # SHA256[:16] of normalized context_lines
    "context_lines": "...",           # the actual code text (signature + first lines)
    "scope": "function|block|struct|module",
    "line_hint": 42                   # last known line, for search optimization
}

When context_lines are found but hash mismatches: anchor is "drifted".
When context_lines are not found at all: anchor is "missing".

The file index (index/by_file.json) maps filepath -> [{addr, anchor_idx}]
for O(1) lookup. Falls back to active-set scan when index is cold.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from mnemo import Store, Node


_INDEX_FILE = "by_file.json"


# ───────────────────────────────────────────────────────────────────
# Hash computation
# ───────────────────────────────────────────────────────────────────

def compute_content_hash(context_lines: str) -> str:
    """SHA256[:16] of whitespace-normalized context_lines.

    Normalizes leading/trailing whitespace per line and collapses blank
    lines — survives indentation changes and minor reformatting.
    """
    normalized = "\n".join(
        line.strip()
        for line in context_lines.splitlines()
        if line.strip()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ───────────────────────────────────────────────────────────────────
# Anchor search
# ───────────────────────────────────────────────────────────────────

def find_anchor_in_file(anchor: dict, filepath: Path) -> dict:
    """Search for an anchor's context_lines in a file, check for drift.

    Returns:
        found       — whether context_lines exist in the file
        line_num    — 1-based line of match (None if not found)
        current_hash — hash of what's actually there (None if not found)
        drifted     — found but hash changed (semantics shifted)
        detail      — human-readable status
    """
    context_lines = anchor.get("context_lines", "")
    stored_hash = anchor.get("content_hash", "")

    if not context_lines:
        return {
            "found": False, "line_num": None,
            "current_hash": None, "drifted": False,
            "detail": "anchor has no context_lines",
        }

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "found": False, "line_num": None,
            "current_hash": None, "drifted": False,
            "detail": f"error reading file: {e}",
        }

    file_lines = text.splitlines()

    # Normalize needle: strip each line, drop blanks
    needle_lines = [
        line.strip()
        for line in context_lines.splitlines()
        if line.strip()
    ]
    if not needle_lines:
        return {
            "found": False, "line_num": None,
            "current_hash": None, "drifted": False,
            "detail": "empty context_lines after normalization",
        }

    # Normalize haystack: strip, track original line numbers
    norm_lines = []       # stripped non-blank lines
    orig_line_nums = []   # corresponding 1-based line numbers
    for i, line in enumerate(file_lines, 1):
        stripped = line.strip()
        if stripped:
            norm_lines.append(stripped)
            orig_line_nums.append(i)

    # Sliding window search
    n = len(needle_lines)
    found_at = None
    for i in range(len(norm_lines) - n + 1):
        if norm_lines[i:i + n] == needle_lines:
            found_at = i
            break

    if found_at is None:
        return {
            "found": False, "line_num": None,
            "current_hash": None, "drifted": False,
            "detail": "context_lines not found in file",
        }

    orig_line = orig_line_nums[found_at]
    found_text = "\n".join(norm_lines[found_at:found_at + n])
    current_hash = hashlib.sha256(found_text.encode("utf-8")).hexdigest()[:16]
    drifted = bool(stored_hash) and (current_hash != stored_hash)

    return {
        "found": True,
        "line_num": orig_line,
        "current_hash": current_hash,
        "drifted": drifted,
        "detail": "drifted" if drifted else "ok",
    }


# ───────────────────────────────────────────────────────────────────
# File index
# ───────────────────────────────────────────────────────────────────

def _index_path(store: Store) -> Path:
    return store.index_dir / _INDEX_FILE


def load_file_index(store: Store) -> dict:
    """Load file -> [{addr, anchor_idx}] index. Returns {} if absent."""
    path = _index_path(store)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {}


def _save_file_index(store: Store, index: dict) -> None:
    path = _index_path(store)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, str(path))


def update_file_index(store: Store, node: Node) -> None:
    """Register all content_hash anchors on a node in the file index.

    Called after node is stored (claim or supersede).
    """
    anchors = node.meta.get("anchors", [])
    ch = [(i, a) for i, a in enumerate(anchors)
          if a.get("type") == "content_hash"]
    if not ch:
        return

    index = load_file_index(store)
    for anchor_idx, anchor in ch:
        filepath = anchor.get("file", "")
        if not filepath:
            continue
        entries = index.setdefault(filepath, [])
        entry = {"addr": node.addr, "anchor_idx": anchor_idx}
        if not any(e["addr"] == node.addr and e["anchor_idx"] == anchor_idx
                   for e in entries):
            entries.append(entry)

    _save_file_index(store, index)


def remove_from_file_index(store: Store, addr: str) -> None:
    """Remove a node's entries from the file index (call on supersede)."""
    index = load_file_index(store)
    changed = False
    for filepath in list(index.keys()):
        before = index[filepath]
        after = [e for e in before if e["addr"] != addr]
        if len(after) != len(before):
            index[filepath] = after
            changed = True
        if not index[filepath]:
            del index[filepath]
    if changed:
        _save_file_index(store, index)


def get_anchors_for_file(filepath: str, store: Store) -> list[dict]:
    """Return all active content_hash anchors pointing to a file.

    Returns list of {"node": Node, "anchor": dict, "anchor_idx": int}.
    Uses file index (O(1)); falls back to active-set scan on cold start.
    """
    basename = os.path.basename(filepath)
    index = load_file_index(store)
    active = store.get_active()

    # Try exact path first, then basename
    entries = index.get(filepath) or index.get(basename) or []

    if entries:
        results = []
        for entry in entries:
            if entry["addr"] not in active:
                continue
            node = store.get(entry["addr"])
            if not node:
                continue
            anchors = node.meta.get("anchors", [])
            idx = entry["anchor_idx"]
            if idx < len(anchors) and anchors[idx].get("type") == "content_hash":
                results.append({
                    "node": node,
                    "anchor": anchors[idx],
                    "anchor_idx": idx,
                })
        return results

    # Cold start: scan active set, build index as side effect
    results = []
    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        for i, anchor in enumerate(node.meta.get("anchors", [])):
            if anchor.get("type") != "content_hash":
                continue
            anchor_file = anchor.get("file", "")
            if anchor_file == filepath or os.path.basename(anchor_file) == basename:
                results.append({"node": node, "anchor": anchor, "anchor_idx": i})

    if results:
        # Populate index from what we found
        index_updates: dict = {}
        for item in results:
            f = item["anchor"].get("file", "")
            if f:
                index_updates.setdefault(f, [])
                entry = {"addr": item["node"].addr, "anchor_idx": item["anchor_idx"]}
                if entry not in index_updates[f]:
                    index_updates[f].append(entry)
        merged = load_file_index(store)
        for k, v in index_updates.items():
            merged.setdefault(k, [])
            for e in v:
                if e not in merged[k]:
                    merged[k].append(e)
        _save_file_index(store, merged)

    return results


# ───────────────────────────────────────────────────────────────────
# Drift detection
# ───────────────────────────────────────────────────────────────────

def detect_drift(filepath: str, store: Store,
                 project_root: Optional[Path] = None) -> list[dict]:
    """Check all anchors for a file after an edit.

    Returns list of {node, anchor, anchor_idx, result} for anchors
    that are drifted (found but hash changed) or missing (not found).
    """
    if project_root is None:
        from mnemo_verify import _resolve_project_root
        project_root = _resolve_project_root() or Path.cwd()

    fp = Path(filepath)
    if not fp.is_absolute():
        fp = project_root / fp

    anchored = get_anchors_for_file(filepath, store)
    drifted = []
    for item in anchored:
        result = find_anchor_in_file(item["anchor"], fp)
        if not result["found"] or result["drifted"]:
            drifted.append({
                "node": item["node"],
                "anchor": item["anchor"],
                "anchor_idx": item["anchor_idx"],
                "result": result,
            })
    return drifted


def mark_drifted(node: Node, anchor_idx: int,
                 current_hash: Optional[str], store: Store) -> None:
    """Mark an anchor as drifted in node meta (no addr change)."""
    anchors = node.meta.get("anchors", [])
    if anchor_idx < len(anchors):
        anchors[anchor_idx]["drifted"] = True
        if current_hash:
            anchors[anchor_idx]["drift_hash"] = current_hash
        anchors[anchor_idx]["drift_detected"] = time.time()
        node.meta["anchors"] = anchors
        store.put(node)  # meta mutation — addr unchanged

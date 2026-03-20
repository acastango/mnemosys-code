"""
mnemo_fs.py — Filesystem integration layer

Full backend for tree-aware file operations used by memory_read,
memory_write, memory_edit, memory_grep, memory_glob MCP tools.

Every write/edit operation:
  - Checks for agent conflicts before touching the file
  - Auto-claims the change as a history node (opt-out via claim=False)
  - Surfaces stale content-hash anchors

Every read:
  - Surfaces tree context header above file content
  - Checks for drifted content-hash anchors

Grep / Glob:
  - Results annotated with tree coverage per file

== Path resolution ==
1. Absolute path → used as-is
2. Relative path → resolved from MNEMO_PROJECT_ROOT env or .mnemo/ parent
3. Fallback → CWD

For index lookups, paths stored as POSIX-style relative from project root.

== File index ==
Uses mnemo_anchor.py's existing index/by_file.json:
  {relative_path: [{addr, anchor_idx}]}

== Auto-claim format ==
Write: "Wrote {rel_path}: {first 120 chars}"
Edit:  "Edited {rel_path}: {old[:50]} → {new[:50]}"
domain: "history", anchor: {"type": "file", "path": rel_path}

== Conflict detection ==
Scans active store for other agents' nodes on same file.
Auto-pings: "low" urgency by default, "high" if any node has requires_response=True
"""

import fnmatch
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from mnemo import Store, Node
from mnemo_anchor import load_file_index, find_anchor_in_file


# ───────────────────────────────────────────────────────────────────
# Path resolution
# ───────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Resolve project root: MNEMO_PROJECT_ROOT env → .mnemo/ parent walk → CWD."""
    # Explicit env override
    env_root = os.environ.get("MNEMO_PROJECT_ROOT", "")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # Walk up from CWD looking for .mnemo/
    current = Path.cwd()
    home = Path.home()
    while True:
        candidate = current / ".mnemo"
        if candidate.is_dir() and current != home:
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return Path.cwd()


def resolve_path(path: str, project_root: Path | None = None) -> Path:
    """Absolute paths used as-is; relative resolved from project_root."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = project_root if project_root is not None else get_project_root()
    return root / p


def normalize_path(path: str, project_root: Path | None = None) -> str:
    """POSIX-style relative path from project root. Used as file index key."""
    resolved = resolve_path(path, project_root)
    root = project_root if project_root is not None else get_project_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        # Path is outside project root — return as-is POSIX
        return resolved.as_posix()


# ───────────────────────────────────────────────────────────────────
# File I/O
# ───────────────────────────────────────────────────────────────────

def fs_read(path: str, offset: int = 0, limit: int = 0,
            project_root: Path | None = None) -> tuple[str, int]:
    """Read file. Returns (content, total_line_count). offset=0-based, limit=0 means all."""
    fp = resolve_path(path, project_root)
    text = fp.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total = len(lines)

    start = offset
    if limit > 0:
        end = min(start + limit, total)
    else:
        end = total

    selected = lines[start:end]
    return "".join(selected), total


def fs_write(path: str, content: str,
             project_root: Path | None = None) -> int:
    """Write file, creating parent dirs. Returns bytes written."""
    fp = resolve_path(path, project_root)
    fp.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    fp.write_bytes(encoded)
    return len(encoded)


def fs_edit(path: str, old_string: str, new_string: str,
            replace_all: bool = False,
            project_root: Path | None = None) -> tuple[str, int]:
    """Find-replace edit. Returns (new_content, replacement_count).
    Raises ValueError if old_string not found or not unique (when replace_all=False)."""
    fp = resolve_path(path, project_root)
    content = fp.read_text(encoding="utf-8", errors="replace")

    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in {path}")
    if not replace_all and count > 1:
        raise ValueError(
            f"old_string found {count} times in {path} — use replace_all=True or provide more context"
        )

    new_content = content.replace(old_string, new_string)
    replacements = count if replace_all else 1
    if not replace_all:
        # Replace only first occurrence
        new_content = content.replace(old_string, new_string, 1)

    fp.write_text(new_content, encoding="utf-8")
    return new_content, replacements


# ───────────────────────────────────────────────────────────────────
# Tree ↔ file linkage
# ───────────────────────────────────────────────────────────────────

def nodes_for_file(store: Store, file_path: str,
                   project_root: Path | None = None) -> list[Node]:
    """Return active nodes referencing this file.
    Uses load_file_index from mnemo_anchor for fast lookup.
    Also scans for {"type": "file", "path": ...} anchors as fallback.
    Deduplicates by addr."""
    rel = normalize_path(file_path, project_root)
    basename = os.path.basename(file_path)
    active = store.get_active()

    seen: set[str] = set()
    results: list[Node] = []

    # Fast path: file index (content_hash anchors)
    index = load_file_index(store)
    for key in (rel, basename):
        for entry in index.get(key, []):
            addr = entry.get("addr", "")
            if addr and addr in active and addr not in seen:
                node = store.get(addr)
                if node:
                    seen.add(addr)
                    results.append(node)

    # Fallback: scan active set for file/grep anchors pointing to this file
    for addr in active:
        if addr in seen:
            continue
        node = store.get(addr)
        if not node:
            continue
        for anchor in node.meta.get("anchors", []):
            anchor_type = anchor.get("type", "")
            if anchor_type in ("file", "grep"):
                anchor_path = anchor.get("path", "")
                if not anchor_path:
                    continue
                anchor_rel = Path(anchor_path).as_posix() if not Path(anchor_path).is_absolute() else anchor_path
                if anchor_rel == rel or os.path.basename(anchor_path) == basename:
                    seen.add(addr)
                    results.append(node)
                    break

    return results


def format_context_header(nodes: list[Node], file_path: str) -> str:
    """Format tree context as header block:
    '── mnemo: auth.py (3 nodes) ──'
    grouped by domain, '  [domain] content preview [addr_short]'
    ends with separator line of dashes"""
    if not nodes:
        return ""

    basename = os.path.basename(file_path)
    lines = [f"── mnemo: {basename} ({len(nodes)} node{'s' if len(nodes) != 1 else ''}) ──"]

    # Group by domain
    by_domain: dict[str, list[Node]] = {}
    for node in nodes:
        domain = node.meta.get("domain", "?")
        by_domain.setdefault(domain, []).append(node)

    for domain, dnodes in by_domain.items():
        for node in dnodes:
            preview = node.content[:100].replace("\n", " ").strip()
            addr_short = node.addr[:8]
            lines.append(f"  [{domain}] {preview} [{addr_short}]")

    lines.append("-" * 60)
    return "\n".join(lines)


def check_stale_anchors(store: Store, file_path: str,
                        project_root: Path | None = None) -> list[dict]:
    """Check content-hash anchors on nodes referencing this file.
    Returns [{node, anchor, status, drifted}] for drifted/missing anchors."""
    rel = normalize_path(file_path, project_root)
    fp = resolve_path(file_path, project_root)

    if not fp.exists():
        return []

    from mnemo_anchor import get_anchors_for_file
    anchored = get_anchors_for_file(rel, store)

    stale = []
    for item in anchored:
        anchor = item["anchor"]
        if anchor.get("type") != "content_hash":
            continue
        result = find_anchor_in_file(anchor, fp)
        if not result["found"] or result["drifted"]:
            status = "missing" if not result["found"] else "drifted"
            stale.append({
                "node": item["node"],
                "anchor": anchor,
                "status": status,
                "drifted": result["drifted"],
            })

    return stale



def auto_claim(store: Store, session_store: Optional[Store],
               file_path: str, summary: str,
               agent_id: str = "", session_id: str = "") -> str:
    """Write history node for a file change. Returns addr.
    Uses session_store if available (preliminary), else project store."""
    target_store = session_store if session_store is not None else store

    rel_path = normalize_path(file_path)

    meta: dict = {
        "domain": "history",
        "source": "conscious",
        "auto_claim": True,
        "ttl_days": 7,
        "anchors": [{"type": "file", "path": rel_path}],
    }
    if agent_id:
        meta["agent_id"] = agent_id
    if session_id:
        meta["session_id"] = session_id

    node = Node(
        type="leaf",
        content=summary,
        meta=meta,
    )
    target_store.put(node)
    active = target_store.get_active()
    active.add(node.addr)
    target_store.set_active(active)
    return node.addr


# ───────────────────────────────────────────────────────────────────
# Grep
# ───────────────────────────────────────────────────────────────────

def fs_grep(pattern: str, search_path: str = ".",
            glob_filter: str = "", context_lines: int = 2,
            project_root: Path | None = None) -> list[dict]:
    """Grep. Returns [{file, line_num, content, before:[str], after:[str]}].
    Tries rg first, falls back to Python re."""
    try:
        return _grep_rg(pattern, search_path, glob_filter, context_lines, project_root)
    except (FileNotFoundError, OSError):
        return _grep_python(pattern, search_path, glob_filter, context_lines, project_root)


def _grep_rg(pattern, search_path, glob_filter, context_lines, project_root) -> list[dict]:
    """rg --json -n -C{N} implementation. Parse begin/match/context JSON objects."""
    root = project_root if project_root is not None else get_project_root()

    resolved_search = resolve_path(search_path, root) if search_path != "." else root

    cmd = ["rg", "--json", "-n", f"-C{context_lines}", pattern, str(resolved_search)]
    if glob_filter:
        cmd.extend(["-g", glob_filter])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # rg exits with 1 when no matches — not an error
    if result.returncode > 1:
        raise OSError(f"rg failed with code {result.returncode}: {result.stderr[:200]}")

    matches: list[dict] = []
    # Track context accumulation: pending_after[match_idx] = lines remaining
    pending_after: dict[int, int] = {}

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        obj_type = obj.get("type", "")
        data = obj.get("data", {})

        if obj_type == "match":
            file_path = data.get("path", {}).get("text", "")
            line_num = data.get("line_number", 0)
            content = data.get("lines", {}).get("text", "").rstrip("\n")
            # Try to make path relative to project root
            try:
                rel = Path(file_path).relative_to(root).as_posix()
            except ValueError:
                rel = file_path

            match_entry = {
                "file": rel,
                "line_num": line_num,
                "content": content,
                "before": [],
                "after": [],
            }
            matches.append(match_entry)

        elif obj_type == "context":
            file_path = data.get("path", {}).get("text", "")
            line_num = data.get("line_number", 0)
            content = data.get("lines", {}).get("text", "").rstrip("\n")
            try:
                rel = Path(file_path).relative_to(root).as_posix()
            except ValueError:
                rel = file_path

            # Find the nearest match for this file
            # Context before a match: line_num < match line_num
            # Context after a match: line_num > match line_num
            # Find the closest match in the same file
            best_match = None
            best_dist = None
            for m in reversed(matches):
                if m["file"] == rel:
                    dist = line_num - m["line_num"]
                    if best_dist is None or abs(dist) < abs(best_dist):
                        best_match = m
                        best_dist = dist
                    break  # reversed — first same-file match is closest preceding

            if best_match is not None:
                if best_dist is not None and best_dist < 0:
                    best_match["before"].append(content)
                else:
                    best_match["after"].append(content)

    return matches


def _grep_python(pattern, search_path, glob_filter, context_lines, project_root) -> list[dict]:
    """Pure Python re fallback. Use fnmatch for glob_filter."""
    root = project_root if project_root is not None else get_project_root()
    search_root = resolve_path(search_path, root) if search_path != "." else root

    try:
        compiled = re.compile(pattern)
    except re.error:
        compiled = re.compile(re.escape(pattern))

    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 ".tox", ".mypy_cache", ".eggs", "dist", "build"}

    matches: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            if glob_filter and not fnmatch.fnmatch(filename, glob_filter):
                continue
            fp = Path(dirpath) / filename
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if compiled.search(line):
                    try:
                        rel = fp.relative_to(root).as_posix()
                    except ValueError:
                        rel = str(fp)
                    before = [lines[j] for j in range(max(0, i - context_lines), i)]
                    after = [lines[j] for j in range(i + 1, min(len(lines), i + 1 + context_lines))]
                    matches.append({
                        "file": rel,
                        "line_num": i + 1,
                        "content": line,
                        "before": before,
                        "after": after,
                    })

    return matches


def format_grep_results(matches: list[dict], store: Store,
                        project_root: Path | None = None) -> str:
    """Format grep results with per-file tree context.
    Header: 'N matches in M files'
    Per file: filename, then '  ↳ [domain] preview [addr]' for each node (max 3)
    Then numbered match lines with before/after context."""
    if not matches:
        return "No matches found."

    # Group by file
    by_file: dict[str, list[dict]] = {}
    for m in matches:
        by_file.setdefault(m["file"], []).append(m)

    total = len(matches)
    num_files = len(by_file)
    lines = [f"{total} match{'es' if total != 1 else ''} in {num_files} file{'s' if num_files != 1 else ''}"]
    lines.append("")

    for file_path, file_matches in by_file.items():
        lines.append(f"{file_path}")

        # Tree context for this file
        nodes = nodes_for_file(store, file_path, project_root)
        for node in nodes[:3]:
            domain = node.meta.get("domain", "?")
            preview = node.content[:80].replace("\n", " ").strip()
            lines.append(f"  ↳ [{domain}] {preview} [{node.addr[:8]}]")

        # Match lines
        for m in file_matches:
            for b in m.get("before", []):
                lines.append(f"  {b}")
            lines.append(f"  {m['line_num']}: {m['content']}")
            for a in m.get("after", []):
                lines.append(f"  {a}")

        lines.append("")

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
# Glob
# ───────────────────────────────────────────────────────────────────

def fs_glob(pattern: str, search_path: str = ".",
            project_root: Path | None = None) -> list[str]:
    """Glob. Returns list of relative POSIX paths."""
    root = project_root if project_root is not None else get_project_root()
    search_root = resolve_path(search_path, root) if search_path != "." else root

    results = []
    for match in sorted(search_root.glob(pattern)):
        if match.is_file():
            try:
                rel = match.relative_to(root).as_posix()
            except ValueError:
                rel = match.as_posix()
            results.append(rel)

    return results


def format_glob_with_coverage(paths: list[str], store: Store,
                               project_root: Path | None = None) -> str:
    """Format glob results with tree coverage.
    Per file: '  auth.py                    ██       2 nodes'
    Footer: 'Tree coverage: K/N files known (X%)'"""
    if not paths:
        return "No files matched."

    lines = []
    known_count = 0

    for rel_path in paths:
        nodes = nodes_for_file(store, rel_path, project_root)
        count = len(nodes)
        if count > 0:
            known_count += 1

        # Build coverage bar
        if count == 0:
            bar = "         "
        elif count == 1:
            bar = "█        "
        elif count <= 3:
            bar = "██       "
        elif count <= 6:
            bar = "████     "
        else:
            bar = "████████ "

        filename = rel_path
        # Pad filename to 40 chars
        padded = filename.ljust(40)
        node_label = f"{count} node{'s' if count != 1 else ''}" if count > 0 else ""
        lines.append(f"  {padded} {bar} {node_label}".rstrip())

    total = len(paths)
    pct = int(known_count * 100 / total) if total > 0 else 0
    lines.append("")
    lines.append(f"Tree coverage: {known_count}/{total} files known ({pct}%)")

    return "\n".join(lines)

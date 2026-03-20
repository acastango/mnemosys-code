"""
mnemo_scan.py — Static codebase scanner: AST extraction → tree claims

Walks a file or directory, extracts structural facts from source code
(module docstrings, class docstrings, public function signatures +
docstrings), and stores them as claims in the tree with content-hash
anchors — the same anchor format memory_write/memory_edit produces.

No LLM in the loop — extraction is deterministic from what the author
wrote. Docstrings are ground truth; this commits them to the tree so:
  - recall surfaces them associatively
  - memory_read injects them inline at the right line

This gives a new instance the same structural foothold that accumulates
naturally from working in a codebase over time.

Idempotent via a scan index at <store>/scan_index.json.
Unchanged files (same content hash) are skipped. Changed files get
their old claims superseded.

Supported languages:
- Python (.py)  — full AST extraction, content-hash anchors per section
- Other         — module-level comment extraction, file anchor only

Usage via MCP:
    memory_scan(".")                  # scan entire project
    memory_scan("mnemo_associate.py") # single file
    memory_scan(".", force=True)      # re-scan everything
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from mnemo import Store, Node, supersede as _supersede
from mnemo_anchor import compute_content_hash, update_file_index, remove_from_file_index
from mnemo_log import emit


_SKIP_DIRS = {
    ".mnemo", ".git", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", "dist", "build", ".next",
    ".claude", "node_modules",
}

_SCAN_INDEX_FILE = "scan_index.json"
_CONTEXT_LINES = 6   # lines captured per section (signature + opening body)


# ───────────────────────────────────────────────────────────────────
# Scan index — tracks file hash → claim addrs for idempotency
# ───────────────────────────────────────────────────────────────────

def _index_path(store: Store) -> Path:
    return store.root / _SCAN_INDEX_FILE


def _load_index(store: Store) -> dict:
    p = _index_path(store)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_index(store: Store, index: dict) -> None:
    p = _index_path(store)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, str(p))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ───────────────────────────────────────────────────────────────────
# Python extraction via AST
# ───────────────────────────────────────────────────────────────────

def _first_line(docstring: str) -> str:
    """Return first non-empty line of a docstring."""
    for line in docstring.splitlines():
        line = line.strip()
        if line:
            return line
    return docstring.strip()


def _arg_list(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a compact argument list string, skipping self/cls."""
    args = [a.arg for a in func_node.args.args if a.arg not in ("self", "cls")]
    if len(args) > 5:
        return ", ".join(args[:5]) + ", ..."
    return ", ".join(args)


def _section_anchor(lines: list[str], lineno: int,
                    rel_path: str, scope: str) -> dict:
    """
    Build a content-hash anchor for a code section.

    Captures _CONTEXT_LINES lines starting at lineno (1-based),
    computes the content hash, and returns the anchor dict in the
    same format memory_map and memory_write produce.
    """
    start = lineno - 1  # 0-based
    end = min(start + _CONTEXT_LINES, len(lines))
    context = "\n".join(line.rstrip() for line in lines[start:end])
    return {
        "type": "content_hash",
        "file": rel_path,
        "content_hash": compute_content_hash(context),
        "context_lines": context,
        "scope": scope,
        "line_hint": lineno,
    }


def extract_python(source: str, rel_path: str) -> list[dict]:
    """
    Extract structural claims from a Python source file.

    Returns list of {"content": str, "domain": str, "anchor": dict}.

    Module docstring  → file anchor (whole-file scope)
    Classes           → content-hash anchor at class definition line
    Public functions  → content-hash anchor at function definition line
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    filename = Path(rel_path).name
    lines = source.splitlines()
    claims = []

    # Module docstring → file-level architecture claim
    module_doc = ast.get_docstring(tree)
    if module_doc:
        first = _first_line(module_doc)
        # Avoid "file.py: file.py — ..." when docstring already starts with filename
        if first.lower().startswith(filename.lower().rstrip(".py")):
            content = first
        else:
            content = f"{filename}: {first}"
        claims.append({
            "content": content,
            "domain": "architecture",
            "anchor": {"type": "file", "path": rel_path},
        })

    # Top-level classes and public functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            claims.append({
                "content": f"{filename} {node.name}: {_first_line(doc)}",
                "domain": "architecture",
                "anchor": _section_anchor(lines, node.lineno, rel_path, "class"),
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            sig = f"{node.name}({_arg_list(node)})"
            claims.append({
                "content": f"{filename} {sig}: {_first_line(doc)}",
                "domain": "patterns",
                "anchor": _section_anchor(lines, node.lineno, rel_path, "function"),
            })

    return claims


# ───────────────────────────────────────────────────────────────────
# Generic extraction for non-Python files
# ───────────────────────────────────────────────────────────────────

def extract_generic(source: str, rel_path: str) -> list[dict]:
    """
    Extract a module-level description from a non-Python file.
    Looks for a substantive comment near the top.
    Returns at most one claim with a file anchor.
    """
    filename = Path(rel_path).name
    for line in source.splitlines()[:20]:
        stripped = line.strip()
        if not stripped:
            continue
        for prefix in ("//", "#", "/*", "*", "---"):
            if stripped.startswith(prefix):
                text = stripped.lstrip("/# *-").strip()
                if len(text) > 15:
                    return [{
                        "content": f"{filename}: {text}",
                        "domain": "architecture",
                        "anchor": {"type": "file", "path": rel_path},
                    }]
        break  # stop at first non-comment line

    return []


# ───────────────────────────────────────────────────────────────────
# Per-file scan
# ───────────────────────────────────────────────────────────────────

def _extract_claims(path: Path, rel_path: str) -> list[dict]:
    """Dispatch to the right extractor based on file extension."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not source.strip():
        return []

    if path.suffix.lower() == ".py":
        return extract_python(source, rel_path)
    return extract_generic(source, rel_path)


def scan_file(
    path: Path,
    rel_path: str,
    store: Store,
    index: dict,
    force: bool = False,
) -> tuple[list[str], bool]:
    """
    Scan one file. Returns (new_claim_addrs, was_scanned).

    Skips if file hash is unchanged and force=False.
    Supersedes old claims if file changed.
    """
    current_hash = _file_hash(path)
    entry = index.get(rel_path, {})

    if not force and entry.get("file_hash") == current_hash:
        return entry.get("claim_addrs", []), False

    claims = _extract_claims(path, rel_path)
    if not claims:
        index[rel_path] = {
            "file_hash": current_hash,
            "claim_addrs": [],
            "scanned_at": time.time(),
        }
        return [], False

    old_addrs = entry.get("claim_addrs", [])
    active = store.get_active()
    new_addrs = []

    for i, claim in enumerate(claims):
        meta = {
            "domain": claim["domain"],
            "source": "scan",
            "confidence": 0.7,
            "anchors": [claim["anchor"]],
        }
        if i < len(old_addrs) and old_addrs[i] in active:
            remove_from_file_index(store, old_addrs[i])
            new_addr = _supersede(
                old_addrs[i], claim["content"], store,
                reason="scan: file updated",
                meta_overrides=meta,
            )
            active = store.get_active()
            # Register new node in file index after supersession
            if claim["anchor"].get("type") == "content_hash":
                new_node = store.get(new_addr)
                if new_node:
                    update_file_index(store, new_node)
        else:
            node = Node(type="leaf", content=claim["content"], meta=meta)
            store.put(node)
            active.add(node.addr)
            if claim["anchor"].get("type") == "content_hash":
                update_file_index(store, node)
            new_addr = node.addr

        new_addrs.append(new_addr)

    # Deactivate leftover old claims (file shrank)
    for old_addr in old_addrs[len(claims):]:
        active.discard(old_addr)

    store.set_active(active)

    index[rel_path] = {
        "file_hash": current_hash,
        "claim_addrs": new_addrs,
        "scanned_at": time.time(),
    }

    return new_addrs, True


# ───────────────────────────────────────────────────────────────────
# Directory walk
# ───────────────────────────────────────────────────────────────────

_DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".rs", ".go", ".c", ".h", ".cpp", ".cs",
    ".rb", ".java", ".kt", ".swift",
}


def collect_files(
    root: Path,
    extensions: Optional[set[str]] = None,
) -> list[tuple[Path, str]]:
    """
    Walk root and return [(abs_path, rel_path)] for scannable files.
    Skips hidden dirs, caches, build artifacts.
    """
    exts = extensions or _DEFAULT_EXTENSIONS
    results = []

    if root.is_file():
        if root.suffix.lower() in exts:
            results.append((root, root.name))
        return results

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            fp = Path(dirpath) / fn
            if fp.suffix.lower() not in exts:
                continue
            if fp.stat().st_size > 200_000:
                continue
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = fn
            results.append((fp, rel))

    return results


# ───────────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────────

def scan(
    target: str,
    store: Store,
    project_root: Optional[Path] = None,
    extensions: Optional[set[str]] = None,
    force: bool = False,
    max_files: int = 200,
) -> dict:
    """
    Scan a file or directory and commit structural claims to the store.

    Args:
        target:       Path to scan (absolute or relative to project_root)
        store:        The node store
        project_root: Used for relative path resolution
        extensions:   File extensions to include (default: common source types)
        force:        Re-scan even unchanged files
        max_files:    Safety cap on files processed per call

    Returns:
        {files_scanned, files_skipped, claims_created, per_file}
    """
    if project_root is None:
        from mnemo_verify import _resolve_project_root
        project_root = _resolve_project_root() or Path.cwd()

    tp = Path(target)
    if not tp.is_absolute():
        tp = project_root / tp

    if not tp.exists():
        return {
            "error": f"path not found: {tp}",
            "files_scanned": 0, "files_skipped": 0, "claims_created": 0,
        }

    files = collect_files(tp, extensions)
    if not files:
        return {"files_scanned": 0, "files_skipped": 0, "claims_created": 0, "per_file": {}}

    if len(files) > max_files:
        files = files[:max_files]

    index = _load_index(store)
    files_scanned = 0
    files_skipped = 0
    claims_created = 0
    per_file: dict[str, int] = {}

    for fp, rel_path in files:
        new_addrs, was_scanned = scan_file(fp, rel_path, store, index, force=force)
        if was_scanned:
            files_scanned += 1
            claims_created += len(new_addrs)
            if new_addrs:
                per_file[rel_path] = len(new_addrs)
        else:
            files_skipped += 1

    _save_index(store, index)

    emit("scan", "conscious",
         f"scanned {files_scanned} files, {claims_created} claims",
         detail={"files_scanned": files_scanned,
                 "files_skipped": files_skipped,
                 "claims_created": claims_created})

    return {
        "files_scanned": files_scanned,
        "files_skipped": files_skipped,
        "claims_created": claims_created,
        "per_file": per_file,
    }

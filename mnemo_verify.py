"""
mnemo_verify.py — Verification anchors for memory claims

Compares anchored claims against the actual codebase. Three anchor types:
- file:       path exists relative to project root
- grep:       pattern found in file (or across *.py files, capped at 50)
- dependency: package name found in imports or requirements files

Zero LLM calls. Pure filesystem checks.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


# Directories to skip during grep walks
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}

REQUIRED_FIELDS = {
    "file": {"path"},
    "grep": {"pattern"},
    "dependency": {"name"},
    "content_hash": {"file", "context_lines"},
}


def _resolve_project_root() -> Optional[Path]:
    """Resolve project root: MNEMO_PROJECT_ROOT env -> git rev-parse -> CWD."""
    env = os.environ.get("MNEMO_PROJECT_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            p = Path(result.stdout.strip())
            if p.is_dir():
                return p
    except Exception:
        pass

    return Path.cwd()


def validate_anchor(anchor: dict) -> bool:
    """Check that an anchor dict has the right shape."""
    if not isinstance(anchor, dict):
        return False
    atype = anchor.get("type", "")
    required = REQUIRED_FIELDS.get(atype)
    if not required:
        return False
    return all(anchor.get(f) for f in required)


def _walk_py_files(root: Path, cap: int = 50) -> list[Path]:
    """Walk project for *.py files, skipping common non-source dirs."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(Path(dirpath) / fn)
                if len(files) >= cap:
                    return files
    return files


def verify_anchor(anchor: dict, project_root: Path) -> dict:
    """Verify a single anchor against the filesystem.

    Returns {anchor, passed: bool, detail: str}.
    """
    atype = anchor.get("type", "")

    if atype == "file":
        target = (project_root / Path(anchor["path"])).resolve()
        if not str(target).startswith(str(project_root.resolve())):
            return {"anchor": anchor, "passed": False,
                    "detail": f"file: {anchor['path']} REJECTED (path traversal)"}
        passed = target.exists()
        detail = (f"file: {anchor['path']} "
                  + ("exists" if passed else "NOT FOUND"))
        return {"anchor": anchor, "passed": passed, "detail": detail}

    elif atype == "grep":
        pattern = anchor["pattern"]
        path = anchor.get("path")
        if path:
            target = (project_root / Path(path)).resolve()
            if not str(target).startswith(str(project_root.resolve())):
                return {"anchor": anchor, "passed": False,
                        "detail": f"grep: {path} REJECTED (path traversal)"}
            if not target.exists():
                return {"anchor": anchor, "passed": False,
                        "detail": f"grep: file '{path}' NOT FOUND"}
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
                passed = pattern in text
                detail = (f"grep: '{pattern}' in {path} "
                          + ("found" if passed else "NOT FOUND"))
                return {"anchor": anchor, "passed": passed, "detail": detail}
            except Exception as e:
                return {"anchor": anchor, "passed": False,
                        "detail": f"grep: error reading '{path}': {e}"}
        else:
            # Search across project *.py files
            for fp in _walk_py_files(project_root):
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    if pattern in text:
                        rel = fp.relative_to(project_root)
                        return {"anchor": anchor, "passed": True,
                                "detail": f"grep: '{pattern}' found in {rel}"}
                except Exception:
                    continue
            return {"anchor": anchor, "passed": False,
                    "detail": f"grep: '{pattern}' not found in any .py file"}

    elif atype == "dependency":
        name = anchor["name"]
        # Check Python imports
        for fp in _walk_py_files(project_root):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                if f"import {name}" in text or f"from {name}" in text:
                    rel = fp.relative_to(project_root)
                    return {"anchor": anchor, "passed": True,
                            "detail": f"dependency: '{name}' imported in {rel}"}
            except Exception:
                continue

        # Check requirements.txt, pyproject.toml
        for req_file in ("requirements.txt", "pyproject.toml"):
            req_path = project_root / req_file
            if req_path.exists():
                try:
                    text = req_path.read_text(encoding="utf-8", errors="replace")
                    if name in text:
                        return {"anchor": anchor, "passed": True,
                                "detail": f"dependency: '{name}' found in {req_file}"}
                except Exception:
                    continue

        return {"anchor": anchor, "passed": False,
                "detail": f"dependency: '{name}' not found in imports or requirements"}

    elif atype == "content_hash":
        from mnemo_anchor import find_anchor_in_file
        file_path = anchor.get("file", "")
        if not file_path:
            return {"anchor": anchor, "passed": False,
                    "detail": "content_hash: no file specified"}
        target = (project_root / Path(file_path)).resolve()
        if not str(target).startswith(str(project_root.resolve())):
            return {"anchor": anchor, "passed": False,
                    "detail": f"content_hash: {file_path} REJECTED (path traversal)"}
        result = find_anchor_in_file(anchor, target)
        if not result["found"]:
            return {"anchor": anchor, "passed": False,
                    "detail": f"content_hash: context not found in {file_path}"}
        if result["drifted"]:
            stored = anchor.get("content_hash", "?")[:8]
            current = (result["current_hash"] or "?")[:8]
            return {"anchor": anchor, "passed": False,
                    "detail": f"content_hash: DRIFTED in {file_path} "
                               f"(expected {stored}, found {current})"}
        return {"anchor": anchor, "passed": True,
                "detail": f"content_hash: ok in {file_path} at line {result['line_num']}"}

    return {"anchor": anchor, "passed": False,
            "detail": f"unknown anchor type: {atype}"}


def verify_node(node, project_root: Path) -> dict:
    """Verify all anchors on a node.

    Returns {addr, content, total, passed, failed, results}.
    """
    anchors = node.meta.get("anchors", [])
    results = [verify_anchor(a, project_root) for a in anchors]
    passed = sum(1 for r in results if r["passed"])
    return {
        "addr": node.addr,
        "content": node.content,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def verify_active(store, project_root: Path) -> list[dict]:
    """Verify all anchored active nodes. Returns only nodes with failures."""
    failures = []
    for addr in store.get_active():
        node = store.get(addr)
        if not node:
            continue
        anchors = node.meta.get("anchors", [])
        if not anchors:
            continue
        result = verify_node(node, project_root)
        if result["failed"] > 0:
            # Only include failed results for brevity
            result["results"] = [r for r in result["results"] if not r["passed"]]
            failures.append(result)
    return failures

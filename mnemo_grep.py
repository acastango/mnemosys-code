"""
mnemo_grep.py — Tree-aware pattern search

Given a regex pattern and an intent (why you're searching), produces:
1. Tree check: does the tree already answer this question?
2. Path guidance: which files should you search based on architecture knowledge?
3. Grep results: actual pattern matches, annotated with tree context
4. Annotations: for each matched file, what the tree knows about it

The intent parameter is what makes this different from raw grep —
it lets the tree narrow the search and explain the results.

Zero LLM calls. Regex + tree retrieval + filesystem.
"""

import os
import re
from pathlib import Path
from typing import Optional

from mnemo import Store, Node
from mnemo_associate import retrieve_relevant
from mnemo_verify import _resolve_project_root
from mnemo_anchor import get_anchors_for_file, find_anchor_in_file

# Directories to skip
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".tox", ".mypy_cache", ".eggs", "dist", "build"}

# Output limits
_MAX_MATCHES_PER_FILE = 15
_MAX_FILES = 20
_OUTPUT_CAP = 10000


# -------------------------------------------------------------------
# Tree-based path guidance
# -------------------------------------------------------------------

def _extract_suggested_paths(nodes: list[Node], intent_keywords: set[str]) -> list[str]:
    """Extract file paths from tree nodes, ranked by relevance to the intent.

    Returns paths that tree nodes mention in context of the search intent.
    These become the 'search here first' guidance.
    """
    # Pattern: filenames referenced in node content
    file_pattern = re.compile(
        r'([\w./\\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|toml|yaml|yml|json|md|cfg))'
    )

    path_scores: dict[str, float] = {}
    for node in nodes:
        # Higher weight for architecture/patterns nodes — they map the codebase
        weight = 1.0
        domain = node.meta.get("domain", "")
        if domain in ("architecture", "patterns"):
            weight = 2.0
        elif domain == "issues":
            weight = 1.5

        for match in file_pattern.finditer(node.content):
            path = match.group(1)
            path_scores[path] = path_scores.get(path, 0) + weight

        # Anchors are high-confidence file references
        for anchor in node.meta.get("anchors", []):
            if anchor.get("path"):
                path_scores[anchor["path"]] = path_scores.get(anchor["path"], 0) + weight * 2

    return sorted(path_scores, key=path_scores.get, reverse=True)


def _annotate_file(filepath: str, nodes: list[Node]) -> list[str]:
    """Find tree nodes that mention this file. Returns annotation lines."""
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]
    annotations = []

    for node in nodes:
        content_lower = node.content.lower()
        # Check if node mentions this file by name or module name
        if basename.lower() in content_lower or name_no_ext.lower() in content_lower:
            domain = node.meta.get("domain", "?")
            addr = node.addr[:8]
            # Extract the relevant sentence/fragment
            snippet = node.content[:150]
            annotations.append(f"  [{domain}] {addr}: {snippet}")

    return annotations[:3]  # cap per file


# -------------------------------------------------------------------
# Content-hash anchor injection per match line
# -------------------------------------------------------------------

def _resolve_file_anchors(filepath: str, store: Store,
                          project_root: Path) -> list[tuple[int, Node]]:
    """Resolve all content_hash anchors for a file to their line positions.

    Returns sorted list of (line_num, node) — one entry per anchor that
    was found in the file. Resolved once per file, used for O(1) per-match
    lookup via bisect.
    """
    anchored = get_anchors_for_file(filepath, store)
    if not anchored:
        return []

    fp = Path(filepath)
    if not fp.is_absolute():
        fp = project_root / filepath

    resolved = []
    for item in anchored:
        result = find_anchor_in_file(item["anchor"], fp)
        if result["found"]:
            resolved.append((result["line_num"], item["node"],
                             result["drifted"]))

    resolved.sort(key=lambda x: x[0])
    return resolved


def _comprehension_for_line(line_num: int,
                             resolved: list[tuple]) -> Optional[str]:
    """Find the anchor whose section contains line_num.

    Uses the resolved anchor list (sorted by line). The containing section
    is the anchor with the largest line_num that is still <= match line_num.
    Returns a formatted annotation string, or None if no anchor covers this line.
    """
    if not resolved:
        return None

    # Find rightmost anchor at or before this line
    best = None
    for anchor_line, node, drifted in resolved:
        if anchor_line <= line_num:
            best = (anchor_line, node, drifted)
        else:
            break  # sorted, no point continuing

    if best is None:
        return None

    _, node, drifted = best
    domain = node.meta.get("domain", "?")
    drift_tag = " [DRIFTED]" if drifted else ""
    return f"  ↳ [{domain}]{drift_tag} {node.addr[:8]}: {node.content[:160]}"


# -------------------------------------------------------------------
# Core grep with tree awareness
# -------------------------------------------------------------------

def _regex_grep(pattern: str, project_root: Path,
                priority_paths: list[str] = None,
                glob_filter: str = None,
                path_scope: str = None) -> list[dict]:
    """Run regex grep across project files.

    Returns list of {file, matches: [{line_num, text}], priority: bool}.
    Priority files (suggested by tree) are searched first and flagged.
    """
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return [{"file": "(error)", "matches": [{"line_num": 0, "text": f"Invalid regex: {e}"}], "priority": False}]

    # Determine which extensions to search
    exact_name: str | None = None
    if glob_filter:
        if glob_filter.startswith("*"):
            # "*.py" → extension filter
            extensions: set[str] | None = {glob_filter.replace("*", "")}
        elif "*" not in glob_filter:
            # "CLAUDE_MNEMO.md" → exact filename match
            extensions = None
            exact_name = glob_filter
        else:
            # "mnemo_*.py" → fnmatch pattern
            import fnmatch as _fnmatch
            extensions = None
            exact_name = glob_filter  # handled via fnmatch below
    else:
        extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go",
                      ".java", ".toml", ".yaml", ".yml", ".json", ".md"}

    results = []
    searched = set()

    # Phase 1: search priority paths first (tree-guided)
    if priority_paths:
        for rel_path in priority_paths:
            if len(results) >= _MAX_FILES:
                break
            fp = project_root / rel_path
            if not fp.exists() or not fp.is_file():
                continue
            searched.add(str(fp))
            matches = _grep_file(compiled, fp)
            if matches:
                results.append({
                    "file": rel_path,
                    "matches": matches,
                    "priority": True,
                })

    # Phase 2: search remaining files
    if path_scope:
        search_root = project_root / path_scope
        if not search_root.is_dir():
            search_root = project_root
    else:
        search_root = project_root

    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if len(results) >= _MAX_FILES:
                return results
            if exact_name is not None:
                import fnmatch as _fnmatch
                if not _fnmatch.fnmatch(fn, exact_name):
                    continue
            elif extensions is not None:
                if not any(fn.endswith(ext) for ext in extensions):
                    continue
            fp = Path(dirpath) / fn
            if str(fp) in searched:
                continue

            matches = _grep_file(compiled, fp)
            if matches:
                try:
                    rel = str(fp.relative_to(project_root))
                except ValueError:
                    rel = str(fp)
                results.append({
                    "file": rel,
                    "matches": matches,
                    "priority": False,
                })

    return results


def _grep_file(compiled: re.Pattern, filepath: Path) -> list[dict]:
    """Grep a single file. Returns list of {line_num, text}."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    matches = []
    for i, line in enumerate(text.splitlines(), 1):
        if compiled.search(line):
            matches.append({
                "line_num": i,
                "text": line.rstrip()[:150],
            })
            if len(matches) >= _MAX_MATCHES_PER_FILE:
                break
    return matches


# -------------------------------------------------------------------
# Main grep function
# -------------------------------------------------------------------

def grep(pattern: str, intent: str, store: Store,
         session_context: dict = None,
         project_root: Path = None,
         glob_filter: str = None,
         path: str = None) -> str:
    """Tree-aware pattern search.

    Args:
        pattern: Regex pattern to search for
        intent: Why you're searching — used to recall tree context
        store: The node store
        session_context: Session tracking state
        project_root: Project root directory
        glob_filter: File extension filter (e.g. "*.py")
        path: Subdirectory to scope the search
    """
    if project_root is None:
        project_root = _resolve_project_root() or Path.cwd()

    parts = []

    # ── Step 1: Tree check ──────────────────────────────────────────
    # Does the tree already know something about this intent?
    scored = retrieve_relevant(intent, store,
                               session_context=session_context,
                               max_nodes=5)
    nodes = [item["node"] for item in scored]

    tree_context = []
    for item in scored:
        n = item["node"]
        domain = n.meta.get("domain", "?")
        # Only include nodes that seem genuinely relevant (score > 1.0)
        if item["score"] > 1.0:
            tree_context.append(f"  [{domain}] {n.addr[:8]}: {n.content[:200]}")

    if tree_context:
        parts.append("## Tree context (what's already known)\n")
        parts.extend(tree_context)
        parts.append("")

    # ── Step 2: Path guidance ───────────────────────────────────────
    # Use architecture knowledge to suggest where to search first
    suggested = _extract_suggested_paths(nodes, set())

    if suggested:
        parts.append("## Suggested paths (from tree)\n")
        for p in suggested[:5]:
            exists = (project_root / p).exists()
            mark = "exists" if exists else "MISSING"
            parts.append(f"  {p} ({mark})")
        parts.append("")

    # ── Step 3: Grep ────────────────────────────────────────────────
    results = _regex_grep(pattern, project_root,
                          priority_paths=suggested,
                          glob_filter=glob_filter,
                          path_scope=path)

    if not results:
        parts.append(f"## No matches for `{pattern}`\n")
        if tree_context:
            parts.append("The tree has context above — the answer may already be known.")
        return "\n".join(parts)

    # ── Step 4: Annotated results ───────────────────────────────────
    total_matches = sum(len(r["matches"]) for r in results)
    parts.append(f"## Matches ({total_matches} in {len(results)} files)\n")

    total_chars = 0
    for result in results:
        filepath = result["file"]
        priority_tag = " (tree-suggested)" if result["priority"] else ""

        # File header with tree annotations
        section_lines = [f"**{filepath}**{priority_tag}"]

        # Add file-level tree annotations (nodes mentioning this file by name)
        annotations = _annotate_file(filepath, nodes)
        if annotations:
            section_lines.append("  Tree knows:")
            section_lines.extend(annotations)

        # Resolve content_hash anchor positions for this file once
        resolved = _resolve_file_anchors(filepath, store, project_root)

        # Add grep matches with per-match comprehension injection
        for m in result["matches"]:
            section_lines.append(f"  {m['line_num']}: {m['text']}")
            comprehension = _comprehension_for_line(m["line_num"], resolved)
            if comprehension:
                section_lines.append(comprehension)

        section = "\n".join(section_lines)
        if total_chars + len(section) > _OUTPUT_CAP:
            remaining = len(results) - results.index(result)
            parts.append(f"  ... and {remaining} more files")
            break
        parts.append(section)
        total_chars += len(section)

    parts.append("")

    # ── Step 5: Summary ─────────────────────────────────────────────
    priority_count = sum(1 for r in results if r["priority"])
    parts.append("## Trace summary\n")
    parts.append(f"  Pattern: `{pattern}`")
    parts.append(f"  Intent: {intent}")
    parts.append(f"  Tree nodes consulted: {len(tree_context)}")
    parts.append(f"  Files matched: {len(results)} ({priority_count} tree-suggested)")
    parts.append(f"  Total matches: {total_matches}")

    result_text = "\n".join(parts)
    if len(result_text) > _OUTPUT_CAP:
        result_text = result_text[:_OUTPUT_CAP] + "\n\n... (output truncated)"

    return result_text

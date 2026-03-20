"""
mnemo_explore.py — Tree-aware codebase exploration

Given a topic, produces a reasoning trace:
1. Recall: what does the tree already know?
2. Locate: extract file references from tree nodes, find relevant code
3. Read: targeted file reads based on tree pointers
4. Diff: what does the code show that the tree doesn't capture?
5. Verify: do anchored claims still hold?
6. Report: unified picture with knowledge, code evidence, gaps, tensions

Zero LLM calls. Uses the tree + filesystem. The calling model interprets.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from mnemo import Store, Node
from mnemo_associate import retrieve_relevant, extract_signals
from mnemo_verify import _resolve_project_root, verify_node

# Directories to skip during file walks
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".tox", ".mypy_cache", ".eggs", "dist", "build"}

# Max chars of file content to include per file
_FILE_CAP = 1500

# Max files to grep through
_GREP_CAP = 80

# Max total output chars
_OUTPUT_CAP = 12000


# -------------------------------------------------------------------
# File reference extraction
# -------------------------------------------------------------------

_FILE_PATTERN = re.compile(
    r'(?:^|\s|[(`\'":])'        # preceded by whitespace, paren, quote, colon
    r'([\w./\\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|toml|yaml|yml|json|md|txt|cfg|ini))'
    r'(?=[\s)\'",;:\]|}]|$)',   # followed by whitespace, punctuation, or EOL
    re.MULTILINE,
)

_FUNC_PATTERN = re.compile(
    r'(?:def|class|function|fn|func)\s+([\w_]+)',
)

_MODULE_PATTERN = re.compile(
    r'\b(mnemo\w*)\b',
)


def _extract_file_refs(nodes: list[Node]) -> set[str]:
    """Extract file path references from node content."""
    refs = set()
    for node in nodes:
        for match in _FILE_PATTERN.finditer(node.content):
            refs.add(match.group(1))
        # Also pull from anchors
        for anchor in node.meta.get("anchors", []):
            if anchor.get("path"):
                refs.add(anchor["path"])
    return refs


def _extract_code_symbols(nodes: list[Node]) -> set[str]:
    """Extract function/class names mentioned in node content."""
    symbols = set()
    for node in nodes:
        for match in _FUNC_PATTERN.finditer(node.content):
            symbols.add(match.group(1))
    return symbols


def _extract_keywords(topic: str) -> set[str]:
    """Extract search keywords from the topic string."""
    signals = extract_signals(topic)
    return signals["keywords"]


# -------------------------------------------------------------------
# Filesystem exploration
# -------------------------------------------------------------------

def _grep_project(keywords: set[str], project_root: Path,
                  extensions: set[str] = None) -> dict[str, list[str]]:
    """Search project files for keyword matches.

    Returns {filepath: [matching lines]} for files with hits.
    """
    if extensions is None:
        extensions = {".py"}

    hits: dict[str, list[str]] = {}
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not any(fn.endswith(ext) for ext in extensions):
                continue
            file_count += 1
            if file_count > _GREP_CAP:
                return hits

            fp = Path(dirpath) / fn
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            matching_lines = []
            for i, line in enumerate(text.splitlines(), 1):
                line_lower = line.lower()
                if any(kw in line_lower for kw in keywords):
                    matching_lines.append(f"  {i}: {line.rstrip()[:120]}")

            if matching_lines:
                rel = str(fp.relative_to(project_root))
                hits[rel] = matching_lines[:10]  # cap lines per file

    return hits


def _read_file_region(filepath: Path, keywords: set[str],
                      context: int = 5) -> Optional[str]:
    """Read regions of a file around keyword matches.

    Returns a compact view: matching lines with context.
    """
    if not filepath.exists():
        return None

    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    # Find lines that match any keyword
    match_indices = set()
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            match_indices.add(i)

    if not match_indices:
        return None

    # Expand with context, merge overlapping regions
    regions = set()
    for idx in match_indices:
        for i in range(max(0, idx - context), min(len(lines), idx + context + 1)):
            regions.add(i)

    # Build output
    sorted_regions = sorted(regions)
    parts = []
    prev = -2
    for i in sorted_regions:
        if i > prev + 1:
            parts.append("  ...")
        parts.append(f"  {i+1}: {lines[i].rstrip()[:120]}")
        prev = i

    result = "\n".join(parts)
    return result[:_FILE_CAP]


# -------------------------------------------------------------------
# Gap and tension detection
# -------------------------------------------------------------------

def _detect_gaps(nodes: list[Node], grep_hits: dict[str, list[str]],
                 file_refs: set[str], project_root: Path) -> list[str]:
    """Find things in the code that the tree doesn't capture."""
    gaps = []

    # Files with grep hits that aren't referenced by any tree node
    known_files = file_refs | {
        ref.replace("\\", "/") for ref in file_refs
    }
    for filepath in grep_hits:
        normalized = filepath.replace("\\", "/")
        basename = os.path.basename(normalized)
        if (normalized not in known_files and
                basename not in known_files and
                not any(normalized in n.content or basename in n.content
                        for n in nodes)):
            gaps.append(f"Code in {filepath} matches topic but has no tree coverage")

    return gaps[:5]  # cap


def _detect_tensions(nodes: list[Node], project_root: Path) -> list[str]:
    """Find tree claims that may conflict with current code state."""
    tensions = []

    for node in nodes:
        anchors = node.meta.get("anchors", [])
        if not anchors:
            continue

        result = verify_node(node, project_root)
        if result["failed"] > 0:
            for r in result["results"]:
                if not r["passed"]:
                    tensions.append(
                        f"[{node.addr[:8]}] anchor failed: {r['detail']} "
                        f"— claim: {node.content[:80]}"
                    )

    return tensions[:5]  # cap


# -------------------------------------------------------------------
# Main explore function
# -------------------------------------------------------------------

def explore(topic: str, store: Store,
            session_context: dict = None,
            project_root: Path = None,
            deep: bool = False) -> str:
    """Tree-aware codebase exploration.

    Returns a structured reasoning trace the calling model can act on.
    """
    if project_root is None:
        project_root = _resolve_project_root() or Path.cwd()

    max_nodes = 12 if deep else 8
    parts = []

    # ── Step 1: Recall ──────────────────────────────────────────────
    scored = retrieve_relevant(topic, store,
                               session_context=session_context,
                               max_nodes=max_nodes)
    nodes = [item["node"] for item in scored]

    if nodes:
        parts.append("## What the tree knows\n")
        for item in scored:
            n = item["node"]
            domain = n.meta.get("domain", "?")
            priority_tag = ""
            if n.meta.get("priority", 0) > 0:
                priority_tag = f" [priority={n.meta['priority']}]"
            parts.append(
                f"[{domain}] {n.addr[:8]}{priority_tag}: "
                f"{n.content[:200]}"
            )
        parts.append("")
    else:
        parts.append("## What the tree knows\n")
        parts.append("Nothing found — this topic has no tree coverage yet.\n")

    # ── Step 2: Locate ──────────────────────────────────────────────
    file_refs = _extract_file_refs(nodes)
    symbols = _extract_code_symbols(nodes)
    keywords = _extract_keywords(topic)

    # Merge symbol names into search keywords
    search_terms = keywords | {s.lower() for s in symbols}

    parts.append("## File references from tree\n")
    if file_refs:
        for ref in sorted(file_refs):
            exists = (project_root / ref).exists()
            mark = "✓" if exists else "✗ MISSING"
            parts.append(f"  {ref} ({mark})")
    else:
        parts.append("  (none — no file paths found in recalled nodes)")
    parts.append("")

    # ── Step 3: Code exploration ────────────────────────────────────
    grep_hits = _grep_project(search_terms, project_root)

    if grep_hits:
        parts.append("## Code matches\n")
        total_chars = 0
        for filepath, lines in sorted(grep_hits.items()):
            section = f"**{filepath}**\n" + "\n".join(lines[:5])
            if total_chars + len(section) > _OUTPUT_CAP // 2:
                parts.append(f"  ... and {len(grep_hits) - len(parts) + 4} more files")
                break
            parts.append(section)
            total_chars += len(section)
        parts.append("")

    # ── Step 4: Targeted reads ──────────────────────────────────────
    # Read files the tree points to, showing regions relevant to topic
    if file_refs and deep:
        parts.append("## Targeted reads\n")
        read_count = 0
        for ref in sorted(file_refs):
            if read_count >= 3:
                break
            filepath = project_root / ref
            region = _read_file_region(filepath, search_terms)
            if region:
                parts.append(f"**{ref}** (relevant regions):\n{region}\n")
                read_count += 1
        if read_count == 0:
            parts.append("  (no matching regions in referenced files)")
        parts.append("")

    # ── Step 5: Gaps ────────────────────────────────────────────────
    gaps = _detect_gaps(nodes, grep_hits, file_refs, project_root)
    if gaps:
        parts.append("## Gaps (code without tree coverage)\n")
        for gap in gaps:
            parts.append(f"  • {gap}")
        parts.append("")

    # ── Step 6: Tensions ────────────────────────────────────────────
    tensions = _detect_tensions(nodes, project_root)
    if tensions:
        parts.append("## Tensions (tree vs code)\n")
        for tension in tensions:
            parts.append(f"  ⚡ {tension}")
        parts.append("")

    # ── Step 7: Summary ─────────────────────────────────────────────
    parts.append("## Trace summary\n")
    parts.append(f"  Tree nodes recalled: {len(nodes)}")
    parts.append(f"  File refs from tree: {len(file_refs)}")
    parts.append(f"  Code files with matches: {len(grep_hits)}")
    parts.append(f"  Gaps detected: {len(gaps)}")
    parts.append(f"  Tensions detected: {len(tensions)}")
    if gaps:
        parts.append(f"  → Consider claiming knowledge about: "
                      f"{', '.join(g.split(' matches ')[0].replace('Code in ', '') for g in gaps[:3])}")

    result = "\n".join(parts)

    # Hard cap on total output
    if len(result) > _OUTPUT_CAP:
        result = result[:_OUTPUT_CAP] + "\n\n... (output truncated)"

    return result

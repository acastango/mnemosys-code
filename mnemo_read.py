"""
mnemo_read.py — Tree-annotated file reading

Reads a file and overlays it with tree knowledge:
1. Header: what the tree knows about this file/module
2. Inline annotations: known issues, decisions, patterns at specific lines
3. Summary: gaps in tree coverage for this file

Like reading code with a senior dev's annotations — "known bug here",
"this was designed this way because Y", "this section handles X".

Zero LLM calls. File read + tree retrieval + line matching.
"""

import os
import re
from pathlib import Path
from typing import Optional

from mnemo import Store, Node
from mnemo_associate import retrieve_relevant
from mnemo_verify import _resolve_project_root
from mnemo_anchor import get_anchors_for_file, find_anchor_in_file


# Output limits
_MAX_FILE_LINES = 300
_OUTPUT_CAP = 15000


# -------------------------------------------------------------------
# Line reference extraction from tree nodes
# -------------------------------------------------------------------

# Patterns like "mnemo_mcp.py:417", "mnemo.py:1011", "line 42", "lines 400-420"
_LINE_REF = re.compile(
    r'(?:(\w[\w.-]+\.(?:py|js|ts|rs|go|java))'  # filename
    r'[:\s]*)?'                                   # optional separator
    r'(?:lines?\s*)'                              # "line" or "lines"
    r'(\d+)(?:\s*[-–]\s*(\d+))?'                  # line number or range
    , re.IGNORECASE
)

# More specific: "filename:linenum" pattern (like mnemo_mcp.py:417)
_FILE_LINE_REF = re.compile(
    r'(\w[\w.-]+\.(?:py|js|ts|rs|go|java)):(\d+)(?:\s*[-–]\s*(\d+))?'
)


def _extract_line_annotations(nodes: list[Node],
                              target_basename: str) -> dict[int, list[str]]:
    """Extract line-specific annotations from tree nodes.

    Returns {line_number: [annotation strings]} for lines mentioned
    in nodes that reference this file.
    """
    annotations: dict[int, list[str]] = {}

    for node in nodes:
        domain = node.meta.get("domain", "?")
        addr = node.addr[:8]

        # Check file:line references
        for match in _FILE_LINE_REF.finditer(node.content):
            filename = match.group(1)
            if filename.lower() != target_basename.lower():
                continue

            start = int(match.group(2))
            end = int(match.group(3)) if match.group(3) else start

            # Extract the relevant fragment around this reference
            # Find the sentence containing this match
            pos = match.start()
            content = node.content
            # Look backward for sentence start
            sent_start = max(0, content.rfind(".", 0, pos) + 1)
            # Look forward for sentence end
            sent_end = content.find(".", pos)
            if sent_end == -1:
                sent_end = len(content)
            fragment = content[sent_start:sent_end + 1].strip()[:200]

            annotation = f"  ## [{domain}] {addr} (lines {start}-{end}): {fragment}"

            # Annotate only the first line of the range, not every line
            if start not in annotations:
                annotations[start] = []
            annotations[start].append(annotation)

    return annotations


def _find_relevant_nodes(filepath: str, store: Store,
                         session_context: dict = None) -> list[dict]:
    """Find tree nodes relevant to this file.

    Uses both direct text matching and retrieval.
    """
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]

    # Search by filename/module name
    scored = retrieve_relevant(
        f"{basename} {name_no_ext}",
        store,
        session_context=session_context,
        max_nodes=10,
    )

    # Filter to nodes that actually mention this file
    relevant = []
    for item in scored:
        content_lower = item["node"].content.lower()
        if (basename.lower() in content_lower or
                name_no_ext.lower() in content_lower):
            relevant.append(item)

    # Nodes with file/grep anchors pointing to this file
    for addr in store.get_active():
        node = store.get(addr)
        if not node:
            continue
        for anchor in node.meta.get("anchors", []):
            if anchor.get("type") in ("file", "grep"):
                anchor_path = anchor.get("path", "")
                if (os.path.basename(anchor_path).lower() == basename.lower() and
                        not any(item["node"].addr == node.addr for item in relevant)):
                    relevant.append({"node": node, "score": 0.5, "reasons": ["anchor"]})

    # Nodes with content_hash anchors pointing to this file — use file index
    for item in get_anchors_for_file(filepath, store):
        node = item["node"]
        if not any(r["node"].addr == node.addr for r in relevant):
            relevant.append({
                "node": node,
                "score": 0.85,
                "reasons": ["content_hash_anchor"],
                "_anchor": item["anchor"],
                "_anchor_idx": item["anchor_idx"],
            })

    return relevant


# -------------------------------------------------------------------
# Section detection
# -------------------------------------------------------------------

def _detect_sections(lines: list[str], ext: str) -> list[dict]:
    """Detect code sections (classes, functions, major blocks).

    Returns list of {line, type, name} for section headers.
    """
    sections = []

    if ext in (".py",):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("class "):
                match = re.match(r'class\s+(\w+)', stripped)
                if match:
                    sections.append({"line": i, "type": "class", "name": match.group(1)})
            elif stripped.startswith("def "):
                match = re.match(r'def\s+(\w+)', stripped)
                if match:
                    sections.append({"line": i, "type": "function", "name": match.group(1)})
    elif ext in (".js", ".ts", ".tsx", ".jsx"):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            match = re.match(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', stripped)
            if match:
                sections.append({"line": i, "type": "function", "name": match.group(1)})
            match = re.match(r'(?:export\s+)?class\s+(\w+)', stripped)
            if match:
                sections.append({"line": i, "type": "class", "name": match.group(1)})

    return sections


def _annotate_sections(sections: list[dict],
                       nodes: list[dict]) -> dict[int, str]:
    """Match tree nodes to code sections by name.

    Returns {section_line: annotation} for sections mentioned in tree.
    """
    section_annotations: dict[int, str] = {}

    for section in sections:
        name = section["name"]
        name_lower = name.lower()

        for item in nodes:
            node = item["node"]
            if name_lower in node.content.lower():
                domain = node.meta.get("domain", "?")
                # Extract fragment mentioning this section
                idx = node.content.lower().find(name_lower)
                start = max(0, node.content.rfind(" ", 0, max(0, idx - 30)))
                fragment = node.content[start:idx + len(name) + 80].strip()[:150]
                annotation = f"  ## [{domain}] {node.addr[:8]}: ...{fragment}..."

                if section["line"] not in section_annotations:
                    section_annotations[section["line"]] = annotation

    return section_annotations


# -------------------------------------------------------------------
# Main read function
# -------------------------------------------------------------------

def read(filepath: str, store: Store,
         session_context: dict = None,
         project_root: Path = None,
         offset: int = 0,
         limit: int = 0,
         visit: int = 1) -> str:
    """Read a file with tree knowledge overlay.

    Args:
        filepath: Path to the file (absolute or relative to project_root)
        store: The node store
        session_context: Session tracking state
        project_root: Project root directory
        offset: Start reading from this line (0-based, default: start)
        limit: Max lines to read (0 = up to _MAX_FILE_LINES)
        visit: How many times this file has been read this session (1=first)
    """
    if project_root is None:
        project_root = _resolve_project_root() or Path.cwd()

    # Resolve filepath
    fp = Path(filepath)
    if not fp.is_absolute():
        fp = project_root / fp
    if not fp.exists():
        return f"File not found: {filepath}"

    basename = fp.name
    ext = fp.suffix
    rel_path = str(fp.relative_to(project_root)) if fp.is_relative_to(project_root) else str(fp)

    parts = []

    # ── Step 1: Find relevant tree nodes ────────────────────────────
    relevant = _find_relevant_nodes(rel_path, store, session_context)

    if relevant:
        if visit <= 1:
            # First visit: full tree context header
            parts.append(f"## Tree context for {basename}\n")
            for item in relevant[:5]:
                n = item["node"]
                domain = n.meta.get("domain", "?")
                priority_tag = f" [priority={n.meta.get('priority')}]" if n.meta.get("priority") else ""
                parts.append(f"  [{domain}] {n.addr[:8]}{priority_tag}: {n.content[:200]}")
            parts.append("")
        else:
            # Repeat visit: only high-priority and issues nodes
            important = [
                item for item in relevant
                if item["node"].meta.get("priority", 0) >= 0.5
                or item["node"].meta.get("domain") == "issues"
            ]
            if important:
                parts.append(f"## Tree context for {basename} ({len(relevant)} nodes known — showing critical only)\n")
                for item in important[:3]:
                    n = item["node"]
                    domain = n.meta.get("domain", "?")
                    parts.append(f"  [{domain}] {n.addr[:8]}: {n.content[:200]}")
                parts.append("")
            # else: suppress header entirely on repeat visit with no critical nodes

    # ── Step 2: Read the file ───────────────────────────────────────
    try:
        all_lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Error reading {filepath}: {e}"

    total_lines = len(all_lines)

    # Apply offset and limit
    start = offset
    end = total_lines
    if limit > 0:
        end = min(start + limit, total_lines)
    else:
        end = min(start + _MAX_FILE_LINES, total_lines)

    visible_lines = all_lines[start:end]

    # ── Step 3: Extract line-specific annotations ───────────────────
    line_annotations = _extract_line_annotations(
        [item["node"] for item in relevant], basename
    )

    # ── Step 3b: Resolve content_hash anchors to precise line positions ──
    for item in relevant:
        anchor = item.get("_anchor")
        if not anchor:
            continue
        result = find_anchor_in_file(anchor, fp)
        if not result["found"]:
            continue
        line = result["line_num"]
        node = item["node"]
        domain = node.meta.get("domain", "?")
        drift_tag = " [DRIFTED — needs update]" if result["drifted"] else ""
        annotation = (
            f"  ## [{domain}] {node.addr[:8]}{drift_tag}: {node.content[:200]}"
        )
        if line not in line_annotations:
            line_annotations[line] = []
        # Only add if not already annotated by text-ref extraction
        if annotation not in line_annotations[line]:
            line_annotations[line].append(annotation)

    # ── Step 4: Detect and annotate sections ────────────────────────
    sections = _detect_sections(all_lines, ext)
    section_annotations = _annotate_sections(sections, relevant)

    # ── Step 5: Build annotated output ──────────────────────────────
    parts.append(f"## {rel_path} ({total_lines} lines)")
    if start > 0 or end < total_lines:
        parts.append(f"## Showing lines {start + 1}-{end} of {total_lines}")
    parts.append("")

    for i, line in enumerate(visible_lines):
        line_num = start + i + 1  # 1-based

        # Insert section annotation before the section header
        if line_num in section_annotations:
            parts.append(section_annotations[line_num])

        # Insert line-specific annotations before the line
        if line_num in line_annotations:
            for ann in line_annotations[line_num]:
                parts.append(ann)

        parts.append(f"  {line_num}: {line.rstrip()}")

    # Truncation notice
    if end < total_lines:
        parts.append(f"\n  ... ({total_lines - end} more lines)")

    parts.append("")

    # ── Step 6: Summary ─────────────────────────────────────────────
    annotation_count = len(line_annotations) + len(section_annotations)
    parts.append("## Read summary\n")
    parts.append(f"  File: {rel_path}")
    parts.append(f"  Lines shown: {end - start} of {total_lines}")
    parts.append(f"  Tree nodes found: {len(relevant)}")
    parts.append(f"  Inline annotations: {annotation_count}")
    parts.append(f"  Code sections: {len(sections)}")

    if not relevant:
        parts.append(f"\n  No tree coverage for {basename} — "
                     "consider claiming knowledge after reading.")

    result = "\n".join(parts)
    if len(result) > _OUTPUT_CAP:
        result = result[:_OUTPUT_CAP] + "\n\n... (output truncated)"

    return result

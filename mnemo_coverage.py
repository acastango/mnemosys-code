"""
mnemo_coverage.py — Anchor coverage report

Walks a file or directory and reports what percentage of code sections
have content-hash-anchored comprehension nodes in the tree.

Answers: "where has the tree seen this codebase, and what's still dark?"
Tells you exactly where to run memory_map next.

Zero API calls. Reads the file index + detects sections via mnemo_map.
"""

import os
from pathlib import Path
from typing import Optional

from mnemo import Store
from mnemo_anchor import get_anchors_for_file, load_file_index
from mnemo_map import _detect_sections, _walk_files, _SKIP_DIRS


_DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".rs", ".go", ".c", ".h", ".cpp", ".cs",
}

_BAR_WIDTH = 10


def _bar(covered: int, total: int) -> str:
    if total == 0:
        return "░" * _BAR_WIDTH
    filled = round(_BAR_WIDTH * covered / total)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def coverage(target: str, store: Store,
             project_root: Optional[Path] = None,
             extensions: Optional[set[str]] = None) -> dict:
    """Compute anchor coverage for a file or directory.

    Returns:
        {
          total_files, covered_files, unmapped_files,
          total_sections, covered_sections,
          file_coverage: [{path, total_sections, covered_sections, anchors}],
          unmapped: [path, ...]
        }
    """
    if project_root is None:
        from mnemo_verify import _resolve_project_root
        project_root = _resolve_project_root() or Path.cwd()

    if extensions is None:
        extensions = _DEFAULT_EXTENSIONS

    tp = Path(target)
    if not tp.is_absolute():
        tp = project_root / tp

    files = _walk_files(tp, extensions)
    index = load_file_index(store)

    file_coverage = []
    unmapped = []
    total_sections = 0
    total_covered_sections = 0

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        try:
            rel_path = str(fp.relative_to(project_root)).replace("\\", "/")
        except ValueError:
            rel_path = str(fp)

        lines = text.splitlines()
        ext = fp.suffix
        sections = _detect_sections(lines, ext)
        n_sections = len(sections)
        total_sections += n_sections

        # Check file index first (O(1)), fall back to anchor scan
        basename = fp.name
        entries = index.get(rel_path) or index.get(basename) or []

        if entries:
            # Count unique anchors pointing to active nodes in this file
            active = store.get_active()
            active_entries = [e for e in entries if e["addr"] in active]
            n_anchored = len(active_entries)
        else:
            # Index cold or file not covered
            anchors = get_anchors_for_file(rel_path, store)
            n_anchored = len(anchors)

        total_covered_sections += min(n_anchored, n_sections)

        if n_anchored == 0:
            unmapped.append(rel_path)
        else:
            file_coverage.append({
                "path": rel_path,
                "total_sections": n_sections,
                "covered_sections": min(n_anchored, n_sections),
                "anchors": n_anchored,
            })

    # Sort covered files by coverage ratio ascending (worst first — actionable)
    file_coverage.sort(
        key=lambda f: f["covered_sections"] / max(f["total_sections"], 1)
    )

    return {
        "total_files": len(files),
        "covered_files": len(file_coverage),
        "unmapped_files": len(unmapped),
        "total_sections": total_sections,
        "covered_sections": total_covered_sections,
        "file_coverage": file_coverage,
        "unmapped": unmapped,
    }


def format_report(result: dict, target: str) -> str:
    """Format a coverage result as a readable report."""
    total_f = result["total_files"]
    covered_f = result["covered_files"]
    unmapped_f = result["unmapped_files"]
    total_s = result["total_sections"]
    covered_s = result["covered_sections"]

    pct_files = round(100 * covered_f / total_f) if total_f else 0
    pct_sections = round(100 * covered_s / total_s) if total_s else 0

    lines = [
        f"Coverage report: {target} ({total_f} file{'s' if total_f != 1 else ''})",
        "",
    ]

    if result["file_coverage"]:
        lines.append(f"  Covered ({covered_f} file{'s' if covered_f != 1 else ''}, {pct_files}%):")
        for f in result["file_coverage"]:
            ts = f["total_sections"]
            cs = f["covered_sections"]
            bar = _bar(cs, ts)
            section_tag = f"{cs}/{ts} sections" if ts > 0 else "no sections detected"
            name = os.path.basename(f["path"])
            lines.append(f"    {name:<30} {section_tag:<18} {bar}")
        lines.append("")

    if result["unmapped"]:
        lines.append(f"  Unmapped ({unmapped_f} file{'s' if unmapped_f != 1 else ''}):")
        for path in result["unmapped"][:20]:
            lines.append(f"    {os.path.basename(path)}")
        if unmapped_f > 20:
            lines.append(f"    ... and {unmapped_f - 20} more")
        lines.append("")

    # Summary bar
    overall_bar = _bar(covered_s, total_s)
    lines.append(
        f"  Overall: {covered_s}/{total_s} sections covered "
        f"({pct_sections}%)  {overall_bar}"
    )

    if unmapped_f > 0:
        # Show the highest-value targets (most sections, unmapped)
        # Re-sort unmapped by section count descending — most valuable to map first
        unmapped_with_sections = [
            f for f in result.get("_unmapped_detail", [])
        ]
        lines.append(
            f"\n  Run memory_map(\"{target}\") to generate coverage for "
            f"{unmapped_f} unmapped file{'s' if unmapped_f != 1 else ''}."
        )

    return "\n".join(lines)

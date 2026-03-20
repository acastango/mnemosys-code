"""
mnemo_map.py — Cartographer: map existing codebase to content-hash anchors

Walks a file or directory, detects every significant code section
(functions, classes, structs), and runs the extraction sidecar against
each with recent session log context. Produces comprehension nodes bound
to content-hash anchors — the initial coverage pass for codebases with
no existing mappings.

The sidecar proposes what's worth storing. Trivial sections (getters,
boilerplate, obvious wrappers) are skipped. Proposals surface immediately
as stored nodes — no pending queue, since map is a deliberate conscious act.

Usage via MCP:
    memory_map("src/battle/")            # map a directory
    memory_map("src/battle/damage.py")   # map a single file
    memory_map("src/", scope="module")   # module-level only (faster)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from mnemo import Store, Node
from mnemo_anchor import compute_content_hash, update_file_index
from mnemo_log import emit


_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", "dist", "build", ".next",
}

# Lines of context captured starting at each section header
_CONTEXT_LINES = 6


# ───────────────────────────────────────────────────────────────────
# Section detection per language
# ───────────────────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, list[tuple]] = {
    ".py": [
        (re.compile(r"^class\s+\w+"), "class"),
        (re.compile(r"^(?:async\s+)?def\s+\w+"), "function"),
    ],
    ".js": [
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+"), "function"),
        (re.compile(r"^(?:export\s+)?class\s+\w+"), "class"),
        (re.compile(r"^(?:export\s+)?(?:const|let)\s+\w+\s*=\s*(?:async\s*)?\(.*\)\s*=>"), "function"),
    ],
    ".ts": [
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+"), "function"),
        (re.compile(r"^(?:export\s+)?class\s+\w+"), "class"),
        (re.compile(r"^(?:export\s+)?interface\s+\w+"), "struct"),
        (re.compile(r"^(?:export\s+)?(?:const|let)\s+\w+\s*=\s*(?:async\s*)?\(.*\)\s*=>"), "function"),
    ],
    ".tsx": [
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+"), "function"),
        (re.compile(r"^(?:export\s+)?class\s+\w+"), "class"),
        (re.compile(r"^(?:export\s+)?const\s+\w+\s*(?::\s*\w+\s*)?=\s*\("), "function"),
    ],
    ".rs": [
        (re.compile(r"^(?:pub\s+)?fn\s+\w+"), "function"),
        (re.compile(r"^(?:pub\s+)?struct\s+\w+"), "struct"),
        (re.compile(r"^(?:pub\s+)?impl\s+\w+"), "class"),
        (re.compile(r"^(?:pub\s+)?enum\s+\w+"), "struct"),
    ],
    ".go": [
        (re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+"), "function"),
        (re.compile(r"^type\s+\w+\s+struct"), "struct"),
        (re.compile(r"^type\s+\w+\s+interface"), "struct"),
    ],
    ".c": [
        (re.compile(r"^(?:static\s+)?(?:inline\s+)?\w[\w\s*]+\w\s*\("), "function"),
        (re.compile(r"^typedef\s+struct\s+\w*"), "struct"),
    ],
    ".h": [
        (re.compile(r"^(?:static\s+)?(?:inline\s+)?\w[\w\s*]+\w\s*\("), "function"),
        (re.compile(r"^typedef\s+struct\s+\w*"), "struct"),
    ],
}

_FALLBACK_PATTERNS = [
    (re.compile(r"^(?:function|def|func|fn)\s+\w+"), "function"),
    (re.compile(r"^(?:class|struct|interface)\s+\w+"), "class"),
]


def _detect_sections(lines: list[str], ext: str) -> list[dict]:
    """Find all significant sections in a file.

    Returns list of {line_num (1-based), scope, context_lines}.
    """
    patterns = _SECTION_PATTERNS.get(ext, _FALLBACK_PATTERNS)
    sections = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            continue
        for pattern, scope in patterns:
            if pattern.match(stripped):
                end = min(i + _CONTEXT_LINES, len(lines))
                ctx = "\n".join(ln.rstrip() for ln in lines[i:end])
                sections.append({
                    "line_num": i + 1,
                    "scope": scope,
                    "context_lines": ctx,
                })
                break

    return sections


def _walk_files(path: Path, extensions: set[str]) -> list[Path]:
    if path.is_file():
        if not extensions or path.suffix in extensions:
            return [path]
        return []

    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            fp = Path(dirpath) / fn
            if not extensions or fp.suffix in extensions:
                files.append(fp)
    return files


# ───────────────────────────────────────────────────────────────────
# Session log context
# ───────────────────────────────────────────────────────────────────

def _read_session_context(log_path: Optional[str], max_lines: int = 40) -> str:
    """Tail recent session log events for sidecar context."""
    if not log_path or not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        recent = lines[-max_lines:] if len(lines) > max_lines else lines
        events = []
        for line in recent:
            try:
                ev = json.loads(line.strip())
                if ev.get("event") in ("recall", "status", "extract_error",
                                       "extract", "maintain"):
                    continue
                summary = ev.get("summary", "")
                if summary:
                    events.append(f"  [{ev.get('event', '?')}] {summary[:100]}")
            except json.JSONDecodeError:
                continue
        return "\n".join(events)
    except Exception:
        return ""


# ───────────────────────────────────────────────────────────────────
# Haiku comprehension proposal
# ───────────────────────────────────────────────────────────────────

_MAP_SYSTEM = """You are a code comprehension agent. Analyze code sections and produce
concise, lasting knowledge nodes that future AI instances will use to understand
this codebase immediately, without re-reading the code.

For each section, capture:
- What it does (functional purpose, one sentence)
- Why it exists (design rationale, if visible from context or session history)
- Key invariants or constraints (what must always be true here)
- Non-obvious dependencies or side effects

Write for a future instance with zero context. Be specific — name the actual
function, class, or module. Don't be generic.

Return a JSON array:
[{
  "section_idx": <int>,
  "content": "Comprehension text — specific, actionable, self-contained.",
  "domain": "architecture|patterns|issues|decisions",
  "confidence": 0.0-1.0,
  "worth_storing": true|false
}]

Skip: trivial getters/setters, obvious wrappers, boilerplate with no design content.
Return [] if nothing is worth storing."""


def _propose_comprehension(sections: list[dict], rel_path: str,
                            session_context: str, client) -> list[dict]:
    """Ask Haiku to produce comprehension nodes for code sections in one file."""
    if not sections or client is None:
        return []

    model = os.environ.get("MNEMO_SMALL_MODEL", "claude-haiku-4-5-20251001")

    sections_text = "\n\n".join(
        f"Section {i} [{s['scope']}] (line {s['line_num']}):\n{s['context_lines']}"
        for i, s in enumerate(sections)
    )

    prompt_parts = [f"File: {rel_path}\n\nCode sections to analyze:\n{sections_text}"]
    if session_context:
        prompt_parts.append(f"Recent session context (use to infer WHY things exist):\n{session_context}")
    prompt_parts.append("Produce comprehension nodes for sections worth storing.")

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=_MAP_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        proposals = json.loads(match.group())
        return [p for p in proposals
                if isinstance(p, dict) and p.get("worth_storing", True)]
    except Exception as e:
        emit("map_error", "conscious",
             f"comprehension proposal failed for {rel_path}: {e}")
        return []


# ───────────────────────────────────────────────────────────────────
# Main map function
# ───────────────────────────────────────────────────────────────────

def map_path(target: str, store: Store,
             project_root: Optional[Path] = None,
             extensions: Optional[set[str]] = None,
             log_path: Optional[str] = None,
             client=None) -> dict:
    """Map a file or directory: detect sections, bind comprehension nodes.

    Args:
        target:       File or directory (absolute or relative to project_root)
        store:        The node store
        project_root: Project root for relative path resolution
        extensions:   File extensions to process (default: common source types)
        log_path:     Session log path for sidecar context
        client:       Anthropic client (lazy-init if None)

    Returns:
        {files_processed, sections_found, nodes_created, nodes_skipped, unmapped}
    """
    if project_root is None:
        from mnemo_verify import _resolve_project_root
        project_root = _resolve_project_root() or Path.cwd()

    if extensions is None:
        extensions = {".py", ".js", ".ts", ".tsx", ".jsx",
                      ".rs", ".go", ".c", ".h", ".cpp", ".cs"}

    if client is None:
        try:
            import anthropic
            client = anthropic.Anthropic()
        except Exception as e:
            return {
                "error": f"anthropic SDK unavailable: {e}",
                "files_processed": 0, "sections_found": 0,
                "nodes_created": 0, "nodes_skipped": 0, "unmapped": [],
            }

    tp = Path(target)
    if not tp.is_absolute():
        tp = project_root / tp

    files = _walk_files(tp, extensions)
    if not files:
        return {
            "files_processed": 0, "sections_found": 0,
            "nodes_created": 0, "nodes_skipped": 0,
            "unmapped": [str(tp)],
        }

    session_ctx = _read_session_context(log_path)
    active = store.get_active()

    total_sections = 0
    total_created = 0
    total_skipped = 0
    unmapped_files = []

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        ext = fp.suffix
        try:
            rel_path = str(fp.relative_to(project_root))
        except ValueError:
            rel_path = str(fp)

        sections = _detect_sections(lines, ext)
        if not sections:
            unmapped_files.append(rel_path)
            continue

        total_sections += len(sections)

        proposals = _propose_comprehension(sections, rel_path, session_ctx, client)

        if not proposals:
            unmapped_files.append(rel_path)
            total_skipped += len(sections)
            continue

        file_created = 0
        for proposal in proposals:
            idx = proposal.get("section_idx", 0)
            if idx >= len(sections):
                continue
            section = sections[idx]

            content_hash = compute_content_hash(section["context_lines"])
            anchor = {
                "type": "content_hash",
                "file": rel_path,
                "content_hash": content_hash,
                "context_lines": section["context_lines"],
                "scope": section["scope"],
                "line_hint": section["line_num"],
            }

            node = Node(
                type="leaf",
                content=proposal["content"],
                meta={
                    "domain": proposal.get("domain", "architecture"),
                    "confidence": proposal.get("confidence", 0.8),
                    "source": "map",
                    "anchors": [anchor],
                    "mapped_file": rel_path,
                    "mapped_line": section["line_num"],
                },
            )

            store.put(node)
            active.add(node.addr)
            update_file_index(store, node)
            file_created += 1

        total_created += file_created
        total_skipped += len(sections) - len(proposals)

        emit("map", "conscious",
             f"mapped {file_created}/{len(sections)} sections in {rel_path}",
             detail={"file": rel_path, "sections": len(sections),
                     "nodes": file_created})

    store.set_active(active)

    if unmapped_files:
        emit("map_unmapped", "conscious",
             f"{len(unmapped_files)} files with no mappable sections",
             detail={"files": unmapped_files[:20]})

    emit("map_complete", "conscious",
         f"mapped {total_created} nodes across {len(files)} files",
         detail={"files": len(files), "sections": total_sections,
                 "created": total_created, "skipped": total_skipped})

    return {
        "files_processed": len(files),
        "sections_found": total_sections,
        "nodes_created": total_created,
        "nodes_skipped": total_skipped,
        "unmapped": unmapped_files,
    }

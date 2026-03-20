"""
mnemo_hook.py — Proactive comprehension hooks for Claude Code

Handles four hook events:

  PreToolUse / Read
    Injects content-hash-anchored comprehension as additionalContext before
    Claude reads a file. First visit = full module context. Subsequent visits
    = warnings and drifted anchors only (adaptive depth).

  PreToolUse / Edit | Write
    Injects tree knowledge about the file being edited (existing behavior).

  PostToolUse / Edit | Write
    Detects anchor drift after an edit. Marks drifted anchors in node meta.
    Queues a comprehension binding via the sidecar (background, no API wait).

  PostToolUse / Write (new file)
    Emits an unmapped log event so the next memory_status shows coverage gaps.

Zero blocking API calls on any PreToolUse path. PostToolUse drift detection
is also zero-API (pure hash comparison). Binding runs in background.

Hook config (add to ~/.claude/settings.json):
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Read|Edit|Write",
      "hooks": [{"type": "command", "command": "python PATH/TO/mnemo_hook.py"}]
    }],
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{"type": "command", "command": "python PATH/TO/mnemo_hook.py"}]
    }]
  }
}
"""

import json
import os
import re
import sys


# ───────────────────────────────────────────────────────────────────
# Visit tracking — per-file read counter for adaptive injection depth
# ───────────────────────────────────────────────────────────────────

def _visits_path(store_path: str) -> str:
    return os.path.join(store_path, "file_visits.json")


def _load_visits(store_path: str) -> dict:
    path = _visits_path(store_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_visits(store_path: str, visits: dict) -> None:
    path = _visits_path(store_path)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(visits, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _increment_visit(store_path: str, filepath: str) -> int:
    """Increment and return the visit count for a file this session."""
    visits = _load_visits(store_path)
    key = os.path.basename(filepath)
    visits[key] = visits.get(key, 0) + 1
    _save_visits(store_path, visits)
    return visits[key]


def _get_visit_count(store_path: str, filepath: str) -> int:
    visits = _load_visits(store_path)
    return visits.get(os.path.basename(filepath), 0)


# ───────────────────────────────────────────────────────────────────
# Store loading
# ───────────────────────────────────────────────────────────────────

def _load_store(store_path: str):
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from mnemo import Store
    return Store(store_path)


# ───────────────────────────────────────────────────────────────────
# PreToolUse: Read — inject comprehension
# ───────────────────────────────────────────────────────────────────

def _handle_pre_read(event: dict, store_path: str) -> None:
    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or tool_input.get("path", "") or tool_input.get("file", "")
    if not file_path:
        return

    store = _load_store(store_path)
    visit = _increment_visit(store_path, file_path)

    from mnemo_anchor import get_anchors_for_file, find_anchor_in_file
    from mnemo_verify import _resolve_project_root
    from pathlib import Path

    project_root = _resolve_project_root() or Path.cwd()
    fp = Path(file_path)
    if not fp.is_absolute():
        fp = project_root / fp

    anchored = get_anchors_for_file(file_path, store)
    if not anchored:
        return

    lines = [f"[mnemo] Comprehension for {os.path.basename(file_path)} "
             f"(visit {visit}):"]

    # First visit: full comprehension. Subsequent: warnings/drifted only.
    shown = 0
    for item in anchored:
        node = item["node"]
        anchor = item["anchor"]
        domain = node.meta.get("domain", "?")

        if visit == 1:
            # Full: all anchored nodes, resolved to line positions
            result = find_anchor_in_file(anchor, fp)
            line_tag = f" line {result['line_num']}" if result.get("found") else ""
            drift_tag = " [DRIFTED]" if result.get("drifted") else ""
            content = node.content[:180].replace("\n", " ")
            lines.append(f"  [{domain}]{line_tag}{drift_tag}: {content} [{node.addr[:8]}]")
            shown += 1
        else:
            # Repeat visit: only high-priority, issues, or drifted
            priority = node.meta.get("priority", 0)
            is_issue = domain == "issues"
            result = find_anchor_in_file(anchor, fp)
            is_drifted = result.get("drifted", False)

            if priority >= 0.5 or is_issue or is_drifted:
                drift_tag = " [DRIFTED — update needed]" if is_drifted else ""
                content = node.content[:180].replace("\n", " ")
                lines.append(f"  [{domain}]{drift_tag}: {content} [{node.addr[:8]}]")
                shown += 1

        if shown >= 8:
            break

    if shown == 0:
        return

    _output_context("\n".join(lines))


# ───────────────────────────────────────────────────────────────────
# PreToolUse: Edit | Write — inject tree knowledge
# ───────────────────────────────────────────────────────────────────

# Terms so generic they match everything in an mnemo-heavy project
_NOISE_TERMS = frozenset({
    "mnemo", "memory", "store", "node", "tree", "tool", "file", "path",
    "chain", "agent", "project", "content", "return", "false", "true",
    "none", "self", "data", "type", "list", "dict", "import", "from",
    "with", "that", "this", "have", "will", "been", "when", "what",
    "where", "which", "there", "their", "about", "after", "before",
})


def _extract_terms(text: str, min_len: int = 5, max_terms: int = 12) -> list[str]:
    """
    Extract meaningful search terms from content being written.
    Prefers longer, more specific tokens. Filters noise terms.
    """
    # Tokenise: split on non-alphanumeric (handles snake_case, camelCase, prose)
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text)
    # Split camelCase: "touchPresence" → ["touch", "Presence"]
    expanded = []
    for tok in tokens:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', tok).split()
        expanded.extend(parts)

    seen: set[str] = set()
    terms: list[str] = []
    # Score by length — longer = more specific
    for tok in sorted(expanded, key=len, reverse=True):
        lower = tok.lower()
        if lower in _NOISE_TERMS or lower in seen or len(lower) < min_len:
            continue
        seen.add(lower)
        terms.append(lower)
        if len(terms) >= max_terms:
            break
    return terms


def _filename_terms(file_path: str) -> list[str]:
    """Fallback terms from filename — strips noise prefixes."""
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    raw_parts = re.split(r'[_\-.]', name_no_ext.lower())
    return [p for p in raw_parts if len(p) >= 5 and p not in _NOISE_TERMS]


def _search_store(store, terms: list[str], min_score: int = 1) -> list:
    """Match active nodes against terms. Returns matched nodes with score >= min_score."""
    active = store.get_active()
    if not active:
        return []
    patterns = [re.compile(r'\b' + re.escape(t) + r'\b') for t in terms]
    matches = []
    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        content_lower = node.content.lower()
        score = sum(1 for p in patterns if p.search(content_lower))
        if score >= min_score:
            matches.append((score, node))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [node for _, node in matches]


def _handle_pre_edit(event: dict, store_path: str) -> None:
    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
    if not file_path:
        return

    basename = os.path.basename(file_path)

    # Primary signal: content being written (new_string for Edit, content for Write)
    content_signal = tool_input.get("new_string", "") or tool_input.get("content", "")
    if len(content_signal) >= 60:
        terms = _extract_terms(content_signal)
    else:
        terms = _filename_terms(file_path)

    if not terms:
        return

    store = _load_store(store_path)
    matches = _search_store(store, terms)

    # Global store — only if project store didn't find enough, and require
    # higher relevance threshold (min_score=2) to suppress cross-project noise
    if len(matches) < 2:
        global_path = os.environ.get("MNEMO_GLOBAL",
                                      os.path.expanduser("~/.mnemo/global"))
        try:
            from mnemo import Store
            global_store = Store(global_path)
            matches.extend(_search_store(global_store, terms, min_score=2))
        except Exception:
            pass

    if not matches:
        return

    lines = [f"[mnemo] Relevant context for {basename}:"]
    for node in matches[:3]:
        domain = node.meta.get("domain", "?")
        content = node.content[:150].replace("\n", " ")
        lines.append(f"  [{domain}] {content} [{node.addr[:8]}]")

    _output_context("\n".join(lines))


# ───────────────────────────────────────────────────────────────────
# PostToolUse: Edit | Write — drift detection + binding trigger
# ───────────────────────────────────────────────────────────────────

def _handle_post_edit(event: dict, store_path: str) -> None:
    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
    if not file_path:
        return

    store = _load_store(store_path)

    from mnemo_anchor import detect_drift, mark_drifted
    from mnemo_log import emit

    drifted = detect_drift(file_path, store)
    for item in drifted:
        result = item["result"]
        current_hash = result.get("current_hash")
        mark_drifted(item["node"], item["anchor_idx"], current_hash, store)
        status = "missing" if not result["found"] else "drifted"
        emit("anchor_drift", "subconscious",
             f"{status}: {item['node'].addr[:8]} in {os.path.basename(file_path)}",
             addresses=[item["node"].addr],
             detail={"file": file_path, "status": status,
                     "current_hash": current_hash})

    # Trigger comprehension binding for what was just written
    tool_response = event.get("tool_response", {})
    # Read the file content that was written
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code_content = f.read()
    except Exception:
        return

    if not code_content.strip():
        return

    from mnemo_log import _get_log_path
    log_path = _get_log_path()



# ───────────────────────────────────────────────────────────────────
# PostToolUse: Write (new file) — emit unmapped flag
# ───────────────────────────────────────────────────────────────────

def _handle_post_write_new(file_path: str, store_path: str) -> None:
    """Emit unmapped event for newly created files."""
    from mnemo_log import emit
    emit("unmapped", "subconscious",
         f"new file has no comprehension coverage: {os.path.basename(file_path)}",
         detail={"file": file_path,
                 "suggestion": "Run memory_map to generate coverage."})


# ───────────────────────────────────────────────────────────────────
# Output helpers
# ───────────────────────────────────────────────────────────────────

def _output_context(context: str) -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def main():
    try:
        event = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return

    hook_event = event.get("hook_event_name", "PreToolUse")
    tool_name = event.get("tool_name", "")

    # Discover the nearest .mnemo/ project store: walk up from the file being touched,
    # then from CWD, before falling back to MNEMO_STORE / legacy ~/mnemo.
    def _discover_store(start_dir: str) -> str | None:
        current = os.path.abspath(start_dir)
        for _ in range(20):
            candidate = os.path.join(current, ".mnemo")
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None

    _ti = event.get("tool_input", {})
    _file = _ti.get("file_path", "") or _ti.get("path", "")
    store_path = None
    if _file:
        store_path = _discover_store(os.path.dirname(os.path.abspath(_file)))
    if not store_path:
        store_path = _discover_store(os.getcwd())
    if not store_path:
        store_path = os.environ.get("MNEMO_STORE", os.path.expanduser("~/mnemo"))

    try:
        code_dir = os.path.dirname(os.path.abspath(__file__))
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
    except Exception:
        return

    try:
        if hook_event == "PreToolUse":
            if tool_name in ("Read", "memory_read"):
                _handle_pre_read(event, store_path)
            elif tool_name in ("Edit", "Write", "memory_edit", "memory_write"):
                _handle_pre_edit(event, store_path)

        elif hook_event == "PostToolUse":
            if tool_name in ("Edit", "Write", "memory_edit", "memory_write"):
                # Check if this was a new file creation
                _ti = event.get("tool_input", {})
                file_path = _ti.get("file_path", "") or _ti.get("path", "")
                tool_response = event.get("tool_response", {})
                is_new = (tool_name in ("Write", "memory_write") and
                          isinstance(tool_response, dict) and
                          tool_response.get("created", False))
                if is_new:
                    _handle_post_write_new(file_path, store_path)
                _handle_post_edit(event, store_path)

    except Exception:
        pass  # hooks must never surface errors to the user


if __name__ == "__main__":
    main()

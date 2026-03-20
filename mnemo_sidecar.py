#!/usr/bin/env python3
"""
mnemo_sidecar.py — Live memory process monitor

Tails the current session's event log and renders a live terminal UI.
Each mnemo session writes to its own timestamped log file under
MNEMO_STORE/logs/. The sidecar auto-resolves the active session via
MNEMO_STORE/logs/current.txt, so just run it and it follows the live session.

Usage:
    python mnemo_sidecar.py                          # auto-finds current session
    python mnemo_sidecar.py --tail 50
    python mnemo_sidecar.py --log ~/mnemo/logs/20260310_061200_abc123.log  # specific session

Layers shown:
    [sub]  subconscious — ambient recall and auto-extraction (cyan)
    [con]  conscious    — deliberate tool calls (green)
    [sys]  system       — housekeeping, status, compression (yellow)

Event markers:
    [~] recall      [+] claim       [>] update
    [v] reinforce   [*] compress    [?] search
    [<] provenance  [=] status      [S] soul
    [R] reroot      [P] prune       [D] dream
"""

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("rich is not installed.")
    print("Run: pip install rich   or   uv pip install rich")
    raise SystemExit(1)

# ─── Defaults ─────────────────────────────────────────────────────────────────

STORE_PATH = os.environ.get("MNEMO_STORE", os.path.expanduser("~/mnemo"))
LOGS_DIR    = os.path.join(STORE_PATH, "logs")
CURRENT_PTR = os.path.join(LOGS_DIR, "current.txt")
POLL_INTERVAL  = 0.25  # seconds between log checks
STATS_INTERVAL = 20    # poll ticks between store stat refreshes (~5 seconds)


def resolve_log_path(given: Optional[str]) -> Optional[str]:
    """
    Resolve which log file to watch.
    - If --log was passed explicitly, use that.
    - Otherwise read LOGS_DIR/current.txt for the active session.
    - Returns None if nothing is found yet (sidecar will wait).
    """
    if given:
        return given
    if Path(CURRENT_PTR).exists():
        candidate = Path(CURRENT_PTR).read_text(encoding="utf-8").strip()
        if candidate and Path(candidate).exists():
            return candidate
    return None

# ─── Styling ──────────────────────────────────────────────────────────────────

LAYER_STYLE = {
    "subconscious": "cyan",
    "conscious":    "bright_green",
    "system":       "yellow",
}

LAYER_LABEL = {
    "subconscious": "sub",
    "conscious":    "con",
    "system":       "sys",
}

EVENT_STYLE = {
    "recall":     "blue",
    "claim":      "green",
    "update":     "yellow",
    "reinforce":  "dim green",
    "compress":   "magenta",
    "search":     "cyan",
    "query":      "cyan",
    "provenance": "dim cyan",
    "status":     "dim yellow",
    "soul":       "bright_white",
    "reroot":     "bright_magenta",
    "prune":      "dim yellow",
    "active":     "dim white",
    "dream":      "bright_blue",
}

EVENT_MARKER = {
    "recall":     "~",
    "claim":      "+",
    "update":     ">",
    "reinforce":  "v",
    "compress":   "*",
    "search":     "?",
    "query":      ".",
    "provenance": "<",
    "status":     "=",
    "soul":       "S",
    "reroot":     "R",
    "prune":      "P",
    "active":     "A",
    "dream":      "D",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_event(line: str) -> Optional[dict]:
    try:
        return json.loads(line.strip())
    except Exception:
        return None


def fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return iso[:8] if iso else "??"


def get_store_stats() -> dict:
    """Read current node count directly from the store."""
    try:
        active_path = os.path.join(STORE_PATH, "active.json")
        with open(active_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data.get("nodes", data))
        else:
            count = "?"
    except Exception:
        count = "?"
    return {"count": count}


def event_counts(events: deque) -> dict:
    """Tally events by type for the stats bar."""
    counts: dict = {}
    for ev in events:
        k = ev.get("event", "?")
        counts[k] = counts.get(k, 0) + 1
    return counts


# ─── UI components ────────────────────────────────────────────────────────────

def build_header(stats: dict, total_logged: int, session_name: str) -> Panel:
    t = Text()
    t.append("mnemo", style="bold bright_white")
    t.append(" project monitor", style="dim white")
    t.append("   |   ", style="dim")
    t.append(str(stats["count"]), style="bold cyan")
    t.append(" nodes", style="dim cyan")
    t.append("   |   ", style="dim")
    t.append(session_name, style="dim white")
    t.append("   |   ", style="dim")
    t.append(f"{total_logged} events", style="dim white")
    return Panel(t, style="dim", padding=(0, 1))


def build_legend() -> Text:
    t = Text()
    # Layers
    for layer, style in LAYER_STYLE.items():
        label = LAYER_LABEL[layer]
        t.append(f" [{label}]", style=f"bold {style}")
        t.append(f" {layer}", style=f"dim {style}")
    t.append("     ")
    # Event markers (first 8)
    shown = ["recall", "claim", "update", "reinforce", "compress", "search", "dream", "reroot"]
    for ev in shown:
        marker = EVENT_MARKER.get(ev, "?")
        t.append(f"[{marker}]", style=EVENT_STYLE.get(ev, "white"))
        t.append(f"{ev} ", style="dim white")
    return t


def build_event_table(events: deque) -> Table:
    tbl = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim white",
        expand=True,
        padding=(0, 1),
        show_edge=False,
    )
    tbl.add_column("time",   style="dim white",  width=8,  no_wrap=True)
    tbl.add_column("lay",    width=5,             no_wrap=True)
    tbl.add_column("event",  width=10,            no_wrap=True)
    tbl.add_column("domain", width=9,             no_wrap=True)
    tbl.add_column("summary", ratio=1, no_wrap=False, overflow="fold")

    for ev in events:
        layer  = ev.get("layer", "system")
        event  = ev.get("event", "?")
        domain = ev.get("domain", "")
        summary = ev.get("summary", "")
        marker = EVENT_MARKER.get(event, "?")

        layer_style = LAYER_STYLE.get(layer, "white")
        event_style = EVENT_STYLE.get(event, "white")

        tbl.add_row(
            fmt_ts(ev.get("ts", "")),
            Text(f"[{LAYER_LABEL.get(layer, layer[:3])}]", style=f"bold {layer_style}"),
            Text(f"[{marker}]{event}", style=event_style),
            Text(domain, style="dim white"),
            Text(summary, style="dim" if layer == "system" else ""),
        )

    return tbl


def build_ui(events: deque, stats: dict, total_logged: int, session_name: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="legend", size=1),
        Layout(name="body"),
    )
    layout["header"].update(build_header(stats, total_logged, session_name))
    layout["legend"].update(build_legend())
    layout["body"].update(
        Panel(
            build_event_table(events),
            title="[dim]project memory log — Ctrl+C to exit[/dim]",
            style="dim",
            padding=0,
        )
    )
    return layout


# ─── Main loop ────────────────────────────────────────────────────────────────

def run(explicit_log: Optional[str], tail: int) -> None:
    console = Console()
    events: deque = deque(maxlen=tail)
    total_logged = 0

    # Wait for a session to appear if none exists yet
    log_path = resolve_log_path(explicit_log)
    if log_path is None:
        console.print("[dim]Waiting for a mnemo session to start...[/dim]")
        while log_path is None:
            time.sleep(0.5)
            log_path = resolve_log_path(explicit_log)

    session_name = Path(log_path).name

    # Seed with events already in the file
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            ev = parse_event(line)
            if ev:
                events.append(ev)
                total_logged += 1
    file_pos = Path(log_path).stat().st_size

    stats = get_store_stats()

    with Live(
        build_ui(events, stats, total_logged, session_name),
        console=console,
        auto_refresh=False,  # only redraw when we explicitly call refresh()
        screen=True,
    ) as live:
        tick = 0
        while True:
            try:
                changed = False

                # Poll for new log lines
                current_size = Path(log_path).stat().st_size
                if current_size > file_pos:
                    with open(log_path, encoding="utf-8") as f:
                        f.seek(file_pos)
                        for line in f:
                            ev = parse_event(line)
                            if ev:
                                events.append(ev)
                                total_logged += 1
                        file_pos = f.tell()
                    changed = True

                # Refresh store stats periodically
                if tick % STATS_INTERVAL == 0:
                    new_stats = get_store_stats()
                    if new_stats != stats:
                        stats = new_stats
                        changed = True

                if changed:
                    live.update(build_ui(events, stats, total_logged, session_name))
                    live.refresh()

                tick += 1
                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                break


def main() -> None:
    p = argparse.ArgumentParser(
        description="mnemo sidecar — live project memory monitor"
    )
    p.add_argument(
        "--log", default=None,
        help="Path to a specific session log (default: auto-resolve from current.txt)"
    )
    p.add_argument(
        "--tail", type=int, default=100,
        help="Max events to display at once (default: 100)"
    )
    args = p.parse_args()

    # Brief pre-launch message before entering full-screen mode
    source = args.log or CURRENT_PTR
    print(f"mnemo sidecar  |  session source: {source}")
    print(f"tail={args.tail} events  |  press Ctrl+C to exit\n")
    time.sleep(0.4)

    run(args.log, args.tail)
    print("\nsidecar closed.")


if __name__ == "__main__":
    main()

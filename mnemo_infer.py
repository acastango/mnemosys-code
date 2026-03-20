"""
mnemo_infer.py — Passive pattern inference from session logs

Learns implicit project knowledge by analyzing behavioral patterns
across sessions. Three inference layers:

1. Co-occurrence: files edited/read together → relates_to links
2. Recall patterns: which nodes are noise vs signal
3. Workflow patterns: domain clusters, session rhythms

Zero LLM calls. Pure log analysis + statistics.

Surfaced via memory_infer MCP tool or integrated into dream mode.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from mnemo import Store, Node


# -------------------------------------------------------------------
# Log parsing
# -------------------------------------------------------------------

def _iter_log_events(logs_dir: str) -> list[dict]:
    """Read all session logs and yield events in chronological order."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []

    events = []
    for log_file in sorted(logs_path.glob("*.log")):
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event["_session"] = log_file.stem
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue

    return events


def _group_by_session(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by session ID (log file stem)."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        sessions[event["_session"]].append(event)
    return dict(sessions)


# -------------------------------------------------------------------
# Layer 1: File co-occurrence
# -------------------------------------------------------------------

# Patterns to extract file references from event summaries/details
_FILE_REF = re.compile(
    r'([\w./\\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|toml|yaml|yml|json|cfg|md))'
)


def _extract_files_from_event(event: dict) -> set[str]:
    """Extract file references from an event's summary and detail.

    Skips recall/status events — they passively mention files from tree
    nodes and would make everything co-occur with everything.
    """
    evt = event.get("event", "")
    if evt in ("recall", "status", "extract", "micro_dream"):
        return set()

    files = set()
    summary = event.get("summary", "")
    for match in _FILE_REF.finditer(summary):
        files.add(match.group(1))

    detail = event.get("detail", {})
    if isinstance(detail, dict):
        detail_str = json.dumps(detail)
        for match in _FILE_REF.finditer(detail_str):
            files.add(match.group(1))

    return files


def analyze_cooccurrence(logs_dir: str,
                         min_sessions: int = 2) -> list[dict]:
    """Find files that are consistently edited/discussed together.

    Scans all session logs, counts file pairs that appear in the same
    session, and returns pairs above the threshold.

    Returns list of {file_a, file_b, sessions, strength} sorted by
    strength descending.
    """
    events = _iter_log_events(logs_dir)
    sessions = _group_by_session(events)

    # Per-session file sets
    session_files: dict[str, set[str]] = {}
    for session_id, session_events in sessions.items():
        files = set()
        for event in session_events:
            files |= _extract_files_from_event(event)
        if len(files) >= 2:
            session_files[session_id] = files

    # Count co-occurrences
    pair_counts: Counter = Counter()
    file_counts: Counter = Counter()

    for session_id, files in session_files.items():
        sorted_files = sorted(files)
        for f in sorted_files:
            file_counts[f] += 1
        for i, a in enumerate(sorted_files):
            for b in sorted_files[i + 1:]:
                pair_counts[(a, b)] += 1

    # Filter and score
    results = []
    total_sessions = len(session_files)
    if total_sessions == 0:
        return results

    for (a, b), count in pair_counts.most_common():
        if count < min_sessions:
            break
        # Jaccard: how often they appear together vs separately
        union = file_counts[a] + file_counts[b] - count
        strength = count / union if union > 0 else 0
        results.append({
            "file_a": a,
            "file_b": b,
            "sessions": count,
            "strength": round(strength, 3),
        })

    return results


# -------------------------------------------------------------------
# Layer 2: Recall pattern analysis
# -------------------------------------------------------------------

def analyze_recall_patterns(logs_dir: str,
                            store: Store) -> dict:
    """Analyze which nodes are signal vs noise in recall.

    Returns:
        {
            "noise_candidates": [...],  # frequently recalled, never acted on
            "high_value": [...],        # frequently recalled AND acted on
            "co_recalled": [...],       # nodes always recalled together
            "never_recalled": [...],    # active nodes never surfaced
        }
    """
    events = _iter_log_events(logs_dir)
    sessions = _group_by_session(events)

    # Track per-session: which addresses were recalled, which were acted on
    recall_counts: Counter = Counter()
    action_counts: Counter = Counter()
    co_recall: Counter = Counter()  # pairs recalled in same session

    action_events = {"claim", "update", "reinforce", "compress", "link"}

    for session_events in sessions.values():
        session_recalled: set[str] = set()
        session_acted: set[str] = set()

        for event in session_events:
            addrs = set(event.get("addresses", []))
            evt = event.get("event", "")

            if evt == "recall":
                session_recalled |= addrs
            elif evt in action_events:
                session_acted |= addrs

        for addr in session_recalled:
            recall_counts[addr] += 1
        for addr in session_acted:
            action_counts[addr] += 1

        # Co-recall: addresses recalled in the same session (any turn)
        sorted_recalled = sorted(session_recalled)
        for i, a in enumerate(sorted_recalled):
            for b in sorted_recalled[i + 1:]:
                co_recall[(a, b)] += 1

    # Classify
    active = store.get_active()
    active_set = set(active)

    noise_candidates = []
    high_value = []

    for addr, count in recall_counts.most_common():
        if count < 3:
            continue
        acted = action_counts.get(addr, 0)
        hit_rate = acted / count

        node = store.get(addr)
        if not node:
            continue
        snippet = node.content[:80]

        if hit_rate < 0.1:
            noise_candidates.append({
                "addr": addr,
                "recalls": count,
                "actions": acted,
                "hit_rate": round(hit_rate, 2),
                "snippet": snippet,
            })
        elif hit_rate > 0.5:
            high_value.append({
                "addr": addr,
                "recalls": count,
                "actions": acted,
                "hit_rate": round(hit_rate, 2),
                "snippet": snippet,
            })

    # Co-recalled pairs (strong association)
    co_recalled = []
    for (a, b), count in co_recall.most_common(20):
        if count < 3:
            break
        # Jaccard-like: co-occurrences / total individual occurrences
        individual_max = max(recall_counts.get(a, 0), recall_counts.get(b, 0))
        if individual_max > 0:
            affinity = count / individual_max
        else:
            affinity = 0
        if affinity > 0.6:
            co_recalled.append({
                "addr_a": a,
                "addr_b": b,
                "co_recalls": count,
                "affinity": round(affinity, 2),
            })

    # Never recalled — active nodes that never surfaced
    never_recalled = []
    recalled_ever = set(recall_counts.keys())
    for addr in active:
        short = addr[:12]
        if short not in recalled_ever and addr not in recalled_ever:
            node = store.get(addr)
            if node:
                never_recalled.append({
                    "addr": addr[:12],
                    "domain": node.meta.get("domain", "?"),
                    "snippet": node.content[:80],
                })

    return {
        "noise_candidates": noise_candidates[:10],
        "high_value": high_value[:10],
        "co_recalled": co_recalled[:10],
        "never_recalled": never_recalled[:10],
    }


# -------------------------------------------------------------------
# Layer 3: Behavioral inference (correction tracking)
# -------------------------------------------------------------------

# Classify update reasons into correction types
_CORRECTION_PATTERNS = {
    "user_correction": re.compile(
        r'(?:clarified|corrected|actually|no[,.]|not that|instead)', re.I),
    "staleness": re.compile(
        r'(?:stale|outdated|drifted|no longer|was listing|legacy)', re.I),
    "recategorization": re.compile(
        r'(?:recategor|wrong domain|fixing.*domain|was.*now)', re.I),
    "refinement": re.compile(
        r'(?:unifying|expanding|resolving|converted|replacing|replaced|'
        r'removed reference|confirmed working|removing hedging)', re.I),
    "decision": re.compile(
        r'(?:decision|decided|removed.*from|replaces|now uses|'
        r'now gathers|now active)', re.I),
    "evolution": re.compile(
        r'(?:added|implemented|expanded|updated to|evolved|grew)', re.I),
    "factual_fix": re.compile(
        r'(?:correcting|fixing|was wrong|actually \d|missing)', re.I),
}


def _classify_correction(reason: str) -> str:
    """Classify an update reason into a correction type."""
    for ctype, pattern in _CORRECTION_PATTERNS.items():
        if pattern.search(reason):
            return ctype
    return "unclassified"


def analyze_corrections(logs_dir: str, store: Store) -> dict:
    """Analyze correction patterns to understand knowledge stability.

    Tracks:
    - Which domains get corrected most (unstable knowledge areas)
    - What types of corrections happen (user vs self vs evolution)
    - Rapid claim→update cycles (wrong from the start)
    - Supersession chains (repeatedly corrected nodes)

    Returns:
        {
            "correction_types": {...},       # type -> count
            "domain_stability": [...],       # domains ranked by correction rate
            "rapid_corrections": [...],      # claims updated within same session
            "supersession_chains": [...],    # nodes updated multiple times
            "user_corrections": [...],       # corrections triggered by user feedback
        }
    """
    events = _iter_log_events(logs_dir)
    sessions = _group_by_session(events)

    correction_types: Counter = Counter()
    domain_claims: Counter = Counter()
    domain_updates: Counter = Counter()
    rapid_corrections = []
    addr_update_count: Counter = Counter()  # how many times each addr was superseded
    user_corrections = []

    for session_id, session_events in sessions.items():
        # Track claims made in this session for rapid correction detection
        session_claims: dict[str, dict] = {}  # addr -> event

        for event in session_events:
            evt = event.get("event", "")
            addrs = event.get("addresses", [])
            domain = event.get("domain", event.get("detail", {}).get("domain", ""))

            if evt == "claim":
                if domain:
                    domain_claims[domain] += 1
                for addr in addrs:
                    session_claims[addr[:12]] = event

            elif evt == "update":
                reason = event.get("detail", {}).get("reason", "")
                ctype = _classify_correction(reason)
                correction_types[ctype] += 1

                if domain:
                    domain_updates[domain] += 1

                # Check for rapid correction (claim and update in same session)
                if len(addrs) >= 2:
                    old_addr = addrs[0][:12]
                    addr_update_count[old_addr] += 1

                    if old_addr in session_claims:
                        rapid_corrections.append({
                            "old_addr": old_addr,
                            "reason": reason[:120],
                            "type": ctype,
                            "session": session_id,
                        })

                # Track user corrections specifically
                if ctype == "user_correction":
                    user_corrections.append({
                        "addr": addrs[0][:12] if addrs else "?",
                        "reason": reason[:150],
                        "domain": domain,
                        "session": session_id,
                    })

    # Domain stability: correction rate per domain
    domain_stability = []
    for domain in set(list(domain_claims.keys()) + list(domain_updates.keys())):
        claims = domain_claims.get(domain, 0)
        updates = domain_updates.get(domain, 0)
        total = claims + updates
        if total > 0:
            correction_rate = updates / total
            domain_stability.append({
                "domain": domain,
                "claims": claims,
                "updates": updates,
                "correction_rate": round(correction_rate, 2),
            })
    domain_stability.sort(key=lambda x: -x["correction_rate"])

    # Supersession chains — nodes updated multiple times
    chains = []
    for addr, count in addr_update_count.most_common(10):
        if count < 2:
            break
        node = store.get(addr)
        snippet = node.content[:80] if node else "(deactivated)"
        chains.append({
            "addr": addr,
            "updates": count,
            "snippet": snippet,
        })

    return {
        "correction_types": dict(correction_types.most_common()),
        "domain_stability": domain_stability,
        "rapid_corrections": rapid_corrections[:10],
        "supersession_chains": chains,
        "user_corrections": user_corrections[:10],
    }


# -------------------------------------------------------------------
# Layer 4: Implicit knowledge (sequence pattern mining)
# -------------------------------------------------------------------

def analyze_sequences(logs_dir: str, store: Store) -> dict:
    """Mine event sequences for implicit knowledge patterns.

    Detects:
    - Volatile nodes: recalled then updated in same session
    - Discovery sequences: recall/search → claim (knowledge gaps filled)
    - Core knowledge: addresses appearing in >70% of recall events
    - Domain cascades: claim in domain A reliably followed by claim in domain B
    - Recall triggers: message topics that always surface the same nodes

    Returns:
        {
            "volatile_nodes": [...],
            "discovery_sequences": [...],
            "core_knowledge": [...],
            "domain_cascades": [...],
            "recall_triggers": [...],
        }
    """
    events = _iter_log_events(logs_dir)
    sessions = _group_by_session(events)

    # ── Volatile nodes: recalled then updated in same session ──────
    # Track by concept (follow supersession chains) not just address
    volatile_instances: list[dict] = []

    for session_id, session_events in sessions.items():
        recalled_addrs: set[str] = set()
        for event in session_events:
            evt = event.get("event", "")
            addrs = event.get("addresses", [])

            if evt == "recall":
                recalled_addrs |= set(addrs)
            elif evt == "update" and len(addrs) >= 2:
                old_addr = addrs[0]
                new_addr = addrs[1]
                if old_addr in recalled_addrs:
                    node = store.get(new_addr) or store.get(old_addr)
                    snippet = node.content[:80] if node else "?"
                    domain = node.meta.get("domain", "?") if node else "?"
                    volatile_instances.append({
                        "old_addr": old_addr[:12],
                        "new_addr": new_addr[:12],
                        "session": session_id,
                        "domain": domain,
                        "snippet": snippet,
                    })

    # Group by domain+snippet similarity to find repeated volatility
    volatile_by_domain: Counter = Counter()
    for v in volatile_instances:
        volatile_by_domain[v["domain"]] += 1

    volatile_nodes = volatile_instances[:10]

    # ── Discovery sequences: recall/search/explore → claim ─────────
    discovery_events = {"recall", "search", "explore", "grep"}
    discovery_sequences = []

    for session_id, session_events in sessions.items():
        # Track topics mentioned in discovery events
        discovery_topics: set[str] = set()
        discovery_files: set[str] = set()

        for event in session_events:
            evt = event.get("event", "")
            detail = event.get("detail", {})

            if evt in discovery_events:
                msg = detail.get("message", "") or detail.get("query", "")
                topic = detail.get("topic", "")
                discovery_topics |= set(re.findall(r'\w{4,}', (msg + " " + topic).lower()))
                # Collect files from discovery
                discovery_files |= _extract_files_from_event(event)

            elif evt == "claim" and discovery_topics:
                # Check if the claim relates to what was discovered
                claim_text = event.get("summary", "").lower()
                claim_words = set(re.findall(r'\w{4,}', claim_text))
                overlap = claim_words & discovery_topics
                if len(overlap) >= 2:
                    domain = event.get("domain", "?")
                    addrs = event.get("addresses", [])
                    discovery_sequences.append({
                        "session": session_id,
                        "domain": domain,
                        "overlap_words": sorted(overlap)[:5],
                        "addr": addrs[0][:12] if addrs else "?",
                    })

    # Deduplicate by addr
    seen_addrs = set()
    unique_discoveries = []
    for d in discovery_sequences:
        if d["addr"] not in seen_addrs:
            seen_addrs.add(d["addr"])
            unique_discoveries.append(d)

    # ── Core knowledge: addresses in >70% of recall events ─────────
    recall_event_count = 0
    addr_recall_freq: Counter = Counter()

    for session_events in sessions.values():
        for event in session_events:
            if event.get("event") == "recall":
                recall_event_count += 1
                for addr in event.get("addresses", []):
                    addr_recall_freq[addr] += 1

    core_knowledge = []
    if recall_event_count > 0:
        for addr, count in addr_recall_freq.most_common(15):
            frequency = count / recall_event_count
            if frequency < 0.1:
                break
            node = store.get(addr)
            snippet = node.content[:80] if node else "(superseded)"
            domain = node.meta.get("domain", "?") if node else "?"
            core_knowledge.append({
                "addr": addr[:12],
                "frequency": round(frequency, 2),
                "appearances": count,
                "total_recalls": recall_event_count,
                "domain": domain,
                "snippet": snippet,
            })

    # ── Domain cascades: claim(A) → claim(B) patterns ──────────────
    domain_pairs: Counter = Counter()

    for session_events in sessions.values():
        prev_claim_domain = None
        for event in session_events:
            if event.get("event") == "claim":
                domain = event.get("domain", "")
                if domain and prev_claim_domain and domain != prev_claim_domain:
                    domain_pairs[(prev_claim_domain, domain)] += 1
                if domain:
                    prev_claim_domain = domain

    domain_cascades = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in domain_pairs.most_common(10)
        if c >= 2
    ]

    # ── Recall triggers: message keywords → consistent node sets ───
    # For each keyword, count how many recall *events* it appears in,
    # and how many of those events include a given node.
    # Consistency = events_with_node / events_with_keyword
    _TRIGGER_STOP = {
        "memory", "recall", "project", "tree", "node", "store",
        "module", "file", "code", "this", "that", "what",
        "with", "from", "about", "into", "have", "been",
        "does", "when", "where", "which", "there", "their",
        "they", "your", "just", "some", "like", "make",
        "more", "also", "well", "very", "know", "think",
        "want", "need", "look", "work", "could", "would",
        "should", "will", "each", "other", "after", "before",
    }

    keyword_event_count: Counter = Counter()
    keyword_node_events: dict[str, Counter] = defaultdict(Counter)

    for session_events in sessions.values():
        for event in session_events:
            if event.get("event") != "recall":
                continue
            msg = event.get("detail", {}).get("message", "")
            addrs = set(event.get("addresses", []))
            words = set(re.findall(r'\w{4,}', msg.lower())) - _TRIGGER_STOP
            for word in words:
                keyword_event_count[word] += 1
                for addr in addrs:
                    keyword_node_events[word][addr] += 1

    recall_triggers = []
    for word, event_count in keyword_event_count.most_common():
        if event_count < 3:
            continue
        if not keyword_node_events[word]:
            continue
        top_addr, top_count = keyword_node_events[word].most_common(1)[0]
        consistency = top_count / event_count
        if consistency >= 0.6 and top_count >= 3:
            node = store.get(top_addr)
            recall_triggers.append({
                "keyword": word,
                "top_node": top_addr[:12],
                "consistency": round(consistency, 2),
                "appearances": top_count,
                "events": event_count,
                "snippet": node.content[:60] if node else "?",
            })

    recall_triggers.sort(key=lambda x: (-x["consistency"], -x["events"]))

    return {
        "volatile_nodes": volatile_nodes,
        "volatile_by_domain": volatile_by_domain,
        "discovery_sequences": unique_discoveries[:10],
        "core_knowledge": core_knowledge,
        "domain_cascades": domain_cascades,
        "recall_triggers": recall_triggers[:15],
    }


# -------------------------------------------------------------------
# Layer 5: Workflow patterns
# -------------------------------------------------------------------

def analyze_workflow(logs_dir: str) -> dict:
    """Detect workflow patterns from session structure.

    Returns:
        {
            "domain_sequences": [...],     # common domain transition patterns
            "session_profiles": {...},      # avg events, duration, domains per session
            "event_distribution": {...},    # which events are most common
        }
    """
    events = _iter_log_events(logs_dir)
    sessions = _group_by_session(events)

    # Domain transition sequences
    domain_transitions: Counter = Counter()
    event_counts: Counter = Counter()
    session_stats = []

    for session_id, session_events in sessions.items():
        if len(session_events) < 2:
            continue

        prev_domain = None
        session_domains = set()

        for event in session_events:
            evt = event.get("event", "")
            event_counts[evt] += 1

            domain = event.get("domain")
            if domain:
                session_domains.add(domain)
                if prev_domain and prev_domain != domain:
                    domain_transitions[(prev_domain, domain)] += 1
                prev_domain = domain

        session_stats.append({
            "session": session_id,
            "events": len(session_events),
            "domains": len(session_domains),
        })

    # Aggregate session profiles
    if session_stats:
        avg_events = sum(s["events"] for s in session_stats) / len(session_stats)
        avg_domains = sum(s["domains"] for s in session_stats) / len(session_stats)
    else:
        avg_events = 0
        avg_domains = 0

    return {
        "domain_sequences": [
            {"from": a, "to": b, "count": c}
            for (a, b), c in domain_transitions.most_common(10)
            if c >= 2
        ],
        "session_profiles": {
            "total_sessions": len(sessions),
            "avg_events_per_session": round(avg_events, 1),
            "avg_domains_per_session": round(avg_domains, 1),
        },
        "event_distribution": dict(event_counts.most_common(15)),
    }


# -------------------------------------------------------------------
# Main inference entry point
# -------------------------------------------------------------------

def infer(store: Store,
          logs_dir: str = None,
          layers: list[str] = None) -> str:
    """Run pattern inference across all layers.

    Args:
        store: The node store
        logs_dir: Path to logs directory (default: store_path/logs)
        layers: Which layers to run. Default: all five.
                Options: "cooccurrence", "recall", "corrections", "sequences", "workflow"

    Returns structured report suitable for display.
    """
    if logs_dir is None:
        logs_dir = os.path.join(os.path.dirname(store.index_dir), "logs")

    if layers is None:
        layers = ["cooccurrence", "recall", "corrections", "sequences", "workflow"]

    parts = ["## Pattern inference report\n"]

    # ── Layer 1: Co-occurrence ─────────────────────────────────────
    if "cooccurrence" in layers:
        cooc = analyze_cooccurrence(logs_dir)
        parts.append("### File co-occurrence\n")
        if cooc:
            parts.append("  Files frequently discussed/edited together:\n")
            for item in cooc[:15]:
                parts.append(
                    f"  {item['file_a']} <-> {item['file_b']}  "
                    f"({item['sessions']} sessions, "
                    f"strength={item['strength']})"
                )

            # Propose links
            strong = [c for c in cooc if c["strength"] > 0.3]
            if strong:
                parts.append(f"\n  {len(strong)} pair(s) strong enough "
                             "to propose relates_to links.")
        else:
            parts.append("  Not enough session data for co-occurrence analysis.")
        parts.append("")

    # ── Layer 2: Recall patterns ───────────────────────────────────
    if "recall" in layers:
        recall = analyze_recall_patterns(logs_dir, store)
        parts.append("### Recall patterns\n")

        if recall["noise_candidates"]:
            parts.append("  Noise candidates (recalled often, rarely acted on):")
            for item in recall["noise_candidates"]:
                parts.append(
                    f"    [{item['addr'][:8]}] {item['recalls']} recalls, "
                    f"{item['actions']} actions "
                    f"({item['hit_rate']:.0%} hit rate): "
                    f"{item['snippet']}"
                )
            parts.append("")

        if recall["high_value"]:
            parts.append("  High-value nodes (recalled and acted on):")
            for item in recall["high_value"]:
                parts.append(
                    f"    [{item['addr'][:8]}] {item['recalls']} recalls, "
                    f"{item['actions']} actions "
                    f"({item['hit_rate']:.0%} hit rate): "
                    f"{item['snippet']}"
                )
            parts.append("")

        if recall["co_recalled"]:
            parts.append("  Co-recalled pairs (implicit associations):")
            for item in recall["co_recalled"]:
                parts.append(
                    f"    {item['addr_a'][:8]} <-> {item['addr_b'][:8]}  "
                    f"({item['co_recalls']} co-recalls, "
                    f"affinity={item['affinity']})"
                )
            parts.append("")

        if recall["never_recalled"]:
            parts.append("  Never recalled (active but invisible):")
            for item in recall["never_recalled"]:
                parts.append(
                    f"    [{item['addr'][:8]}] [{item['domain']}] "
                    f"{item['snippet']}"
                )
            parts.append("")

        if not any(recall.values()):
            parts.append("  Not enough recall data for pattern analysis.")
            parts.append("")

    # ── Layer 3: Corrections ──────────────────────────────────────
    if "corrections" in layers:
        corr = analyze_corrections(logs_dir, store)
        parts.append("### Correction patterns\n")

        if corr["correction_types"]:
            parts.append("  Correction types:")
            for ctype, count in corr["correction_types"].items():
                parts.append(f"    {ctype}: {count}")
            parts.append("")

        if corr["domain_stability"]:
            parts.append("  Domain stability (higher correction rate = less stable):")
            for item in corr["domain_stability"]:
                bar = "#" * int(item["correction_rate"] * 20)
                parts.append(
                    f"    {item['domain']:15s}  {item['claims']:3d} claims, "
                    f"{item['updates']:3d} updates  "
                    f"({item['correction_rate']:.0%}) {bar}"
                )
            parts.append("")

        if corr["user_corrections"]:
            parts.append("  User corrections (direct feedback):")
            for item in corr["user_corrections"]:
                parts.append(
                    f"    [{item['addr'][:8]}] [{item['domain']}] "
                    f"{item['reason']}"
                )
            parts.append("")

        if corr["rapid_corrections"]:
            parts.append("  Rapid corrections (claimed and updated same session):")
            for item in corr["rapid_corrections"]:
                parts.append(
                    f"    [{item['old_addr'][:8]}] ({item['type']}) "
                    f"{item['reason']}"
                )
            parts.append("")

        if corr["supersession_chains"]:
            parts.append("  Supersession chains (updated multiple times):")
            for item in corr["supersession_chains"]:
                parts.append(
                    f"    [{item['addr'][:8]}] {item['updates']}x updated: "
                    f"{item['snippet']}"
                )
            parts.append("")

        if not any(corr.values()):
            parts.append("  No correction data available.")
            parts.append("")

    # ── Layer 4: Sequences ─────────────────────────────────────────
    if "sequences" in layers:
        seq = analyze_sequences(logs_dir, store)
        parts.append("### Implicit knowledge (sequence patterns)\n")

        if seq["volatile_nodes"]:
            parts.append("  Volatile nodes (recalled then updated in same session):")
            for item in seq["volatile_nodes"]:
                parts.append(
                    f"    [{item['old_addr'][:8]}] -> [{item['new_addr'][:8]}] "
                    f"[{item['domain']}]: {item['snippet']}"
                )
            if seq.get("volatile_by_domain"):
                parts.append("")
                parts.append("  Volatility by domain:")
                for domain, count in seq["volatile_by_domain"].most_common():
                    parts.append(f"    {domain}: {count} recall->update cycles")
            parts.append("")

        if seq["core_knowledge"]:
            parts.append("  Core knowledge (most frequently recalled nodes):")
            for item in seq["core_knowledge"]:
                parts.append(
                    f"    [{item['addr'][:8]}] [{item['domain']}] "
                    f"{item['frequency']:.0%} ({item['appearances']}/"
                    f"{item['total_recalls']}): "
                    f"{item['snippet']}"
                )
            parts.append("")

        if seq["discovery_sequences"]:
            parts.append("  Discovery sequences (recall/search -> claim):")
            for item in seq["discovery_sequences"]:
                words = ", ".join(item["overlap_words"][:3])
                parts.append(
                    f"    [{item['addr'][:8]}] [{item['domain']}] "
                    f"triggered by: {words}"
                )
            parts.append("")

        if seq["domain_cascades"]:
            parts.append("  Domain cascades (claim A -> claim B patterns):")
            for item in seq["domain_cascades"]:
                parts.append(
                    f"    {item['from']} -> {item['to']}  "
                    f"({item['count']}x)"
                )
            parts.append("")

        if seq["recall_triggers"]:
            parts.append("  Recall triggers (keywords -> consistent nodes):")
            for item in seq["recall_triggers"]:
                parts.append(
                    f"    \"{item['keyword']}\" -> [{item['top_node'][:8]}] "
                    f"({item['consistency']:.0%} of {item['events']} events): "
                    f"{item['snippet']}"
                )
            parts.append("")

        if not any(seq.values()):
            parts.append("  Not enough session data for sequence analysis.")
            parts.append("")

    # ── Layer 5: Workflow ──────────────────────────────────────────
    if "workflow" in layers:
        workflow = analyze_workflow(logs_dir)
        parts.append("### Workflow patterns\n")

        prof = workflow["session_profiles"]
        parts.append(f"  Sessions analyzed: {prof['total_sessions']}")
        parts.append(f"  Avg events/session: {prof['avg_events_per_session']}")
        parts.append(f"  Avg domains/session: {prof['avg_domains_per_session']}")
        parts.append("")

        if workflow["domain_sequences"]:
            parts.append("  Common domain transitions:")
            for item in workflow["domain_sequences"]:
                parts.append(
                    f"    {item['from']} -> {item['to']}  "
                    f"({item['count']}x)"
                )
            parts.append("")

        if workflow["event_distribution"]:
            parts.append("  Event distribution:")
            for evt, count in list(workflow["event_distribution"].items())[:10]:
                parts.append(f"    {evt}: {count}")
            parts.append("")

    # ── Summary ────────────────────────────────────────────────────
    parts.append("### Inference summary\n")
    parts.append(f"  Layers run: {', '.join(layers)}")
    parts.append(f"  Log directory: {logs_dir}")

    return "\n".join(parts)

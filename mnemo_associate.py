"""
mnemo_associate.py — The subconscious layer (project development)

A lightweight associative retrieval system that runs BEFORE the main
model thinks. Given an incoming message, it traverses the project
memory tree and surfaces relevant nodes, tensions, and associations.

This is the small model's job. It doesn't interpret. It doesn't
author. It associates. The output becomes the substrate on which
the working model's thinking forms.

== What it surfaces for dev work ==

When starting a fresh session on a big project, the things that
matter most are:
  - Why is the code structured this way? (architecture, decisions)
  - What patterns/conventions should I follow? (patterns)
  - What's currently in progress or blocked? (tasks)
  - What breaks or has known gotchas? (issues)
  - What was already tried and didn't work? (history)
  - What dependency constraints exist? (dependencies)

== The output format ==

The associate produces a "preload" — a structured context block
that feels like project knowledge, not search results. The model
should experience this as "what I already know about this project"
not "what was retrieved."

Example:
    This project uses FastAPI with SQLAlchemy async. The auth module
    was recently refactored from middleware to dependency injection
    [a7f3c2d1]. There's a known issue with connection pooling under
    load [b91e4d3f]. Convention: all route handlers go in routes/,
    business logic in services/ [c3d4e5f6].

Not:
    Retrieved 3 nodes: [architecture] uses FastAPI, [issues] connection
    pooling bug, [patterns] file structure convention...
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from mnemo import Store, Node, GENESIS

# Lazy-initialized retrieval backends — keyed by index_dir to support
# multiple stores (project + global) without conflicts
_retrieval_backends: dict[str, object] = {}


# ===================================================================
# Signal extraction (lightweight — no LLM needed)
# ===================================================================

# Common stop words to ignore in matching
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "like",
    "through", "after", "before", "between", "under", "above", "up",
    "down", "out", "off", "over", "again", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "that", "this",
    "what", "which", "who", "whom", "these", "those", "i", "me", "my",
    "you", "your", "we", "our", "they", "them", "their", "it", "its",
    "im", "ive", "dont", "didnt", "cant", "wont", "thats", "whats",
    "youre", "hes", "shes", "were", "theyre", "ill", "youll",
    "file", "code", "make", "use", "using", "used", "need", "want",
    "get", "got", "let", "lets", "look", "looks", "thing", "things",
}


def extract_signals(message: str) -> dict:
    """
    Extract associative signals from a message in a dev context.
    No LLM needed — just pattern matching and keyword extraction.
    """
    lower = message.lower()
    words = set(re.findall(r'[a-z_]+', lower)) - STOP_WORDS

    # Filter to meaningful words (3+ chars)
    keywords = {w for w in words if len(w) >= 3}

    # Detect question patterns
    is_question = "?" in message
    question_type = None
    if is_question:
        if any(p in lower for p in ["how does", "how do", "how is", "how are"]):
            question_type = "how_works"
        elif any(p in lower for p in ["where is", "where do", "where are", "which file",
                                       "which module", "where does"]):
            question_type = "find_location"
        elif any(p in lower for p in ["why does", "why is", "why do", "why did",
                                       "what's the reason", "why was"]):
            question_type = "why_decision"
        elif any(p in lower for p in ["what pattern", "what convention", "how should",
                                       "what's the standard", "what approach"]):
            question_type = "convention"
        else:
            question_type = "general"

    # Detect architecture/design discussion
    architecture_discussion = bool(re.search(
        r'\b(architect|design|pattern|structur|module|component|layer|'
        r'service|interface|abstract|refactor|reorganiz|decouple|'
        r'separate|split|merge|monolith|microservice)\w*\b', lower))

    # Detect debugging / issue signals
    debugging = bool(re.search(
        r'\b(error|bug|broken|fail|exception|crash|stack\s*trace|'
        r'issue|problem|wrong|unexpected|doesn.t work|not working|'
        r'regression|flaky|hang|timeout|leak|corrupt|segfault)\w*\b', lower))

    # Detect decision-making signals
    decision_making = bool(re.search(
        r'\b(decided|choosing|chose|trade.?off|instead of|went with|'
        r'going with|option|alternative|approach|let.s go with|'
        r'pros and cons|better to|should we|picked|selected|'
        r'switched to|migrated|moving to)\b', lower))

    # Detect convention/pattern-setting
    convention_setting = bool(re.search(
        r'\b(always|never|convention|rule|standard|pattern|'
        r'consistent|naming|style|format|prefix|suffix|'
        r'we do it|the way we|our approach|best practice)\b', lower))

    # Detect dependency discussion
    dependency_talk = bool(re.search(
        r'\b(install|upgrade|downgrade|version|package|library|'
        r'dependency|import|require|pip|npm|cargo|gem|maven|'
        r'compatible|breaking change|deprecat)\w*\b', lower))

    # Detect task/progress signals
    task_signal = bool(re.search(
        r'\b(todo|fixme|hack|working on|in progress|blocked|'
        r'done|finished|completed|next|priority|sprint|'
        r'milestone|deadline|ship|deploy|release)\w*\b', lower))

    # Detect "what was tried" / history signals
    history_signal = bool(re.search(
        r'\b(tried|attempted|didn.t work|abandoned|reverted|'
        r'rolled back|used to|previously|before we|originally|'
        r'old approach|failed attempt|lesson learned)\b', lower))

    # Detect correction/update patterns (something changed)
    is_correction = bool(re.search(
        r'\b(actually|not anymore|changed|moved|no longer|'
        r'switched|updated|replaced|deprecated|removed|'
        r'renamed|restructured)\b', lower))

    # Signal density — how much context this message needs
    domain_signals = sum([
        architecture_discussion, debugging, decision_making,
        convention_setting, dependency_talk, task_signal,
        history_signal, is_correction,
    ])
    signal_density = "high" if (domain_signals >= 2 or is_question or len(keywords) >= 6) \
        else "medium" if (domain_signals >= 1 or len(keywords) >= 3) \
        else "low"

    return {
        "keywords": keywords,
        "is_question": is_question,
        "question_type": question_type,
        "architecture_discussion": architecture_discussion,
        "debugging": debugging,
        "decision_making": decision_making,
        "convention_setting": convention_setting,
        "dependency_talk": dependency_talk,
        "task_signal": task_signal,
        "history_signal": history_signal,
        "is_correction": is_correction,
        "message_length": len(message),
        "signal_density": signal_density,
    }


# ===================================================================
# Tree traversal / retrieval
# ===================================================================

def _get_backend(store: Store):
    """Lazy-init the retrieval backend for a given store.

    MNEMO_RETRIEVAL env var selects the backend:
    - "tfidf" (default): keyword-based, zero external deps
    - "embedding": semantic vectors via API (Voyage/OpenAI), falls back to TF-IDF
    """
    key = store.index_dir
    if key not in _retrieval_backends:
        retrieval = os.environ.get("MNEMO_RETRIEVAL", "tfidf")

        if retrieval == "embedding":
            from mnemo_retrieval import EmbeddingBackend, make_embedder
            result = make_embedder()
            if result:
                embed_fn, model_name = result
                _retrieval_backends[key] = EmbeddingBackend(
                    store.index_dir, embed_fn, model_name)
                return _retrieval_backends[key]
            # No provider available — fall through to TF-IDF

        from mnemo_retrieval import TfIdfBackend
        _retrieval_backends[key] = TfIdfBackend(store.index_dir)
    return _retrieval_backends[key]


def retrieve_relevant(message: str, store: Store,
                      max_nodes: int = 8, backend=None,
                      session_context: dict = None) -> list[dict]:
    """
    Given a message, find the most relevant nodes in the active set.

    Uses a RetrievalBackend (default: TF-IDF) for text similarity,
    layered with domain-aware signal boosts and session affinity.

    session_context (optional):
        session_addrs: set of addresses created/modified this session
        recalled_recent: list of sets — each set is addresses surfaced
                        on a recent turn, most recent last

    Returns nodes scored and sorted by relevance, each with:
    - node: the Node object
    - score: relevance score
    - reason: why this was surfaced (for debugging, not shown to model)
    """
    if backend is None:
        backend = _get_backend(store)
    backend.ensure_fresh(store)
    backend.prepare_query(message)

    signals = extract_signals(message)
    keywords = signals["keywords"]
    active = store.get_active()

    # Unpack session context
    session_addrs = set()
    recalled_recent = []
    focus_file = ""
    if session_context:
        session_addrs = session_context.get("session_addrs", set())
        recalled_recent = session_context.get("recalled_recent", [])
        focus_file = session_context.get("focus_file", "")

    scored = []

    now = time.time()
    for addr in active:
        node = store.get(addr)
        if not node:
            continue

        # TTL expiry — skip silently if past expiry date
        ttl = node.meta.get("ttl_days", 0)
        if ttl and (now - node.created) > ttl * 86400:
            continue

        score = 0.0
        reasons = []

        # Text similarity via retrieval backend
        similarity = backend.score(keywords, addr)
        if similarity > 0:
            # Length penalty: nodes with many unique terms match more queries
            # but aren't necessarily more relevant. Penalize proportionally.
            content_words = len(set(node.content.lower().split()))
            length_factor = min(1.0, 30.0 / max(content_words, 1))  # full score at ≤30 words, decays above
            adjusted = similarity * length_factor
            score += adjusted * 3.0
            reasons.append(f"tfidf: {similarity:.3f} (len_adj: {length_factor:.2f})")

        # Domain relevance boosting
        domain = node.meta.get("domain", "")

        # Architecture discussion → boost architecture + decisions nodes
        if signals["architecture_discussion"] and domain in ("architecture", "decisions"):
            score += 0.8
            reasons.append("architecture context boost")

        # Debugging → boost issues nodes heavily
        if signals["debugging"] and domain == "issues":
            score += 1.0
            reasons.append("debugging boost")

        # Debugging → also boost architecture (understanding structure helps debug)
        if signals["debugging"] and domain == "architecture":
            score += 0.3
            reasons.append("architecture for debugging")

        # Decision-making → boost decisions and history (what was tried before)
        if signals["decision_making"] and domain in ("decisions", "history"):
            score += 0.8
            reasons.append("decision context boost")

        # Convention question → boost patterns
        if signals["convention_setting"] and domain == "patterns":
            score += 0.8
            reasons.append("convention boost")
        if signals["question_type"] == "convention" and domain == "patterns":
            score += 1.0
            reasons.append("convention question boost")

        # "How does X work" → boost architecture
        if signals["question_type"] == "how_works" and domain == "architecture":
            score += 0.8
            reasons.append("how-works boost")

        # "Why" questions → boost decisions
        if signals["question_type"] == "why_decision" and domain == "decisions":
            score += 1.0
            reasons.append("why-decision boost")

        # "Where is" → boost architecture (module responsibilities)
        if signals["question_type"] == "find_location" and domain == "architecture":
            score += 0.5
            reasons.append("location boost")

        # Dependency discussion → boost dependencies
        if signals["dependency_talk"] and domain == "dependencies":
            score += 0.8
            reasons.append("dependency boost")

        # Task signals → boost tasks
        if signals["task_signal"] and domain == "tasks":
            score += 0.7
            reasons.append("task boost")

        # History signals → boost history (what was tried)
        if signals["history_signal"] and domain == "history":
            score += 0.8
            reasons.append("history boost")

        # Correction detected → relevant nodes get extra weight
        if signals["is_correction"]:
            score += 0.3
            reasons.append("correction context")

        # Recency boost — fresher nodes slightly preferred
        age_days = (time.time() - node.created) / 86400
        if age_days < 1:
            score += 0.3
            reasons.append("very fresh")
        elif age_days < 7:
            score += 0.1
            reasons.append("recent")

        # Reinforcement boost — frequently reinforced nodes are important
        reinforcements = node.meta.get("reinforcement_count", 0)
        if reinforcements > 0:
            score += min(reinforcements * 0.2, 1.0)
            reasons.append(f"reinforced {reinforcements}x")

        # Utility signal — recall feedback loop
        # Nodes recalled often but never acted on are likely noise;
        # nodes that trigger action are high-value.
        recall_count = node.meta.get("recall_count", 0)
        recall_hits = node.meta.get("recall_hits", 0)
        if recall_count >= 3:  # minimum sample before signal activates
            hit_rate = recall_hits / recall_count
            utility_boost = (hit_rate - 0.4) * 0.5  # maps [0,1] -> [-0.2, +0.3]
            score += utility_boost
            reasons.append(f"utility: {hit_rate:.0%} hit rate ({recall_hits}/{recall_count})")

        # Session affinity — boost nodes we're actively working with
        if addr in session_addrs:
            score += 1.0
            reasons.append("session-created")

        for turns_ago, turn_addrs in enumerate(reversed(recalled_recent)):
            if addr in turn_addrs:
                boost = 0.5 * (0.7 ** turns_ago)  # decay: 0.5, 0.35, 0.24...
                score += boost
                reasons.append(f"recalled {turns_ago + 1} turn(s) ago")
                break

        # Focus boost — nodes about the file this agent is actively editing
        if focus_file:
            anchors = node.meta.get("anchors", [])
            focus_base = focus_file.split("/")[-1]  # e.g. "mnemo_mcp.py"
            focus_stem = focus_base.rsplit(".", 1)[0]  # e.g. "mnemo_mcp"
            if any(focus_file in a.get("path", "") or focus_base in a.get("path", "")
                   for a in anchors):
                score += 0.4
                reasons.append(f"focus: {focus_base}")
            elif focus_stem in node.content.lower():
                score += 0.2
                reasons.append(f"focus mention: {focus_base}")

        # Priority boost — high-priority nodes (user preferences, working
        # agreements, critical invariants) surface above regular facts
        priority = node.meta.get("priority", 0)
        if priority:
            score += priority
            reasons.append(f"priority: +{priority}")

        # Confidence with temporal decay — unreinforced nodes fade gently
        base_confidence = node.meta.get("confidence", 0.5)
        last_fresh = node.meta.get("last_reinforced", node.created)
        days_since_fresh = (time.time() - last_fresh) / 86400
        decay_rate = 0.02  # lose ~50% confidence over 25 days unreinforced
        decay_floor = 0.3  # never below 30% of base confidence
        decay_factor = max(decay_floor, 1.0 - (days_since_fresh * decay_rate))

        # Drifted anchors accelerate decay — code changed, description stale
        anchors = node.meta.get("anchors", [])
        if any(a.get("drifted") for a in anchors):
            decay_factor = max(decay_floor, decay_factor * 0.6)
            reasons.append("anchor drifted")

        confidence = base_confidence * decay_factor
        score *= (0.5 + confidence * 0.5)
        if decay_factor < 0.8:
            reasons.append(f"confidence decayed to {confidence:.2f}")

        # Auto-claim nodes (write/edit history) are useful for file context
        # but shouldn't dominate general recall — they're working notes, not knowledge.
        if node.meta.get("auto_claim"):
            score *= 0.25
            reasons.append("auto-claim penalty")

        if score > 0:
            scored.append({
                "node": node,
                "score": round(score, 3),
                "reasons": reasons,
            })

    # --- Link traversal: multi-hop BFS (depth 3) from high-scoring seeds ---
    # Score decays by HOP_DECAY per additional hop from the seed.
    # Causal links (caused_by, depends_on, blocks) propagate stronger than
    # associative links (relates_to, enables, contradicts).
    _CAUSAL_RELS = frozenset({"caused_by", "depends_on", "blocks"})
    _FWD_CAUSAL  = 0.6   # forward causal weight
    _FWD_REG     = 0.4   # forward regular weight
    _REV_CAUSAL  = 0.5   # reverse causal weight (weaker — provenance is noisier)
    _REV_REG     = 0.3   # reverse regular weight
    _HOP_DECAY   = 0.6   # score multiplier per hop beyond the first
    _MAX_DEPTH   = 3     # maximum hops from seed
    _BUDGET      = 50    # max nodes expanded in BFS (guards against dense graphs)

    scored_addrs = {item["node"].addr: item for item in scored}
    link_boosts: dict[str, float] = {}
    visited: set[str] = set(scored_addrs.keys())  # seeds already scored
    expanded = 0

    # frontier entries: (node, seed_score, depth)
    frontier = [(item["node"], item["score"], 1) for item in scored]

    while frontier and expanded < _BUDGET:
        next_frontier = []
        for source_node, seed_score, depth in frontier:
            if expanded >= _BUDGET:
                break
            decay = _HOP_DECAY ** (depth - 1)  # hop 1: 1.0x, hop 2: 0.6x, hop 3: 0.36x

            # Forward links: source_node → target
            for link in source_node.meta.get("links", []):
                target = link.get("addr", "")
                rel = link.get("rel", "relates_to")
                if not target or target not in active:
                    continue
                w = _FWD_CAUSAL if rel in _CAUSAL_RELS else _FWD_REG
                boost = seed_score * w * decay
                link_boosts[target] = link_boosts.get(target, 0) + boost
                if target not in visited and depth < _MAX_DEPTH:
                    visited.add(target)
                    t_node = store.get(target)
                    if t_node:
                        next_frontier.append((t_node, seed_score, depth + 1))
                        expanded += 1

            # Reverse links: other nodes → source_node
            for rl in store.get_reverse_links(source_node.addr):
                src = rl.get("source_addr", "")
                rel = rl.get("rel", "relates_to")
                if not src or src not in active:
                    continue
                w = _REV_CAUSAL if rel in _CAUSAL_RELS else _REV_REG
                boost = seed_score * w * decay
                link_boosts[src] = link_boosts.get(src, 0) + boost
                if src not in visited and depth < _MAX_DEPTH:
                    visited.add(src)
                    r_node = store.get(src)
                    if r_node:
                        next_frontier.append((r_node, seed_score, depth + 1))
                        expanded += 1

        frontier = next_frontier

    # Apply link boosts — add to existing scored nodes or pull in new ones
    for addr, boost in link_boosts.items():
        if addr in scored_addrs:
            scored_addrs[addr]["score"] = round(scored_addrs[addr]["score"] + boost, 3)
            scored_addrs[addr]["reasons"].append(f"linked: +{boost:.2f}")
        else:
            node = store.get(addr)
            if node:
                scored.append({
                    "node": node,
                    "score": round(boost, 3),
                    "reasons": [f"linked: +{boost:.2f}"],
                })

    # Sort by score descending
    scored.sort(key=lambda x: -x["score"])
    return scored[:max_nodes]


def detect_tensions(message: str, relevant: list[dict],
                    store: Store) -> list[dict]:
    """
    Detect potential tensions between the incoming message and
    existing project memory. Returns pairs of (node, tension_description).

    Tensions in a dev context:
    - Message describes behavior that contradicts a stored claim
    - A dependency or version has changed
    - A task status may be outdated
    - An approach was chosen that contradicts a stored decision
    """
    signals = extract_signals(message)
    tensions = []

    if not signals["is_correction"]:
        return tensions

    for item in relevant:
        node = item["node"]
        if item["score"] > 0.5:
            tensions.append({
                "node": node,
                "type": "potential_update",
                "description": f"May be outdated: {node.content[:60]}",
            })

    return tensions


# ===================================================================
# Recall budget
# ===================================================================

def _recall_budget() -> dict:
    """
    Return character budgets for recall output sections.
    Configurable via MNEMO_RECALL_BUDGET (total chars).
    """
    total = int(os.environ.get("MNEMO_RECALL_BUDGET", 25000))
    return {
        "chains": int(total * 0.80),      # 20,000 of 25,000
        "standalone": int(total * 0.12),  # 3,000
        "pings": int(total * 0.08),       # 2,000 (Phase 3+)
        "total": total,
    }


# ===================================================================
# Preload formatting — chain-first (v2) and flat-node (v1-compat)
# ===================================================================

def format_preload(
    relevant: list[dict],
    tensions: list[dict],
    signals: dict,
    store: Store | None = None,
    *,
    ranked_chains: list[dict] | None = None,
    narrative: bool = True,
) -> str:
    """
    Format the retrieved context as a natural preload.

    v2 path (ranked_chains provided): chain-narrative format.
    v1-compat path (ranked_chains is None): flat node list, legacy format.

    `store` is required for v2 chain rendering.
    Pings are handled separately in memory_recall before this output is returned.
    """
    if ranked_chains is not None and store is not None:
        return _format_chain_narrative(
            ranked_chains, relevant, tensions, signals, store
        )

    if not relevant and not tensions:
        return ""

    if narrative:
        return _format_narrative_v1(relevant, tensions, signals)
    return _format_structured(relevant, tensions)


def _format_chain_narrative(
    ranked_chains: list[dict],
    relevant: list[dict],
    tensions: list[dict],
    signals: dict,
    store: Store,
) -> str:
    """
    v2 format: reasoning paths as stories + standalone nodes.

    Layout:
      ── Chain 1: <summary> (<N> nodes) ──
      <tail content> [addr]
      ...
      <head content> [addr]

      ── Chain 2: ... ──
      ...

      ── Always active ──
      <high-priority standalone nodes>

      Possibly outdated: ...
    """
    from mnemo_chains import render_chain, render_standalone_nodes, get_chains_for_node

    budget = _recall_budget()
    parts = []
    chars_used = 0

    # --- Chains ---
    chain_budget = budget["chains"]
    per_chain = chain_budget // max(len(ranked_chains), 1) if ranked_chains else chain_budget

    # Collect addrs already covered by chains (to exclude from standalone)
    chained_addrs: set[str] = set()
    for chain in ranked_chains:
        chained_addrs.update(chain.get("members", []))

    for chain in ranked_chains:
        if chars_used >= chain_budget:
            break
        rendered = render_chain(chain, store, max_chars=per_chain)
        parts.append(rendered)
        chars_used += len(rendered)

    # --- Standalone nodes ---
    # Nodes that are either:
    #   (a) not members of any ranked chain, OR
    #   (b) high-priority (priority >= 0.5) — always surface these
    standalone_items = []
    for item in relevant:
        node = item["node"]
        priority = node.meta.get("priority", 0)
        in_chain = node.addr in chained_addrs

        # Always include high-priority nodes
        if priority >= 0.5:
            standalone_items.append(item)
        # Include unlinked nodes (not covered by any active chain)
        elif not in_chain:
            # Only include if it scored reasonably well on its own
            if item["score"] >= 0.5:
                standalone_items.append(item)

    # Sort standalone: priority desc, then score desc
    standalone_items.sort(
        key=lambda x: (
            -x["node"].meta.get("priority", 0),
            -x["score"],
        )
    )

    if standalone_items:
        rendered = render_standalone_nodes(
            standalone_items, max_chars=budget["standalone"]
        )
        if rendered:
            parts.append(rendered)

    # --- Tensions and meta-signals ---
    meta = []
    if tensions:
        notes = [t["description"] for t in tensions]
        meta.append("Possibly outdated: " + "; ".join(notes))
    if signals.get("is_correction"):
        meta.append("Something in the project may have changed — check for stale knowledge.")
    if signals.get("history_signal"):
        meta.append("They're referencing past approaches — check what was tried before.")
    if meta:
        parts.append("\n".join(meta))

    # Pings are rendered before recall content in memory_recall — not here.

    return "\n\n".join(p for p in parts if p)


def _format_narrative_v1(relevant: list[dict], tensions: list[dict],
                         signals: dict) -> str:
    """
    v1-compat flat node format. Used when no chains.json is present.
    Preserved exactly from v1 so existing stores work without migration.
    """
    parts = []

    if relevant:
        by_domain: dict = {}
        for item in relevant:
            domain = item["node"].meta.get("domain", "general")
            by_domain.setdefault(domain, []).append(item)

        memory_fragments = []
        for domain, items in by_domain.items():
            for item in items:
                content = item["node"].content
                addr = item["node"].addr
                preserved = item["node"].meta.get("preserved_values")
                if preserved:
                    pv = "; ".join(p["fragment"] for p in preserved[:10])
                    content = f"{content} (values: {pv})"
                memory_fragments.append(f"{content} [{addr[:8]}]")

        parts.append("Project context: " + " — ".join(memory_fragments))

    if tensions:
        parts.append("Possibly outdated: " + "; ".join(t["description"] for t in tensions))

    if signals.get("is_correction"):
        parts.append("Something in the project may have changed — check for stale knowledge.")
    if signals.get("history_signal"):
        parts.append("They're referencing past approaches — check what was tried before.")

    return "\n".join(parts)


def _format_structured(relevant: list[dict], tensions: list[dict]) -> str:
    """Structured debug format — unchanged from v1."""
    lines = []

    if relevant:
        lines.append("=== RELEVANT NODES ===")
        for item in relevant:
            n = item["node"]
            lines.append(
                f"  [{n.meta.get('domain', '?')}] {n.addr} "
                f"(score:{item['score']}) {n.content[:80]}"
            )

    if tensions:
        lines.append("=== TENSIONS ===")
        for t in tensions:
            lines.append(f"  {t['type']}: {t['description']}")

    return "\n".join(lines)


def _is_v2_store(store: Store) -> bool:
    """True if this store has a chains.json (v2 project-local store)."""
    return (store.root / "chains.json").exists()


# ===================================================================
# Main interface — what gets called per-message
# ===================================================================

def associate(message: str, store: Store,
              narrative: bool = True,
              max_nodes: int = 0,
              session_context: dict = None) -> dict:
    """
    The main entry point. Given a message and the store,
    produce the associative preload.

    v2 stores (chains.json present): chain-first retrieval.
      - Scores nodes, assembles into chains, returns narrative stories.
    v1-compat stores: flat node list (legacy format, no migration needed).

    max_nodes adapts to signal density if set to 0 (default):
      low density  → 3 nodes  (v1: 2)
      medium       → 8 nodes  (v1: 5)
      high         → 12 nodes (v1: 8)

    session_context (optional):
        session_addrs: set of node addrs created this session
        recalled_recent: list of sets of addrs surfaced on recent turns
        agent_id: current agent's id (for continuity boost)
        recently_extended_chain_ids: set of chain ids extended this session

    Returns:
        {
            "preload": str — context to inject
            "signals": dict — extracted signals (debug)
            "relevant_count": int — nodes surfaced
            "tension_count": int — tensions detected
            "relevant_addrs": list[str] — node addresses surfaced
            "signal_density": str — low/medium/high
            "chains": list[dict] | None — ranked chains (v2 only)
            "chain_count": int — chains rendered
        }
    """
    signals = extract_signals(message)
    v2 = _is_v2_store(store)

    # Adaptive node depth — cast wider net in v2 (chain assembly reduces noise)
    if max_nodes == 0:
        if v2:
            max_nodes = {"low": 3, "medium": 8, "high": 12}[signals["signal_density"]]
        else:
            max_nodes = {"low": 2, "medium": 5, "high": 8}[signals["signal_density"]]

    relevant = retrieve_relevant(message, store, max_nodes,
                                 session_context=session_context)
    tensions = detect_tensions(message, relevant, store)

    ranked_chains = None
    if v2 and relevant:
        try:
            from mnemo_chains import rank_chains
            node_scores = {item["node"].addr: item["score"] for item in relevant}
            agent_id = (session_context or {}).get("agent_id")
            recently_extended = (session_context or {}).get(
                "recently_extended_chain_ids"
            )
            ranked_chains = rank_chains(
                store,
                node_scores,
                agent_id=agent_id,
                recently_extended_ids=recently_extended,
                max_chains=3,
            )
        except Exception:
            ranked_chains = None  # graceful fallback to v1 format

    preload = format_preload(
        relevant,
        tensions,
        signals,
        store=store if v2 else None,
        ranked_chains=ranked_chains,
        narrative=narrative,
    )

    return {
        "preload": preload,
        "signals": signals,
        "relevant_count": len(relevant),
        "tension_count": len(tensions),
        "relevant_addrs": [item["node"].addr for item in relevant],
        "signal_density": signals["signal_density"],
        "chains": ranked_chains,
        "chain_count": len(ranked_chains) if ranked_chains else 0,
    }


# ===================================================================
# CLI for testing
# ===================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="mnemo associate — test associative retrieval")
    p.add_argument("message", help="Simulate a message")
    p.add_argument("--store", default=os.path.expanduser("~/mnemo"))
    p.add_argument("--structured", action="store_true", help="Show structured output instead of narrative")
    p.add_argument("--debug", action="store_true", help="Show signals and scoring")

    args = p.parse_args()

    s = Store(args.store)
    result = associate(args.message, s, narrative=not args.structured)

    if args.debug:
        print("=== SIGNALS ===")
        for k, v in result["signals"].items():
            print(f"  {k}: {v}")
        print()

    chain_info = f" | Chains: {result['chain_count']}" if result.get("chains") is not None else ""
    print(f"Relevant: {result['relevant_count']} nodes | Tensions: {result['tension_count']}{chain_info}")
    print()
    if result["preload"]:
        print(result["preload"])
    else:
        print("(no relevant context)")


if __name__ == "__main__":
    main()

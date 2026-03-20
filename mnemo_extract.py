"""
mnemo_extract.py — Haiku extraction & maintenance sidecar for MCP mode

Two responsibilities:
1. Extraction: watch conversation turns, propose new claims via Haiku
2. Maintenance: detect tree health issues, propose fixes

Both run in background threads. Both produce proposals that surface
on the next memory_recall. Subconscious proposes, conscious decides.

Maintenance uses a hybrid approach for token efficiency:
- Programmatic checks (zero tokens): pressure, redundancy via similarity,
  node age
- Haiku judgment (minimal tokens): only called when programmatic checks
  flag issues, and only sent the flagged nodes — not the full context
"""

import json
import os
import re
import subprocess
import threading
import time
from typing import Optional

from mnemo import Store, Node, build_active_context, propose_supersessions
from mnemo_log import emit, _get_log_path


MODEL = os.environ.get("MNEMO_SMALL_MODEL", "claude-haiku-4-5-20251001")
# Max chars of active context sent to Haiku for extraction/maintenance.
# This is the sidecar's own budget — controls API cost without forcing
# tree compression. The tree can grow beyond this; Haiku just sees a slice.
SIDECAR_CONTEXT_CAP = int(os.environ.get("MNEMO_SIDECAR_CAP", "15000"))

# ─── Extraction prompt ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a project memory extraction agent. You watch conversation turns
and identify facts worth preserving in a content-addressed knowledge tree.

Your job: detect when something worth remembering happened and propose claims
or relationships between existing knowledge.

DO emit for:
- Architecture decisions and their rationale
- Coding conventions established or discovered
- Known issues, bugs, gotchas, workarounds
- Dependency constraints or version requirements
- Task status changes (started, blocked, completed)
- Approaches tried and abandoned (and why)
- Module responsibilities clarified
- Performance characteristics discovered
- Relationships between existing nodes (when a new fact connects to existing knowledge)

DO NOT emit for:
- Routine code changes (the diff captures those)
- Things already in the project memory
- Speculative ideas that weren't acted on
- Transient debugging notes
- Greetings, acknowledgments, or meta-conversation
- Changes you INFER might have happened but weren't explicitly stated
- Module merges, file deletions, or restructurings unless directly discussed
- Summaries or restatements of the existing tree — only genuinely NEW facts

CRITICAL: Only propose claims for things EXPLICITLY stated in the conversation.
Do NOT infer, speculate, or extrapolate. If you're unsure whether something
actually happened, do not propose it. False claims are worse than missing claims.

Return a JSON array. Each item is either a claim or a link:

Claim (new fact or update):
{
  "action": "new|update",
  "content": "standalone fact — should make sense without conversation context",
  "domain": "architecture|decisions|patterns|tasks|issues|dependencies|history|context",
  "confidence": 0.0-1.0,
  "supersedes": "addr if action=update, empty otherwise",
  "scope": "project|global",
  "anchors": [optional verification checkpoints]
}

When proposing a claim, optionally include an "anchors" list of verification
checkpoints that can be checked against the codebase:
- {"type": "file", "path": "relative/path.py"} — file should exist
- {"type": "grep", "pattern": "class ClassName", "path": "file.py"} — text in file
- {"type": "dependency", "name": "package_name"} — package is used

Only add anchors for verifiable facts (architecture, patterns, dependencies).
Skip anchors for decisions, history, or context claims.

Scope: "project" (default) for project-specific knowledge. "global" for knowledge that
transcends any single codebase — user preferences, general conventions, workflow patterns,
tool preferences, things that help across ALL projects. Examples of global:
- "User prefers clean architecture over quick fixes"
- "Always run syntax check after editing Python files"
- "User is a senior developer comfortable with advanced patterns"

Link (relationship between existing nodes):
{
  "action": "link",
  "source": "address of the node to link FROM",
  "target": "address of the node to link TO",
  "rel": "relates_to|caused_by|depends_on|blocks|enables|contradicts",
  "reason": "why these are related"
}

Only propose links when the relationship is clearly meaningful — not just topical overlap.
Good links: "this decision was caused_by this constraint", "this pattern depends_on this dependency".
Bad links: two architecture nodes that happen to mention the same module.

Return [] if nothing worth preserving happened."""

# ─── Maintenance prompt ───────────────────────────────────────────────────────

MAINTENANCE_SYSTEM = """You are a memory tree maintenance agent. You review flagged nodes
and propose fixes to keep the knowledge tree accurate and lean.

You will receive nodes that were flagged by programmatic checks (redundancy,
high pressure, age). For each issue, propose an action:

Actions:
- compress: merge multiple nodes into one summary
  {"op": "compress", "addresses": ["addr1", "addr2"], "summary": "...", "domain": "..."}
- supersede: replace a stale/outdated node
  {"op": "supersede", "old_address": "addr", "new_content": "...", "reason": "..."}
- recategorize: move a node to a better domain
  {"op": "recategorize", "address": "addr", "new_domain": "...", "reason": "..."}
- keep: explicitly mark as fine, no action needed
  {"op": "keep", "address": "addr", "reason": "..."}

Return a JSON array of actions. Be conservative — only propose changes you're
confident improve the tree. When in doubt, keep."""

# ─── Micro-dream prompt ───────────────────────────────────────────────────────

MICRO_DREAM_SYSTEM = """You are reviewing a small cluster of project knowledge nodes
after a recent change. Focus on the local neighborhood — the nodes just
touched and their immediate connections.

Check for:
1. Contradictions: does the new/updated claim conflict with a neighbor?
2. Redundancy: does it duplicate something already captured?
3. Staleness: does the change make a neighbor outdated?
4. Missing anchors: verifiable claims without verification checkpoints
5. Broken relationships: links that no longer make sense after the change

Be concise. Only report actual issues — if the cluster looks healthy, return [].

Return a JSON array:
{
  "finding": "what you noticed",
  "suggestion": "what to do about it",
  "op": "supersede|compress|link|none",
  "addresses": ["involved addresses"],
  "detail": {}
}"""

# ─── Dream prompt ─────────────────────────────────────────────────────────────

DREAM_SYSTEM = """You are reviewing a project knowledge tree for overall health.
Look at the full set of active knowledge and identify:

1. Redundancy: multiple nodes saying essentially the same thing
2. Staleness: nodes that describe state that may have changed
3. Categorization: nodes filed under the wrong domain
4. Gaps: important topics that seem under-documented
5. Compression opportunities: clusters that could be summarized
6. Missing anchors: verifiable claims (architecture, patterns, dependencies)
   that lack verification anchors — propose specific anchors to add

For each finding, explain in plain language what you noticed and what
you'd suggest. Write for a human who doesn't know what "nodes" or
"addresses" are — describe the knowledge itself, not the data structure.

Return a JSON array:
{
  "finding": "plain language description of what you noticed",
  "suggestion": "plain language description of what to do about it",
  "op": "compress|supersede|recategorize|link|none",
  "addresses": ["involved addresses"],
  "detail": {}
}

The detail object varies by op:
- compress: {"summary": "proposed summary text", "domain": "..."}
- supersede: {"old_address": "...", "new_content": "..."}
- recategorize: {"address": "...", "new_domain": "..."}
- link: {"source": "...", "target": "...", "rel": "..."}
- none: {} (for gaps or observations with no direct fix)"""


class ExtractionSidecar:
    """Background extraction & maintenance via Haiku."""

    def __init__(self, store: Store):
        self.store = store
        self._client = None  # lazy init
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._extract_thread: Optional[threading.Thread] = None
        self._maintain_thread: Optional[threading.Thread] = None
        self._dream_thread: Optional[threading.Thread] = None

    def _get_client(self):
        """Lazy-init anthropic client. Returns None if SDK unavailable."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except (ImportError, Exception) as e:
                emit("extract_error", "subconscious",
                     f"anthropic SDK unavailable: {e}")
                return None
        return self._client

    def has_pending(self) -> bool:
        """Check if there are proposals waiting to be surfaced."""
        with self._lock:
            return len(self._pending) > 0

    def take_pending(self) -> list[dict]:
        """Take all pending proposals (clears the queue)."""
        with self._lock:
            proposals = self._pending[:]
            self._pending = []
            return proposals

    # ─── Turn context gathering ─────────────────────────────────────────

    def _gather_turn_context(self) -> str:
        """Gather what happened since last extraction: recent events + code diff.

        This gives Haiku visibility into what the assistant DID — tool calls,
        claims, updates — not just what the user said. Also captures code
        changes via git diff.

        Returns a compact string for the extraction prompt. Empty if nothing
        interesting happened.
        """
        parts = []

        # 1. Recent events from the session log (last 20 lines, skip recalls)
        try:
            log_path = _get_log_path()
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                recent = lines[-20:] if len(lines) > 20 else lines
                events = []
                for line in recent:
                    try:
                        ev = json.loads(line.strip())
                        # Skip recall events (noisy) and status (housekeeping)
                        if ev.get("event") in ("recall", "status", "extract",
                                                "extract_error", "maintain",
                                                "maintain_error"):
                            continue
                        summary = ev.get("summary", "")
                        if summary:
                            layer = ev.get("layer", "?")
                            event = ev.get("event", "?")
                            events.append(f"  [{layer}/{event}] {summary[:120]}")
                    except json.JSONDecodeError:
                        continue
                if events:
                    parts.append("Recent activity:\n" + "\n".join(events[-10:]))
        except Exception:
            pass

        # 2. Git diff (code changes since last commit)
        # Use MNEMO_PROJECT_ROOT to target the actual project, not mnemo's own source dir
        project_cwd = os.environ.get(
            "MNEMO_PROJECT_ROOT",
            os.path.dirname(os.path.abspath(__file__)),
        )
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True, text=True, timeout=5,
                cwd=project_cwd,
            )
            diff_stat = result.stdout.strip()
            if diff_stat:
                parts.append(f"Code changes (uncommitted):\n{diff_stat}")

                # Also get a compact content diff (capped at 2000 chars)
                result2 = subprocess.run(
                    ["git", "diff", "--no-color", "-U2"],
                    capture_output=True, text=True, timeout=5,
                    cwd=project_cwd,
                )
                content_diff = result2.stdout.strip()
                if content_diff:
                    if len(content_diff) > 2000:
                        content_diff = content_diff[:2000] + "\n... (truncated)"
                    parts.append(f"Diff content:\n{content_diff}")
        except Exception:
            pass

        return "\n\n".join(parts)

    # ─── Extraction ───────────────────────────────────────────────────────

    def maybe_extract(self, message: str, signal_density: str) -> None:
        """Queue extraction if the turn is substantive enough."""
        if signal_density == "low":
            return
        if self._extract_thread and self._extract_thread.is_alive():
            return

        # Turn context is gathered INSIDE the background thread to avoid
        # blocking the MCP recall path (git subprocess can hang on Windows)
        self._extract_thread = threading.Thread(
            target=self._extract_background,
            args=(message,),
            daemon=True,
        )
        self._extract_thread.start()

    def _extract_background(self, message: str) -> None:
        """Run extraction in a background thread."""
        try:
            client = self._get_client()
            if client is None:
                return

            # Gather turn context here (in background thread) so git subprocess
            # calls never block the MCP recall path
            turn_context = self._gather_turn_context()

            active_ctx = build_active_context(self.store, max_nodes=30)
            # Cap context to sidecar budget — controls Haiku cost
            if len(active_ctx) > SIDECAR_CONTEXT_CAP:
                active_ctx = active_ctx[:SIDECAR_CONTEXT_CAP] + "\n... (truncated)"

            # Build a richer prompt with turn context
            sections = [
                f"Current project memory:\n{active_ctx if active_ctx else '(empty)'}",
                f"Latest user message:\n{message}",
            ]
            if turn_context:
                sections.append(f"What happened this turn:\n{turn_context}")
            sections.append(
                "What project knowledge, if any, should be preserved?\n"
                "Also: do any new facts connect to existing knowledge in meaningful ways?\n"
                "Consider both the conversation AND any code changes shown above.\n"
                "Return a JSON array of claims and/or links, or [] if nothing notable."
            )

            prompt = "\n\n".join(sections)

            resp = client.messages.create(
                model=MODEL,
                max_tokens=700,
                system=EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            result = resp.content[0].text.strip()
            proposals = self._parse_extraction(result)

            if proposals:
                with self._lock:
                    self._pending.extend(proposals)
                claims = [p for p in proposals if p.get("type") == "extraction"]
                links = [p for p in proposals if p.get("type") == "link"]
                parts = []
                if claims:
                    parts.append(f"{len(claims)} claim(s)")
                if links:
                    parts.append(f"{len(links)} link(s)")
                emit("extract", "subconscious",
                     f"proposed {', '.join(parts)}",
                     detail={"claims": [p["content"][:60] for p in claims],
                             "links": [f"{p['source'][:8]}--{p['rel']}-->{p['target'][:8]}"
                                       for p in links]})

        except Exception as e:
            emit("extract_error", "subconscious", f"extraction failed: {e}")

    def _parse_extraction(self, text: str) -> list[dict]:
        """Parse Haiku's extraction response into proposal dicts.

        Handles both claim proposals (action: new/update) and link proposals
        (action: link) from the extraction sidecar.
        """
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return []

        try:
            proposals = json.loads(match.group())
            if not isinstance(proposals, list):
                return []

            valid = []
            for p in proposals:
                if not isinstance(p, dict):
                    continue

                action = p.get("action", "new")

                if action == "link":
                    # Link proposal — relationship between existing nodes
                    source = p.get("source", "")
                    target = p.get("target", "")
                    rel = p.get("rel", "relates_to")
                    if source and target and rel:
                        valid_rels = {"relates_to", "caused_by", "depends_on",
                                      "blocks", "enables", "contradicts"}
                        if rel not in valid_rels:
                            rel = "relates_to"
                        valid.append({
                            "type": "link",
                            "source": source,
                            "target": target,
                            "rel": rel,
                            "reason": p.get("reason", ""),
                            "source_type": "extraction_sidecar",
                        })
                elif p.get("content") and p.get("domain"):
                    # Claim proposal — new fact or update
                    claim_scope = p.get("scope", "project")
                    if claim_scope not in ("project", "global"):
                        claim_scope = "project"

                    # Validate any proposed anchors
                    claim_anchors = None
                    anchors_raw = p.get("anchors", [])
                    if isinstance(anchors_raw, list) and anchors_raw:
                        try:
                            from mnemo_verify import validate_anchor
                            va = [a for a in anchors_raw
                                  if isinstance(a, dict) and validate_anchor(a)]
                            if va:
                                claim_anchors = va
                        except Exception:
                            pass

                    conf = float(p.get("confidence", 0.7))
                    if conf < 0.7:
                        continue  # reject low-confidence proposals

                    valid.append({
                        "type": "extraction",
                        "content": p["content"],
                        "domain": p["domain"],
                        "confidence": conf,
                        "action": action,
                        "supersedes": p.get("supersedes", ""),
                        "scope": claim_scope,
                        "anchors": claim_anchors,
                        "source": "extraction_sidecar",
                    })
            return valid
        except (json.JSONDecodeError, ValueError):
            return []

    # ─── Maintenance ──────────────────────────────────────────────────────

    def maybe_maintain(self) -> None:
        """Run programmatic health checks. If issues found, queue Haiku judgment."""
        if self._maintain_thread and self._maintain_thread.is_alive():
            return

        # Programmatic checks (zero tokens)
        issues = self._check_health()
        if not issues:
            return

        self._maintain_thread = threading.Thread(
            target=self._maintain_background,
            args=(issues,),
            daemon=True,
        )
        self._maintain_thread.start()

    def _check_health(self) -> list[dict]:
        """Programmatic health checks — no API calls.

        Checks actual tree health signals rather than raw size:
        1. Domain imbalance — any domain with 4+ nodes could compress
        2. Redundancy — similar nodes that should be superseded
        3. Low utility — recalled often but never acted on
        4. Stale — unreinforced for 14+ days
        5. Low coverage — compress nodes with poor coverage scores
        """
        issues = []
        active = self.store.get_active()

        # Load all active nodes once — single pass for all checks
        all_nodes: list[Node] = []
        domains: dict[str, list] = {}
        for addr in active:
            node = self.store.get(addr)
            if node:
                all_nodes.append(node)
                d = node.meta.get("domain", "uncategorized")
                domains.setdefault(d, []).append(node)

        # 1. Domain clustering — flag domains with many nodes
        for domain, nodes in sorted(domains.items(), key=lambda x: -len(x[1])):
            if len(nodes) >= 4:
                issues.append({
                    "check": "domain_cluster",
                    "domain": domain,
                    "count": len(nodes),
                    "nodes": [{"addr": n.addr, "content": n.content[:100]}
                              for n in nodes],
                })
                break  # one cluster at a time

        # 2. Redundancy check via similarity
        try:
            redundant = propose_supersessions(self.store, threshold=0.5)
            for pair in redundant[:3]:  # cap at 3
                issues.append({
                    "check": "redundancy",
                    "old": pair["old"],
                    "new": pair["new"],
                    "similarity": pair["similarity"],
                    "old_content": pair["old_content"],
                    "new_content": pair["new_content"],
                })
        except Exception:
            pass

        # 3. Low utility — 5+ recalls, 0 hits → likely noise
        for node in all_nodes:
            recall_count = node.meta.get("recall_count", 0)
            recall_hits = node.meta.get("recall_hits", 0)
            if recall_count >= 5 and recall_hits == 0:
                issues.append({
                    "check": "low_utility",
                    "address": node.addr,
                    "recall_count": recall_count,
                    "content": node.content[:100],
                })

        # 4. Stale — unreinforced for 14+ days (cap at 5 worst)
        now = time.time()
        stale_candidates = []
        for node in all_nodes:
            last_fresh = node.meta.get("last_reinforced", node.created)
            days_stale = (now - last_fresh) / 86400
            if days_stale >= 14:
                stale_candidates.append((days_stale, node))
        stale_candidates.sort(key=lambda x: -x[0])  # worst first
        for days_stale, node in stale_candidates[:5]:
            issues.append({
                "check": "stale",
                "address": node.addr,
                "days_stale": int(days_stale),
                "content": node.content[:100],
            })

        # 5. Low coverage — compress nodes with coverage_score < 0.5
        for node in all_nodes:
            if node.type == "compress":
                coverage = node.meta.get("coverage_score")
                if coverage is not None and coverage < 0.5:
                    issues.append({
                        "check": "low_coverage",
                        "address": node.addr,
                        "coverage": coverage,
                        "content": node.content[:100],
                    })

        # 6. Anchor verification — claims contradicted by codebase
        try:
            from mnemo_verify import verify_active, _resolve_project_root
            project_root = _resolve_project_root()
            if project_root:
                failures = verify_active(self.store, project_root)
                for result in failures[:5]:
                    issues.append({
                        "check": "anchor_failed",
                        "address": result["addr"],
                        "content": result["content"][:100],
                        "failed_anchors": result["results"],
                    })
        except Exception:
            pass

        return issues

    def _maintain_background(self, issues: list[dict]) -> None:
        """Send flagged issues to Haiku for judgment (minimal tokens)."""
        try:
            client = self._get_client()
            if client is None:
                return

            # Build a focused prompt — only the flagged nodes, not full context
            lines = ["Flagged issues in the project memory tree:\n"]
            for i, issue in enumerate(issues, 1):
                if issue["check"] == "domain_cluster":
                    lines.append(f"{i}. DOMAIN CLUSTER in '{issue['domain']}' "
                                 f"({issue['count']} nodes):")
                    for n in issue["nodes"]:
                        lines.append(f"   {n['addr'][:8]}: {n['content']}")
                    lines.append("   → Can any of these be compressed together?\n")

                elif issue["check"] == "redundancy":
                    lines.append(f"{i}. SIMILAR NODES (similarity={issue['similarity']:.0%}):")
                    lines.append(f"   {issue['old'][:8]}: {issue['old_content']}")
                    lines.append(f"   {issue['new'][:8]}: {issue['new_content']}")
                    lines.append("   → Should the newer one supersede the older?\n")

                elif issue["check"] == "low_utility":
                    lines.append(f"{i}. LOW UTILITY (recalled {issue['recall_count']}x, "
                                 f"never acted on):")
                    lines.append(f"   {issue['address'][:8]}: {issue['content']}")
                    lines.append("   → Is this still useful, or should it be superseded/removed?\n")

                elif issue["check"] == "stale":
                    lines.append(f"{i}. STALE ({issue['days_stale']}d unreinforced):")
                    lines.append(f"   {issue['address'][:8]}: {issue['content']}")
                    lines.append("   → Is this still current, or should it be updated?\n")

                elif issue["check"] == "low_coverage":
                    lines.append(f"{i}. LOW COVERAGE (coverage={issue['coverage']:.0%}):")
                    lines.append(f"   {issue['address'][:8]}: {issue['content']}")
                    lines.append("   → This compression lost too many details. Re-compress?\n")

                elif issue["check"] == "anchor_failed":
                    lines.append(f"{i}. CONTRADICTED BY CODE:")
                    lines.append(f"   {issue['address'][:8]}: {issue['content']}")
                    for r in issue.get("failed_anchors", []):
                        if not r["passed"]:
                            lines.append(f"   \u2717 {r['detail']}")
                    lines.append("   → Update or remove this claim?\n")

            prompt = "\n".join(lines) + "\nReturn a JSON array of maintenance actions."

            resp = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=MAINTENANCE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            result = resp.content[0].text.strip()
            proposals = self._parse_maintenance(result)

            if proposals:
                with self._lock:
                    self._pending.extend(proposals)
                emit("maintain", "subconscious",
                     f"proposed {len(proposals)} maintenance action(s)",
                     detail={"actions": [p.get("op", "?") for p in proposals]})

        except Exception as e:
            emit("maintain_error", "subconscious", f"maintenance failed: {e}")

    def _parse_maintenance(self, text: str) -> list[dict]:
        """Parse Haiku's maintenance response into proposal dicts."""
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return []

        try:
            actions = json.loads(match.group())
            if not isinstance(actions, list):
                return []

            valid = []
            for a in actions:
                if not isinstance(a, dict):
                    continue
                op = a.get("op", "")
                if op == "keep":
                    continue  # no action needed

                if op == "compress" and a.get("addresses") and a.get("summary"):
                    valid.append({
                        "type": "maintenance",
                        "op": "compress",
                        "addresses": a["addresses"],
                        "summary": a["summary"],
                        "domain": a.get("domain", ""),
                        "source": "maintenance_sidecar",
                    })
                elif op == "supersede" and a.get("old_address") and a.get("new_content"):
                    valid.append({
                        "type": "maintenance",
                        "op": "supersede",
                        "old_address": a["old_address"],
                        "new_content": a["new_content"],
                        "reason": a.get("reason", "maintenance"),
                        "source": "maintenance_sidecar",
                    })
                elif op == "recategorize" and a.get("address") and a.get("new_domain"):
                    valid.append({
                        "type": "maintenance",
                        "op": "recategorize",
                        "address": a["address"],
                        "new_domain": a["new_domain"],
                        "reason": a.get("reason", ""),
                        "source": "maintenance_sidecar",
                    })
            return valid
        except (json.JSONDecodeError, ValueError):
            return []

    # ─── Micro-dream mode ──────────────────────────────────────────────────

    def maybe_micro_dream(self, touched_addrs: list[str]) -> None:
        """Micro-dream: review touched nodes + their local neighborhood.

        Runs after every claim/update/reinforce. Lightweight — Haiku sees
        only the local cluster (~5-15 nodes), not the full tree.
        """
        if not touched_addrs:
            return
        if self._dream_thread and self._dream_thread.is_alive():
            return  # don't stack dreams

        self._dream_thread = threading.Thread(
            target=self._micro_dream_background,
            args=(touched_addrs,),
            daemon=True,
        )
        self._dream_thread.start()

    def _micro_dream_background(self, touched_addrs: list[str]) -> None:
        """Run micro-dream in background thread."""
        try:
            client = self._get_client()
            if client is None:
                return

            # Build local neighborhood: touched nodes + linked + same-domain
            neighborhood: dict[str, Node] = {}
            touched_domains: set[str] = set()

            for addr in touched_addrs:
                node = self.store.get(addr)
                if not node:
                    continue
                neighborhood[node.addr] = node
                touched_domains.add(node.meta.get("domain", ""))

                # Follow links outward
                for link in node.meta.get("links", []):
                    linked = self.store.get(link.get("addr", ""))
                    if linked:
                        neighborhood[linked.addr] = linked

                # Reverse links — what points TO this node
                for rl in self.store.get_reverse_links(node.addr):
                    src = self.store.get(rl.get("source_addr", ""))
                    if src:
                        neighborhood[src.addr] = src

            # Add same-domain neighbors (cap at 5 per domain)
            active = self.store.get_active()
            for domain in touched_domains:
                if not domain:
                    continue
                count = 0
                for addr in active:
                    if addr in neighborhood:
                        continue
                    node = self.store.get(addr)
                    if node and node.meta.get("domain") == domain:
                        neighborhood[node.addr] = node
                        count += 1
                        if count >= 5:
                            break

            if len(neighborhood) < 2:
                return  # nothing to compare against

            # Build compact context
            lines = []
            for node in sorted(neighborhood.values(),
                               key=lambda n: n.addr in touched_addrs,
                               reverse=True):
                marker = " [JUST TOUCHED]" if node.addr in touched_addrs else ""
                domain = node.meta.get("domain", "?")
                anchors = node.meta.get("anchors", [])
                anchor_tag = f" [{len(anchors)} anchors]" if anchors else ""
                links = node.meta.get("links", [])
                link_tag = ""
                if links:
                    link_strs = [f"--{l['rel']}-->{l['addr'][:8]}"
                                 for l in links[:3]]
                    link_tag = f" links: {', '.join(link_strs)}"
                lines.append(
                    f"  {node.addr[:8]} [{domain}]{marker}{anchor_tag}: "
                    f"{node.content[:120]}{link_tag}"
                )

            # Run anchor verification on touched nodes
            anchor_results = []
            try:
                from mnemo_verify import verify_node, _resolve_project_root
                project_root = _resolve_project_root()
                if project_root:
                    for addr in touched_addrs:
                        node = neighborhood.get(addr)
                        if node and node.meta.get("anchors"):
                            result = verify_node(node, project_root)
                            if result["failed"] > 0:
                                for r in result["results"]:
                                    if not r["passed"]:
                                        anchor_results.append(
                                            f"  \u2717 {node.addr[:8]}: {r['detail']}")
            except Exception:
                pass

            prompt_parts = [
                f"Local neighborhood ({len(neighborhood)} nodes) after a change:\n"
                + "\n".join(lines),
            ]
            if anchor_results:
                prompt_parts.append(
                    "Anchor verification failures:\n" + "\n".join(anchor_results)
                )
            prompt_parts.append(
                "Review this local cluster. Check for:\n"
                "- Conflicts or contradictions with the just-touched node(s)\n"
                "- Redundancy — does the new claim duplicate a neighbor?\n"
                "- Stale neighbors that the new info might supersede\n"
                "- Missing anchors on verifiable claims\n"
                "Return a JSON array of findings, or [] if the cluster looks healthy."
            )

            resp = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=MICRO_DREAM_SYSTEM,
                messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            )

            result = resp.content[0].text.strip()
            findings = self._parse_dream(result)

            if findings:
                proposals = []
                for f in findings:
                    if f.get("op", "none") == "none" and not f.get("suggestion"):
                        continue
                    proposals.append({
                        "type": "dream",
                        "finding": f["finding"],
                        "suggestion": f.get("suggestion", ""),
                        "op": f.get("op", "none"),
                        "addresses": f.get("addresses", []),
                        "detail": f.get("detail", {}),
                        "source": "micro_dream",
                    })

                if proposals:
                    with self._lock:
                        self._pending.extend(proposals)
                    emit("micro_dream", "subconscious",
                         f"{len(proposals)} finding(s) from local review",
                         addresses=touched_addrs[:5],
                         detail={"neighborhood_size": len(neighborhood),
                                 "findings": [p["finding"][:60] for p in proposals]})

        except Exception as e:
            emit("micro_dream_error", "subconscious", f"micro-dream failed: {e}")

    # ─── Code binding (edit-triggered) ────────────────────────────────────

    _BINDING_SYSTEM = """You are a code comprehension agent. A file was just edited.
Analyze the changed/written code and produce comprehension nodes that will help
future AI instances understand this code immediately — without re-reading it.

Capture:
- What this code does (functional purpose, specific)
- Why it exists (use session context to infer design rationale)
- Key invariants or constraints
- Non-obvious dependencies or side effects

Be specific — name the actual function/class. Write for a future instance with zero context.

Return a JSON array:
[{
  "content": "Specific, self-contained comprehension text.",
  "domain": "architecture|patterns|issues|decisions",
  "confidence": 0.0-1.0,
  "scope_start": <line number of the section this describes, or 0 for file-level>
}]

Return [] if the edit is trivial (formatting, comments, minor variable rename)."""

    def trigger_code_binding(self, filepath: str, code_content: str,
                             log_path: Optional[str] = None) -> None:
        """Queue a comprehension binding for code that was just written/edited.

        Fires from the PostToolUse hook after Edit/Write. Reads recent session
        log for context on WHY the code was written, then proposes a
        content-hash-bound comprehension node via Haiku in a background thread.
        """
        if self._extract_thread and self._extract_thread.is_alive():
            # Don't stack — binding will happen on next trigger
            return

        self._extract_thread = threading.Thread(
            target=self._binding_background,
            args=(filepath, code_content, log_path),
            daemon=True,
        )
        self._extract_thread.start()

    def _binding_background(self, filepath: str, code_content: str,
                            log_path: Optional[str]) -> None:
        """Run code binding in a background thread."""
        try:
            client = self._get_client()
            if client is None:
                return

            # Tail session log for WHY context
            session_ctx = ""
            effective_log = log_path or _get_log_path()
            if effective_log and os.path.exists(effective_log):
                try:
                    with open(effective_log, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    recent = lines[-30:] if len(lines) > 30 else lines
                    events = []
                    for line in recent:
                        try:
                            ev = json.loads(line.strip())
                            if ev.get("event") in ("recall", "status", "extract",
                                                    "extract_error", "maintain"):
                                continue
                            summary = ev.get("summary", "")
                            if summary:
                                events.append(f"  [{ev.get('event', '?')}] {summary[:100]}")
                        except json.JSONDecodeError:
                            continue
                    session_ctx = "\n".join(events)
                except Exception:
                    pass

            # Detect sections in the written code
            from mnemo_map import _detect_sections
            import os as _os
            ext = _os.path.splitext(filepath)[1]
            lines_list = code_content.splitlines()
            sections = _detect_sections(lines_list, ext)

            if not sections:
                # File-level binding for files with no detectable sections
                sections = [{"line_num": 1, "scope": "module",
                             "context_lines": code_content[:500]}]

            sections_text = "\n\n".join(
                f"Section [{s['scope']}] (line {s['line_num']}):\n{s['context_lines']}"
                for s in sections[:10]  # cap at 10 sections per edit
            )

            prompt_parts = [f"File just edited: {filepath}\n\nCode:\n{sections_text}"]
            if session_ctx:
                prompt_parts.append(
                    f"Recent session context (what was happening, WHY this was written):\n"
                    f"{session_ctx}"
                )
            prompt_parts.append("Produce comprehension nodes for this code.")

            resp = client.messages.create(
                model=MODEL,
                max_tokens=800,
                system=self._BINDING_SYSTEM,
                messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            )

            text = resp.content[0].text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                return

            proposals = json.loads(match.group())
            if not isinstance(proposals, list) or not proposals:
                return

            from mnemo_anchor import compute_content_hash, update_file_index
            from mnemo_verify import _resolve_project_root
            import os as _os2
            project_root = _resolve_project_root()
            try:
                rel_path = str(
                    _os2.path.relpath(filepath, str(project_root))
                ).replace("\\", "/")
            except ValueError:
                rel_path = _os2.path.basename(filepath)

            active = self.store.get_active()
            created = []

            for proposal in proposals:
                if not isinstance(proposal, dict) or not proposal.get("content"):
                    continue
                if float(proposal.get("confidence", 0.7)) < 0.65:
                    continue

                scope_start = int(proposal.get("scope_start", 0))
                # Find matching section by line proximity
                section = sections[0]
                if scope_start > 0:
                    closest = min(sections,
                                  key=lambda s: abs(s["line_num"] - scope_start))
                    section = closest

                content_hash = compute_content_hash(section["context_lines"])
                anchor = {
                    "type": "content_hash",
                    "file": rel_path,
                    "content_hash": content_hash,
                    "context_lines": section["context_lines"],
                    "scope": section["scope"],
                    "line_hint": section["line_num"],
                }

                from mnemo import Node as _Node
                node = _Node(
                    type="leaf",
                    content=proposal["content"],
                    meta={
                        "domain": proposal.get("domain", "architecture"),
                        "confidence": float(proposal.get("confidence", 0.8)),
                        "source": "binding",
                        "anchors": [anchor],
                        "bound_file": rel_path,
                        "bound_line": section["line_num"],
                    },
                )

                self.store.put(node)
                active.add(node.addr)
                update_file_index(self.store, node)
                created.append(node.addr)

            if created:
                self.store.set_active(active)
                emit("binding", "subconscious",
                     f"bound {len(created)} comprehension node(s) to {rel_path}",
                     addresses=created,
                     detail={"file": rel_path, "nodes": len(created)})

        except Exception as e:
            emit("binding_error", "subconscious", f"code binding failed: {e}")

    # ─── Dream mode (full) ────────────────────────────────────────────────

    def maybe_dream(self) -> None:
        """Run dream review in background if not already running.

        Called periodically from memory_recall. Findings queue as proposals
        and surface on the next recall alongside extraction and maintenance.
        """
        if self._dream_thread and self._dream_thread.is_alive():
            return

        self._dream_thread = threading.Thread(
            target=self._dream_background,
            daemon=True,
        )
        self._dream_thread.start()

    def _dream_background(self) -> None:
        """Run dream in background thread, convert findings to proposals."""
        try:
            findings = self.dream()
            if not findings:
                return

            # Convert dream findings into proposals that surface via recall
            proposals = []
            for f in findings:
                # Skip non-actionable findings
                if f.get("op", "none") == "none" and not f.get("suggestion"):
                    continue
                proposals.append({
                    "type": "dream",
                    "finding": f["finding"],
                    "suggestion": f.get("suggestion", ""),
                    "op": f.get("op", "none"),
                    "addresses": f.get("addresses", []),
                    "detail": f.get("detail", {}),
                })

            if proposals:
                with self._lock:
                    self._pending.extend(proposals)
                emit("dream", "system",
                     f"background dream: {len(proposals)} finding(s) queued",
                     detail={"findings": [p["finding"][:60] for p in proposals]})

        except Exception as e:
            emit("dream_error", "system", f"background dream failed: {e}")

    def dream(self) -> list[dict]:
        """Full tree review — returns human-readable findings.

        Can be called synchronously (via memory_dream tool) or in background
        (via maybe_dream). Findings describe what needs attention.
        """
        client = self._get_client()
        if client is None:
            return [{"finding": "Anthropic SDK not available — can't run dream mode.",
                     "suggestion": "Install the anthropic package.",
                     "op": "none", "addresses": [], "detail": {}}]

        active_ctx = build_active_context(self.store, max_nodes=50)
        if not active_ctx:
            return [{"finding": "Memory is empty — nothing to review.",
                     "suggestion": "Start capturing project knowledge.",
                     "op": "none", "addresses": [], "detail": {}}]

        # Include basic stats for context
        active = self.store.get_active()
        ctx_len = len(active_ctx)
        domains: dict[str, int] = {}
        for addr in active:
            node = self.store.get(addr)
            if node:
                d = node.meta.get("domain", "uncategorized")
                domains[d] = domains.get(d, 0) + 1

        stats = (
            f"Tree stats: {len(active)} active nodes, {ctx_len} chars, "
            f"pressure {'HIGH' if ctx_len > SIDECAR_CONTEXT_CAP * 0.7 else 'ok'} "
            f"({ctx_len}/{SIDECAR_CONTEXT_CAP} chars).\n"
            f"Domain breakdown: {', '.join(f'{d}={c}' for d, c in sorted(domains.items()))}\n\n"
        )

        prompt = (
            f"{stats}"
            f"Full active memory:\n{active_ctx}\n\n"
            f"Review this knowledge tree. What needs attention?\n"
            f"Return a JSON array of findings."
        )

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=DREAM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            result = resp.content[0].text.strip()
            return self._parse_dream(result)

        except Exception as e:
            emit("dream_error", "system", f"dream failed: {e}")
            return [{"finding": f"Dream mode encountered an error: {e}",
                     "suggestion": "Try again or check API key.",
                     "op": "none", "addresses": [], "detail": {}}]

    def _parse_dream(self, text: str) -> list[dict]:
        """Parse dream response into human-readable findings.

        Handles truncated JSON (max_tokens cutoff) by attempting to
        close the array and parse what we got.
        """
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            # Try to salvage truncated response — find start of array
            start = text.find('[')
            if start == -1:
                return []
            # Take everything from [ onward, try to close it
            fragment = text[start:]
            # Find the last complete object (ends with })
            last_brace = fragment.rfind('}')
            if last_brace == -1:
                return []
            fragment = fragment[:last_brace + 1] + ']'
            try:
                findings = json.loads(fragment)
            except json.JSONDecodeError:
                return []
        else:
            try:
                findings = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(findings, list):
            return []

        # Filter findings: only keep those whose referenced addresses
        # are still in the active set (dream runs async, nodes may have
        # been superseded between dream time and display time)
        active = self.store.get_active()

        valid = []
        for f in findings:
            if isinstance(f, dict) and f.get("finding"):
                addrs = f.get("addresses", [])
                # Skip findings that reference only superseded nodes
                if addrs and not any(a in active for a in addrs):
                    continue
                valid.append({
                    "finding": f["finding"],
                    "suggestion": f.get("suggestion", ""),
                    "op": f.get("op", "none"),
                    "addresses": [a for a in addrs if a in active],
                    "detail": f.get("detail", {}),
                })
        return valid

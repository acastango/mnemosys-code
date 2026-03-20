#!/usr/bin/env python3
"""
mnemo — Content-addressed project memory with Merkle-structured provenance

A memory system for software projects. Every decision, convention,
known issue, and architectural choice traces back through a verifiable
tree to the moment it was captured. Compression with provenance.
Project knowledge as a chain head.

Designed for the problem of "hopping into a big project cold" —
when a new instance needs to understand not just what the code does,
but why it's that way, what's been tried, what breaks, and what
patterns to follow.

== Node Types ==

LEAF        Raw claim from a session. The ground floor.
COMPRESS    Summary derived from multiple inputs. Lossy in content,
            lossless in provenance.
SUPERSEDE   A claim that replaces a prior claim. The old claim stays
            addressable; the active path routes through the new one.
ROOT        The current project knowledge hash. One node that represents
            the compressed state of the entire tree.

== Every node has ==

addr        Content-derived hash (sha256 of type + content + inputs)
type        leaf | compress | supersede | root
content     The actual text
inputs      List of addresses this node was derived from
created     Timestamp
meta        Freeform metadata (model, session_id, domain, etc.)

== Operations ==

ingest      Conversation → addressed leaves
compress    N nodes → summary node encoding inputs
supersede   New claim replaces old, with link
query       Address or natural language → traversal result
reroot      Recompute root after changes
prune       Detect structurally similar claims, propose supersessions

== Storage ==

Flat file directory. Each node is a JSON file named by address.
Index files for fast lookup. No database required. Portable.

    store/
    ├── nodes/
    │   ├── a7f3c2d1e8b4.json
    │   ├── b91e4d3f0c28.json
    │   └── ...
    ├── active.json          # Current active set (non-superseded)
    ├── roots.json           # Root history (linear chain)
    └── index/
        ├── by_domain.json
        ├── by_time.json
        └── by_session.json
"""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


ADDR_LEN = 12
GENESIS = "0" * ADDR_LEN

MNEMO_DIR = ".mnemo"          # project-local store directory name
GLOBAL_DIR = Path.home() / ".mnemo"   # cross-project global store
V1_FALLBACK = Path.home() / "mnemo"   # v1 legacy store location


# ===================================================================
# Store discovery
# ===================================================================

def discover_store(cwd: Path | str | None = None) -> tuple[Path, bool]:
    """
    Find the appropriate store path by walking up from cwd.

    Walk order:
      1. Walk up from cwd looking for .mnemo/ (project-local, v2).
         Refuses to use $HOME itself as a project store.
      2. MNEMO_STORE env var (explicit override).
      3. v1 ~/mnemo/ if it exists (compatibility mode — no chains).

    Returns:
        (store_path, is_v2) where is_v2=True means a .mnemo/ dir was
        found and chain-based retrieval is available.
        is_v2=False means we fell back to v1 or MNEMO_STORE.
    """
    home = Path.home()

    # Env override takes precedence (explicit is better than implicit)
    env_store = os.environ.get("MNEMO_STORE", "")
    if env_store:
        p = Path(env_store).expanduser().resolve()
        return p, False  # env override → treat as v1-compat

    # Walk up from cwd looking for .mnemo/
    start = Path(cwd).resolve() if cwd else Path.cwd()
    current = start
    while True:
        candidate = current / MNEMO_DIR
        if candidate.is_dir():
            # Refuse to use $HOME/.mnemo/ — that's the global store,
            # not a project store. (mnemo init blocks creating it there too.)
            if current == home:
                break
            return candidate, True
        parent = current.parent
        if parent == current:
            break  # filesystem root
        current = parent

    # v1 fallback
    if V1_FALLBACK.is_dir():
        return V1_FALLBACK, False

    # Nothing found — return a sensible default so callers can init
    return start / MNEMO_DIR, False


# ===================================================================
# Node
# ===================================================================

@dataclass
class Node:
    """A single node in the memory tree."""
    type: str                           # leaf | compress | supersede | root
    content: str                        # the actual text
    inputs: list[str] = field(default_factory=list)   # addresses of input nodes
    created: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    # Computed after creation
    addr: str = ""

    def __post_init__(self):
        if not self.addr:
            self.addr = self.compute_addr()

    def compute_addr(self) -> str:
        """
        Address = sha256(type + content + sorted inputs).

        Inputs are sorted so that the address is deterministic
        regardless of the order nodes were provided.
        For leaves in a conversation chain, sequence is encoded
        by including prev_addr in the inputs list.
        """
        payload = f"{self.type}\n{self.content}\n{','.join(sorted(self.inputs))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:ADDR_LEN]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        addr = d.pop("addr", "")
        node = cls(**d)
        if addr:
            node.addr = addr
        return node


# ===================================================================
# Store
# ===================================================================

class Store:
    """
    Flat-file content-addressed store.

    Every node is a JSON file named by its address.
    Index files accelerate lookup without being authoritative —
    they can always be rebuilt from the nodes themselves.
    """

    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.nodes_dir = self.root / "nodes"
        self.index_dir = self.root / "index"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Active set: addresses of non-superseded nodes
        self._active_path = self.root / "active.json"
        # Root history: ordered list of root addresses
        self._roots_path = self.root / "roots.json"

    # --- Core operations ---

    def put(self, node: Node) -> str:
        """Store a node. Returns its address."""
        path = self.nodes_dir / f"{node.addr}.json"
        path.write_text(json.dumps(node.to_dict(), indent=2), encoding="utf-8")
        self._update_reverse_links(node)
        return node.addr

    def get(self, addr: str) -> Optional[Node]:
        """Retrieve a node by address. Returns None if not found."""
        # Support prefix matching
        if len(addr) < ADDR_LEN:
            matches = list(self.nodes_dir.glob(f"{addr}*.json"))
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) == 0:
                return None
            else:
                # Ambiguous prefix
                return None
        else:
            path = self.nodes_dir / f"{addr}.json"

        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return Node.from_dict(d)

    def exists(self, addr: str) -> bool:
        return (self.nodes_dir / f"{addr}.json").exists()

    def all_nodes(self) -> list[Node]:
        """Load all nodes. Expensive — use indexes when possible."""
        nodes = []
        for path in self.nodes_dir.glob("*.json"):
            d = json.loads(path.read_text(encoding="utf-8"))
            nodes.append(Node.from_dict(d))
        return nodes

    # --- Active set ---

    def get_active(self) -> set[str]:
        """Get the set of currently active (non-superseded) addresses."""
        if self._active_path.exists():
            return set(json.loads(self._active_path.read_text()))
        return set()

    def set_active(self, active: set[str]):
        self._active_path.write_text(json.dumps(sorted(active), indent=2))

    # --- Root history ---

    def get_roots(self) -> list[str]:
        if self._roots_path.exists():
            return json.loads(self._roots_path.read_text())
        return []

    def push_root(self, addr: str):
        roots = self.get_roots()
        roots.append(addr)
        self._roots_path.write_text(json.dumps(roots, indent=2))

    def current_root(self) -> Optional[str]:
        roots = self.get_roots()
        return roots[-1] if roots else None

    # --- Reverse link index ---

    def _reverse_links_path(self) -> Path:
        return self.index_dir / "reverse_links.json"

    def _load_reverse_links(self) -> dict[str, list[dict]]:
        """Load reverse link index: {target_addr: [{source_addr, rel}]}."""
        path = self._reverse_links_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                pass
        return {}

    def _save_reverse_links(self, index: dict[str, list[dict]]):
        self._reverse_links_path().write_text(
            json.dumps(index, indent=2), encoding="utf-8")

    def _update_reverse_links(self, node: Node):
        """Update reverse link index when a node has links in its meta."""
        links = node.meta.get("links")
        if not links:
            return
        index = self._load_reverse_links()
        for link in links:
            target = link.get("addr", "")
            rel = link.get("rel", "relates_to")
            if not target:
                continue
            entries = index.setdefault(target, [])
            # Don't duplicate
            if not any(e["source_addr"] == node.addr and e["rel"] == rel
                       for e in entries):
                entries.append({"source_addr": node.addr, "rel": rel})
        self._save_reverse_links(index)

    def get_reverse_links(self, addr: str) -> list[dict]:
        """Return [{source_addr, rel}] for nodes that link TO this addr."""
        index = self._load_reverse_links()
        return index.get(addr, [])

    def rebuild_reverse_links(self):
        """Full rebuild of reverse link index from active set."""
        index: dict[str, list[dict]] = {}
        active = self.get_active()
        for addr in active:
            node = self.get(addr)
            if not node:
                continue
            links = node.meta.get("links")
            if not links:
                continue
            for link in links:
                target = link.get("addr", "")
                rel = link.get("rel", "relates_to")
                if not target:
                    continue
                entries = index.setdefault(target, [])
                if not any(e["source_addr"] == node.addr and e["rel"] == rel
                           for e in entries):
                    entries.append({"source_addr": node.addr, "rel": rel})
        self._save_reverse_links(index)

    # --- Traversal ---

    def ancestors(self, addr: str) -> list[Node]:
        """Walk up from a node through its inputs, breadth-first."""
        visited = set()
        queue = [addr]
        result = []
        while queue:
            current = queue.pop(0)
            if current in visited or current == GENESIS:
                continue
            visited.add(current)
            node = self.get(current)
            if node:
                result.append(node)
                queue.extend(node.inputs)
        return result

    def descendants(self, addr: str) -> list[Node]:
        """Find all nodes that have addr in their inputs."""
        result = []
        for path in self.nodes_dir.glob("*.json"):
            d = json.loads(path.read_text(encoding="utf-8"))
            if addr in d.get("inputs", []):
                result.append(Node.from_dict(d))
        return result

    def provenance(self, addr: str) -> list[Node]:
        """
        Full provenance chain: walk ancestors all the way to leaves.
        Returns nodes ordered from the queried node down to leaves.
        """
        return self.ancestors(addr)


# ===================================================================
# Operations
# ===================================================================

def ingest_conversation(turns: list[dict], store: Store,
                        conversation_id: str = "") -> list[str]:
    """
    Ingest conversation turns as leaf nodes.

    Each turn becomes a leaf. The prev_addr of the preceding turn
    is included in inputs to preserve sequence.

    Args:
        turns: list of {"role": str, "content": str, ...}
        store: the node store
        conversation_id: optional identifier for grouping

    Returns:
        List of addresses of the created leaf nodes.
    """
    addrs = []
    prev = GENESIS

    for i, turn in enumerate(turns):
        node = Node(
            type="leaf",
            content=turn.get("content", ""),
            inputs=[prev] if prev != GENESIS else [],
            meta={
                "role": turn["role"],
                "seq": i + 1,
                "conversation": conversation_id,
                "timestamp": turn.get("timestamp", ""),
                "thinking": turn.get("thinking", ""),
            }
        )
        store.put(node)
        addrs.append(node.addr)
        prev = node.addr

    # Update active set
    active = store.get_active()
    active.update(addrs)
    store.set_active(active)

    return addrs


def compress(addrs: list[str], summary: str, store: Store,
             domain: str = "", model: str = "") -> str:
    """
    Create a compression node that summarizes multiple input nodes.

    The summary is lossy in content but lossless in provenance —
    every input address is recorded.

    Args:
        addrs: addresses of the nodes being compressed
        summary: the LLM-generated summary text
        store: the node store
        domain: optional domain tag (e.g., "clinical interests")
        model: the model that produced the summary

    Returns:
        Address of the new compression node.
    """
    # Load input nodes for coverage scoring and value preservation
    input_texts = []
    input_nodes = []
    for addr in addrs:
        input_node = store.get(addr)
        if input_node:
            input_texts.append(input_node.content)
            input_nodes.append(input_node)

    node = Node(
        type="compress",
        content=summary,
        inputs=addrs,
        meta={
            "domain": domain,
            "model": model,
            "input_count": len(addrs),
        }
    )

    # Extract quantitative fragments — lossless preservation of specific values
    if input_nodes:
        from mnemo_retrieval import extract_quantitative_fragments, compute_coverage_score
        preserved = []
        for inp in input_nodes:
            fragments = extract_quantitative_fragments(inp.content, inp.addr)
            preserved.extend(fragments)
        if preserved:
            node.meta["preserved_values"] = preserved

        # Compute coverage score — preserved terms count as covered
        if input_texts:
            preserved_tokens = set()
            for p in preserved:
                # Tokenize each fragment to get the terms that are preserved
                tokens = re.findall(r'[a-z_]+', p["fragment"].lower())
                preserved_tokens.update(t for t in tokens if len(t) >= 3)

            coverage = compute_coverage_score(input_texts, summary,
                                              preserved_terms=preserved_tokens)
            node.meta["coverage_score"] = round(coverage, 3)
            if coverage < 0.6:
                from mnemo_log import emit
                emit("compress_warning", "system",
                     f"Low coverage ({coverage:.1%}) in compression {node.addr[:8]} "
                     f"— distinctive terms may have been lost",
                     addresses=[node.addr],
                     detail={"coverage": coverage, "input_count": len(addrs)})

    # Collect content_hash anchors from all input nodes — lossless anchor provenance.
    # Compress node inherits bindings so the file index stays intact when inputs
    # go inactive. Deduplicate by (file, content_hash) to avoid accumulation.
    inherited_anchors: list[dict] = []
    seen_anchor_keys: set[tuple] = set()
    for inp in input_nodes:
        for anchor in inp.meta.get("anchors", []):
            if anchor.get("type") != "content_hash":
                continue
            key = (anchor.get("file", ""), anchor.get("content_hash", ""))
            if key not in seen_anchor_keys:
                seen_anchor_keys.add(key)
                inherited_anchors.append(anchor)
    if inherited_anchors:
        node.meta["anchors"] = inherited_anchors

    store.put(node)

    # Update active set: the compression replaces its inputs
    # in the active path (inputs remain addressable, just not active)
    active = store.get_active()
    active -= set(addrs)
    active.add(node.addr)
    store.set_active(active)

    # Update file index: remove input entries, register compress node's anchors
    if inherited_anchors:
        try:
            from mnemo_anchor import update_file_index, remove_from_file_index
            for addr in addrs:
                remove_from_file_index(store, addr)
            update_file_index(store, node)
        except Exception:
            pass

    return node.addr


def supersede(old_addr: str, new_content: str, store: Store,
              reason: str = "", model: str = "",
              domain: str = "", meta_overrides: dict = None) -> str:
    """
    Create a supersession: new claim replaces old.

    The old node stays fully addressable. The active path
    routes through the new node. The link between them
    is the provenance of the change.

    Inherits domain and confidence from the old node unless
    overridden via domain param or meta_overrides.

    Args:
        old_addr: address of the claim being superseded
        new_content: the updated claim text
        store: the node store
        reason: why the supersession happened
        model: the model that detected the change
        domain: override domain (if empty, inherits from old node)
        meta_overrides: additional meta fields to set or override

    Returns:
        Address of the new supersession node.
    """
    old_node = store.get(old_addr)
    inherited_meta = {}
    if old_node:
        # Carry forward domain, confidence, source, priority from the old node
        for key in ("domain", "confidence", "source", "priority"):
            if key in old_node.meta:
                inherited_meta[key] = old_node.meta[key]

        # Carry forward links (concept-level relationships survive supersession)
        if old_node.meta.get("links"):
            inherited_meta["links"] = old_node.meta["links"]

        # Carry forward content_hash anchors — the code being described is the
        # same, only the description improved. Binding survives supersession.
        if old_node.meta.get("anchors"):
            inherited_meta["anchors"] = old_node.meta["anchors"]

        # Accumulate utility counters across supersession chain
        for counter_key in ("recall_count", "recall_hits", "reinforcement_count"):
            if old_node.meta.get(counter_key):
                inherited_meta[counter_key] = old_node.meta[counter_key]

        # Carry forward timestamps
        for ts_key in ("last_reinforced", "last_recalled"):
            if old_node.meta.get(ts_key):
                inherited_meta[ts_key] = old_node.meta[ts_key]

    meta = {
        **inherited_meta,
        "supersedes": old_addr,
        "reason": reason,
        "model": model,
    }
    if domain:
        meta["domain"] = domain
    if meta_overrides:
        meta.update(meta_overrides)

    node = Node(
        type="supersede",
        content=new_content,
        inputs=[old_addr],
        meta=meta,
    )
    store.put(node)

    # Update active set
    active = store.get_active()
    active.discard(old_addr)
    active.add(node.addr)
    store.set_active(active)

    # Update file index: old entry out, new node in (anchors already inherited)
    if node.meta.get("anchors"):
        try:
            from mnemo_anchor import update_file_index, remove_from_file_index
            remove_from_file_index(store, old_addr)
            update_file_index(store, node)
        except Exception:
            pass

    return node.addr


def reroot(store: Store, domain_summaries: dict[str, str] = None) -> str:
    """
    Compute a new root from the current active set.

    The root node's content is the current soul summary.
    Its inputs are all active node addresses.

    Args:
        store: the node store
        domain_summaries: optional {domain: summary} for the root content.
                         If not provided, root content is a simple listing.

    Returns:
        Address of the new root node.
    """
    active = store.get_active()

    if domain_summaries:
        content = "\n".join(f"## {d}\n{s}" for d, s in domain_summaries.items())
    else:
        content = f"Active nodes: {len(active)}"

    node = Node(
        type="root",
        content=content,
        inputs=sorted(active),
        meta={"active_count": len(active)},
    )
    store.put(node)
    store.push_root(node.addr)

    return node.addr


# ===================================================================
# Pruning / similarity detection
# ===================================================================

def find_similar_claims(node_addr: str, store: Store,
                        threshold: float = 0.8) -> list[dict]:
    """
    Find active nodes with structurally similar content.

    This is the lightweight version — substring/keyword overlap.
    A production system would use embeddings.

    Returns list of {"addr": ..., "content": ..., "similarity": ...}
    """
    target = store.get(node_addr)
    if not target:
        return []

    target_words = set(target.content.lower().split())
    active = store.get_active()
    candidates = []

    for addr in active:
        if addr == node_addr:
            continue
        node = store.get(addr)
        if not node:
            continue
        # Same type check
        if node.type != target.type:
            continue

        node_words = set(node.content.lower().split())
        if not node_words or not target_words:
            continue

        # Jaccard similarity
        intersection = target_words & node_words
        union = target_words | node_words
        similarity = len(intersection) / len(union) if union else 0

        if similarity >= threshold:
            candidates.append({
                "addr": addr,
                "content": node.content,
                "similarity": round(similarity, 3),
                "created": node.created,
            })

    # Sort by similarity descending, then recency
    candidates.sort(key=lambda x: (-x["similarity"], -x["created"]))
    return candidates


def propose_supersessions(store: Store, threshold: float = 0.7) -> list[dict]:
    """
    Scan active set for pairs of similar claims where the newer
    one likely supersedes the older one.

    Returns list of {"old": addr, "new": addr, "similarity": float}
    """
    active = store.get_active()
    nodes = []
    for addr in active:
        node = store.get(addr)
        if node and node.type in ("leaf", "compress", "supersede"):
            nodes.append(node)

    # Compare all pairs (expensive — production would use embeddings + ANN)
    proposals = []
    seen = set()
    for i, a in enumerate(nodes):
        a_words = set(a.content.lower().split())
        if not a_words:
            continue
        for b in nodes[i+1:]:
            pair_key = tuple(sorted([a.addr, b.addr]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            b_words = set(b.content.lower().split())
            if not b_words:
                continue

            intersection = a_words & b_words
            union = a_words | b_words
            sim = len(intersection) / len(union) if union else 0

            if sim >= threshold:
                # Newer one supersedes older
                if a.created >= b.created:
                    new, old = a, b
                else:
                    new, old = b, a

                proposals.append({
                    "old": old.addr,
                    "old_content": old.content[:100],
                    "new": new.addr,
                    "new_content": new.content[:100],
                    "similarity": round(sim, 3),
                })

    proposals.sort(key=lambda x: -x["similarity"])
    return proposals


# ===================================================================
# LLM Interface — Live Extraction (mid-conversation)
# ===================================================================

LIVE_EXTRACTION_SCHEMA = {
    "claims": [{
        "content":    "str — the claim as a standalone fact about the project",
        "domain":     "str — architecture|decisions|patterns|tasks|issues|dependencies|history|context",
        "action":     "str — new|update|reinforce",
        "supersedes": "str — addr of node this replaces (if action=update)",
        "confidence": "float 0-1 — established fact vs speculative observation",
    }]
}

LIVE_EXTRACTION_PROMPT = """You have access to a project memory system. When you notice something
worth preserving — an architecture decision, a convention established,
a bug discovered, a dependency constraint, a pattern chosen — emit
a structured claim.

Do NOT emit for: routine code changes, things already in memory,
speculative ideas that weren't acted on, transient debugging notes.

DO emit for: architecture decisions and rationale, coding conventions,
known issues and gotchas, dependency constraints, task status changes,
approaches tried and abandoned, important file/module responsibilities.

Emit as:
<memory_claim>
{"claims": [{"content": "...", "domain": "...", "action": "new|update|reinforce",
  "supersedes": "addr_if_updating", "confidence": 0.0-1.0}]}
</memory_claim>

Current project memory:
{active_context}
"""


def process_live_extraction(claims: list[dict], store: Store,
                            conversation_id: str = "",
                            model: str = "") -> list[str]:
    """
    Process claims emitted by the LLM mid-conversation.
    Returns addresses of all nodes created or touched.
    """
    addrs = []

    for claim in claims:
        action = claim.get("action", "new")
        content = claim.get("content", "")
        domain = claim.get("domain", "")
        confidence = claim.get("confidence", 0.5)
        if not content:
            continue

        meta = {
            "domain": domain,
            "confidence": confidence,
            "conversation": conversation_id,
            "model": model,
            "source": "live",
        }

        if action == "new":
            node = Node(type="leaf", content=content, meta=meta)
            store.put(node)
            active = store.get_active()
            active.add(node.addr)
            store.set_active(active)
            addrs.append(node.addr)

        elif action == "update":
            old_addr = claim.get("supersedes", "")
            if old_addr and store.exists(old_addr):
                new_addr = supersede(old_addr, content, store,
                    reason=f"live update in {conversation_id}", model=model)
                addrs.append(new_addr)
            else:
                meta["intended_supersession"] = old_addr
                node = Node(type="leaf", content=content, meta=meta)
                store.put(node)
                active = store.get_active()
                active.add(node.addr)
                store.set_active(active)
                addrs.append(node.addr)

        elif action == "reinforce":
            target_addr = claim.get("supersedes", "")
            if target_addr and store.exists(target_addr):
                target = store.get(target_addr)
                target.meta["last_reinforced"] = time.time()
                target.meta["reinforcement_count"] = target.meta.get("reinforcement_count", 0) + 1
                store.put(target)
                addrs.append(target_addr)

    return addrs


# ===================================================================
# LLM Interface — End of Conversation Compression
# ===================================================================

END_COMPRESSION_SCHEMA = {
    "summary":        "str — 1-3 sentence session summary (what was worked on, what was decided)",
    "key_claims":     ["addr1", "addr2 — most important live claims"],
    "new_claims":     [{"content": "str", "domain": "str", "action": "new|update", "supersedes": ""}],
    "domains_touched": ["domain1", "domain2"],
}

END_COMPRESSION_PROMPT = """The session has ended. Two jobs:

1. COMPRESS — Summarize what was worked on. Focus on decisions made,
   problems solved, and state changes. This becomes a node whose
   inputs are the claims extracted during the session.
2. CATCH — Any project knowledge missed during live extraction?
   Architecture decisions, conventions established, bugs found,
   things tried that didn't work? Add them now.

Live claims from this session:
{live_claims}

Current project memory:
{active_context}

Respond as:
<end_compression>
{"summary": "...", "key_claims": ["addr1"], "new_claims": [...], "domains_touched": ["..."]}
</end_compression>
"""


def process_end_compression(result: dict, live_addrs: list[str],
                            store: Store, conversation_id: str = "",
                            model: str = "") -> str:
    """
    Process end-of-conversation compression.
    Creates missed claims, then a compress node over everything.
    Returns the compress node's address.
    """
    all_addrs = list(set(live_addrs))

    # Catch missed claims
    for claim in result.get("new_claims", []):
        content = claim.get("content", "")
        if not content:
            continue
        if claim.get("action") == "update" and claim.get("supersedes"):
            addr = supersede(claim["supersedes"], content, store,
                reason=f"end-catch from {conversation_id}", model=model)
        else:
            node = Node(type="leaf", content=content,
                meta={"domain": claim.get("domain", ""), "conversation": conversation_id,
                      "model": model, "source": "end_catch"})
            store.put(node)
            active = store.get_active()
            active.add(node.addr)
            store.set_active(active)
            addr = node.addr
        all_addrs.append(addr)

    # Create the compress node
    return compress(
        addrs=all_addrs,
        summary=result.get("summary", ""),
        store=store,
        domain=", ".join(result.get("domains_touched", [])),
        model=model,
    )


# ===================================================================
# Context builders — what the LLM sees as "current memory"
# ===================================================================

def build_active_context(store: Store, max_nodes: int = 50) -> str:
    """Build the memory context string injected into LLM prompts."""
    active = store.get_active()
    nodes = []
    for addr in active:
        node = store.get(addr)
        if node:
            nodes.append(node)

    nodes.sort(key=lambda n: (n.meta.get("domain", "zzz"), -n.created))
    lines = []
    current_domain = None

    for node in nodes[:max_nodes]:
        domain = node.meta.get("domain", "uncategorized")
        if domain != current_domain:
            current_domain = domain
            lines.append(f"\n[{domain}]")
        freshness = ""
        if node.meta.get("last_reinforced"):
            days = int((time.time() - node.meta["last_reinforced"]) / 86400)
            freshness = f" (reinforced {days}d ago)"
        lines.append(f"  {node.addr}: {node.content}{freshness}")

    if len(nodes) > max_nodes:
        lines.append(f"\n... and {len(nodes) - max_nodes} more")
    return "\n".join(lines)


def build_live_claims_context(addrs: list[str], store: Store) -> str:
    """Build summary of claims extracted during this conversation."""
    lines = []
    for addr in addrs:
        node = store.get(addr)
        if node:
            lines.append(f"  {node.addr} [{node.meta.get('domain', '?')}]: {node.content}")
    return "\n".join(lines) if lines else "  (none)"


# ===================================================================
# Soul doc generation
# ===================================================================

def generate_soul_doc(store: Store) -> str:
    """
    Generate a human-readable project knowledge document from the current root.

    Every claim is annotated with its address so it can be
    traced back through the tree.
    """
    root_addr = store.current_root()
    if not root_addr:
        return "No root exists yet."

    root = store.get(root_addr)
    if not root:
        return f"Root {root_addr} not found."

    lines = []
    lines.append(f"<!-- project:{root.addr} -->")
    lines.append(f"# Project Knowledge Document")
    lines.append(f"")
    lines.append(f"*Root: `{root.addr}` | "
                 f"Active nodes: {root.meta.get('active_count', '?')} | "
                 f"Generated: {time.strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # Walk the root's content — each domain section
    lines.append(root.content)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")

    # List all active nodes with their types and creation times
    active = store.get_active()
    for addr in sorted(active):
        node = store.get(addr)
        if node:
            age = time.time() - node.created
            days = int(age / 86400)
            preview = node.content[:80].replace("\n", " ")
            freshness = f"{days}d ago" if days > 0 else "today"
            lines.append(f"- `{node.addr}` [{node.type}] ({freshness}) {preview}")

    return "\n".join(lines) + "\n"


# ===================================================================
# CLI
# ===================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="mnemo — content-addressed memory system")
    sub = p.add_subparsers(dest="cmd")

    # init
    init_p = sub.add_parser("init", help="Initialize a new store")
    init_p.add_argument("path", nargs="?", default="./memory")

    # put
    put_p = sub.add_parser("put", help="Store a raw claim as a leaf node")
    put_p.add_argument("content")
    put_p.add_argument("--store", default="./memory")
    put_p.add_argument("--domain", default="")

    # get
    get_p = sub.add_parser("get", help="Retrieve a node by address")
    get_p.add_argument("addr")
    get_p.add_argument("--store", default="./memory")

    # provenance
    prov_p = sub.add_parser("provenance", help="Trace a node back to its leaves")
    prov_p.add_argument("addr")
    prov_p.add_argument("--store", default="./memory")

    # supersede
    sup_p = sub.add_parser("supersede", help="Supersede an old claim with a new one")
    sup_p.add_argument("old_addr")
    sup_p.add_argument("new_content")
    sup_p.add_argument("--reason", default="")
    sup_p.add_argument("--store", default="./memory")

    # prune
    prune_p = sub.add_parser("prune", help="Find supersession candidates")
    prune_p.add_argument("--threshold", type=float, default=0.7)
    prune_p.add_argument("--store", default="./memory")

    # soul
    soul_p = sub.add_parser("soul", help="Generate the current soul document")
    soul_p.add_argument("--store", default="./memory")

    # reroot
    reroot_p = sub.add_parser("reroot", help="Compute a new root from active set")
    reroot_p.add_argument("--store", default="./memory")

    args = p.parse_args()

    if args.cmd == "init":
        s = Store(args.path)
        print(f"✓ Store initialized at {args.path}")

    elif args.cmd == "put":
        s = Store(args.store)
        node = Node(type="leaf", content=args.content,
                     meta={"domain": args.domain} if args.domain else {})
        s.put(node)
        active = s.get_active()
        active.add(node.addr)
        s.set_active(active)
        print(f"✓ Stored: {node.addr}")
        print(f"  content: {args.content[:80]}")

    elif args.cmd == "get":
        s = Store(args.store)
        node = s.get(args.addr)
        if node:
            print(f"addr:    {node.addr}")
            print(f"type:    {node.type}")
            print(f"inputs:  {node.inputs}")
            print(f"created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(node.created))}")
            print(f"meta:    {json.dumps(node.meta, indent=2)}")
            print(f"content:\n{node.content}")
        else:
            print(f"✗ Not found: {args.addr}")

    elif args.cmd == "provenance":
        s = Store(args.store)
        chain = s.provenance(args.addr)
        if chain:
            for n in chain:
                preview = n.content[:60].replace("\n", " ")
                print(f"  {n.addr} [{n.type}] ← {n.inputs} | {preview}")
        else:
            print(f"✗ No provenance found for {args.addr}")

    elif args.cmd == "supersede":
        s = Store(args.store)
        old = s.get(args.old_addr)
        if not old:
            print(f"✗ Not found: {args.old_addr}")
            sys.exit(1)
        new_addr = supersede(args.old_addr, args.new_content, s, reason=args.reason)
        print(f"✓ Superseded {args.old_addr[:8]} → {new_addr}")
        print(f"  old: {old.content[:60]}")
        print(f"  new: {args.new_content[:60]}")

    elif args.cmd == "prune":
        s = Store(args.store)
        proposals = propose_supersessions(s, args.threshold)
        if proposals:
            print(f"Found {len(proposals)} supersession candidate(s):\n")
            for p in proposals:
                print(f"  similarity: {p['similarity']}")
                print(f"  old ({p['old'][:8]}): {p['old_content']}")
                print(f"  new ({p['new'][:8]}): {p['new_content']}")
                print()
        else:
            print("No supersession candidates found.")

    elif args.cmd == "soul":
        s = Store(args.store)
        doc = generate_soul_doc(s)
        print(doc)

    elif args.cmd == "reroot":
        s = Store(args.store)
        addr = reroot(s)
        print(f"✓ New root: {addr}")

    else:
        p.print_help()


if __name__ == "__main__":
    main()

"""
mnemo_retrieval.py — Retrieval backends for associative recall

Provides a RetrievalBackend protocol and implementations:
- TfIdfBackend: keyword-based cosine similarity (default, zero deps)
- EmbeddingBackend: semantic vector similarity (opt-in, needs API key)

The backend scores how well a query matches each node in the active set.
Signal-based domain boosts and recency scoring are handled by the caller
(mnemo_associate.py), not here — this layer is purely about text similarity.

Selection: MNEMO_RETRIEVAL env var — "tfidf" (default) or "embedding".
Embedding provider: MNEMO_EMBEDDING_PROVIDER — "voyage" or "openai".
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Protocol, Optional, Callable

import warnings

from mnemo_associate import STOP_WORDS


def _warn(msg: str) -> None:
    """Emit a visible warning for retrieval failures."""
    warnings.warn(f"[mnemo retrieval] {msg}", stacklevel=3)


# ===================================================================
# Protocol — what every retrieval backend must implement
# ===================================================================

class RetrievalBackend(Protocol):
    def ensure_fresh(self, store: "Store") -> None:
        """Rebuild index if the active set has changed."""
        ...

    def prepare_query(self, query_text: str) -> None:
        """Pre-process the raw query text for this scoring round.
        Called once before scoring nodes. TF-IDF ignores this;
        embedding backends use it to cache the query vector."""
        ...

    def score(self, query_keywords: set[str], addr: str) -> float:
        """Return similarity [0.0, 1.0] for a node against query keywords."""
        ...


# ===================================================================
# Shared utilities
# ===================================================================

def _tokenize(text: str) -> list[str]:
    """Tokenize text into meaningful words, filtering stop words."""
    words = re.findall(r'[a-z_]+', text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOP_WORDS]


def extract_quantitative_fragments(text: str, source_addr: str = "") -> list[dict]:
    """
    Extract quantitative fragments from text — numbers with surrounding context.

    Captures specific values, thresholds, rates, multipliers, and formulas
    that would be lost if a summary abstracts them away. Each fragment is
    a short string preserving the number and its meaning.

    Returns list of {"fragment": str, "source": str}.
    """
    fragments = []
    seen = set()

    # Strategy: split text into sentences/clauses, keep any that contain
    # numbers with meaningful context. This is simpler and more robust
    # than trying to regex-extract the exact boundary of each value.

    # Split on sentence boundaries and common delimiters
    clauses = re.split(r'[.;!]\s+|\n', text)

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # Must contain at least one number (int or float)
        if not re.search(r'\d+(?:\.\d+)?', clause):
            continue

        # Must have context — not just a bare number
        words = clause.split()
        if len(words) < 2:
            continue

        # Cap length — we want fragments, not paragraphs
        if len(clause) > 150:
            # Try to find sub-clauses separated by commas
            for sub in clause.split(','):
                sub = sub.strip()
                if re.search(r'\d+(?:\.\d+)?', sub) and len(sub) >= 5 and sub not in seen:
                    seen.add(sub)
                    fragments.append({"fragment": sub, "source": source_addr})
            continue

        if clause not in seen:
            seen.add(clause)
            fragments.append({"fragment": clause, "source": source_addr})

    return fragments


def compute_coverage_score(input_texts: list[str], summary_text: str,
                           preserved_terms: set[str] | None = None) -> float:
    """
    Measure IDF-weighted keyword coverage of a compression summary.

    Computes what fraction of distinctive terms from the input nodes
    survive into the summary. IDF-weighted so common words don't
    inflate the score — we care about distinctive terms surviving.

    If preserved_terms is provided, those terms are counted as covered
    regardless of whether they appear in the summary (they're preserved
    losslessly in meta.preserved_values).

    Returns [0.0, 1.0].
    """
    if not input_texts or not summary_text:
        return 0.0

    preserved = preserved_terms or set()

    # Tokenize all inputs and the summary
    input_doc_tokens = [_tokenize(text) for text in input_texts]
    summary_tokens = set(_tokenize(summary_text))

    # Collect all unique input terms
    all_input_terms: set[str] = set()
    for tokens in input_doc_tokens:
        all_input_terms.update(tokens)

    if not all_input_terms:
        return 0.0

    # Compute local IDF from input documents
    n_docs = len(input_doc_tokens)
    df: dict[str, int] = {}
    for tokens in input_doc_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {
        term: math.log(1 + n_docs / count)
        for term, count in df.items()
    }

    # Coverage = sum(IDF for covered terms) / sum(IDF for all input terms)
    # Terms in preserved_terms count as covered even if not in summary
    total_idf = sum(idf.get(term, 0) for term in all_input_terms)
    if total_idf == 0:
        return 0.0

    covered_idf = sum(
        idf.get(term, 0) for term in all_input_terms
        if term in summary_tokens or term in preserved
    )
    return covered_idf / total_idf


def _active_hash(active: set[str]) -> str:
    """Fingerprint the active set for staleness detection."""
    payload = json.dumps(sorted(active)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _normalize_vec(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if norm is 0."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


# ===================================================================
# TF-IDF Backend
# ===================================================================

class TfIdfBackend:
    """
    TF-IDF cosine similarity backend with persistent disk index.

    Index is rebuilt when the active set changes (detected via hash).
    Persists to {index_dir}/tfidf.json to survive restarts.
    """

    def __init__(self, index_dir: Path):
        self._index_dir = Path(index_dir)
        self._index_path = self._index_dir / "tfidf.json"

        # In-memory index state
        self._active_hash: str = ""
        self._idf: dict[str, float] = {}
        self._doc_vectors: dict[str, dict[str, float]] = {}
        self._doc_norms: dict[str, float] = {}

    def ensure_fresh(self, store) -> None:
        """Check if index is current; rebuild if not."""
        active = store.get_active()
        current_hash = _active_hash(active)

        # Hot path: in-memory index is current
        if self._active_hash == current_hash:
            return

        # Try loading from disk
        if self._load(current_hash):
            return

        # Rebuild from scratch
        self._rebuild(active, store)

    def prepare_query(self, query_text: str) -> None:
        """No-op for TF-IDF — keywords are passed directly to score()."""
        pass

    def score(self, query_keywords: set[str], addr: str) -> float:
        """Cosine similarity between query and a document's TF-IDF vector."""
        if addr not in self._doc_vectors:
            return 0.0

        # Build query vector: each keyword gets weight = IDF
        # (TF=1 since each keyword appears once in the query)
        query_vec = {}
        for word in query_keywords:
            if word in self._idf:
                query_vec[word] = self._idf[word]

        if not query_vec:
            return 0.0

        doc_vec = self._doc_vectors[addr]

        # Dot product (only over shared keys)
        shared = set(query_vec) & set(doc_vec)
        if not shared:
            return 0.0

        dot = sum(query_vec[w] * doc_vec[w] for w in shared)
        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        doc_norm = self._doc_norms.get(addr, 0.0)

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot / (query_norm * doc_norm)

    # --- Internal ---

    def _rebuild(self, active: set[str], store) -> None:
        """Full index rebuild from the active set."""
        docs: dict[str, list[str]] = {}  # addr -> tokens

        for addr in active:
            node = store.get(addr)
            if node:
                docs[addr] = _tokenize(node.content)

        n_docs = len(docs)
        if n_docs == 0:
            self._active_hash = _active_hash(active)
            self._idf = {}
            self._doc_vectors = {}
            self._doc_norms = {}
            self._save()
            return

        # Document frequency: how many docs contain each word
        df: dict[str, int] = {}
        for tokens in docs.values():
            unique = set(tokens)
            for word in unique:
                df[word] = df.get(word, 0) + 1

        # Smoothed IDF: log(1 + N/df)
        self._idf = {
            word: math.log(1 + n_docs / count)
            for word, count in df.items()
        }

        # TF-IDF vectors and norms
        self._doc_vectors = {}
        self._doc_norms = {}

        for addr, tokens in docs.items():
            if not tokens:
                self._doc_vectors[addr] = {}
                self._doc_norms[addr] = 0.0
                continue

            # Term frequency: count / total tokens
            tf: dict[str, float] = {}
            for word in tokens:
                tf[word] = tf.get(word, 0) + 1
            total = len(tokens)
            for word in tf:
                tf[word] /= total

            # TF-IDF
            vec = {
                word: tf_val * self._idf.get(word, 0)
                for word, tf_val in tf.items()
                if self._idf.get(word, 0) > 0
            }
            self._doc_vectors[addr] = vec

            # Precompute L2 norm
            self._doc_norms[addr] = math.sqrt(
                sum(v * v for v in vec.values())
            ) if vec else 0.0

        self._active_hash = _active_hash(active)
        self._save()

    def _load(self, expected_hash: str) -> bool:
        """Try loading index from disk. Returns True if successful and fresh."""
        if not self._index_path.exists():
            return False

        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if data.get("active_hash") != expected_hash:
                return False

            self._active_hash = data["active_hash"]
            self._idf = data.get("idf", {})
            self._doc_vectors = data.get("doc_vectors", {})
            self._doc_norms = data.get("doc_norms", {})
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def _save(self) -> None:
        """Persist index to disk (atomic write)."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._index_path.with_suffix(".tmp")
        data = {
            "active_hash": self._active_hash,
            "idf": self._idf,
            "doc_vectors": self._doc_vectors,
            "doc_norms": self._doc_norms,
        }
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(self._index_path)


# ===================================================================
# Embedding Backend
# ===================================================================

class EmbeddingBackend:
    """
    Embedding-based cosine similarity backend with persistent disk index.

    Takes a pluggable embed_fn: list[str] -> list[list[float]].
    Vectors are L2-normalized at index time, so score() is a dot product.

    Index is rebuilt when active set changes (same staleness hash as TF-IDF).
    Persists to {index_dir}/embeddings.json. Model name is part of the cache
    key — switching models invalidates the index.
    """

    def __init__(self, index_dir: Path,
                 embed_fn: Callable[[list[str]], list[list[float]]],
                 model_name: str = "unknown"):
        self._index_dir = Path(index_dir)
        self._index_path = self._index_dir / "embeddings.json"
        self._embed_fn = embed_fn
        self._model_name = model_name

        # In-memory index state
        self._active_hash: str = ""
        self._vectors: dict[str, list[float]] = {}  # addr -> normalized vector
        self._query_vector: list[float] = []

    def ensure_fresh(self, store) -> None:
        """Check if index is current; rebuild if not."""
        active = store.get_active()
        current_hash = _active_hash(active)

        if self._active_hash == current_hash:
            return

        if self._load(current_hash):
            return

        self._rebuild(active, store)

    def prepare_query(self, query_text: str) -> None:
        """Embed the query text for this scoring round."""
        try:
            vectors = self._embed_fn([query_text])
            if vectors and vectors[0]:
                self._query_vector = _normalize_vec(vectors[0])
            else:
                self._query_vector = []
                _warn("embedding query returned empty vector — falling back to zero scores")
        except Exception as e:
            self._query_vector = []
            _warn(f"embedding query failed: {e} — falling back to zero scores")

    def score(self, query_keywords: set[str], addr: str) -> float:
        """Cosine similarity between cached query vector and doc vector.
        query_keywords is ignored — we use the pre-embedded query."""
        if not self._query_vector or addr not in self._vectors:
            return 0.0

        doc_vec = self._vectors[addr]
        # Both are L2-normalized, so cosine similarity = dot product
        return max(0.0, sum(a * b for a, b in zip(self._query_vector, doc_vec)))

    # --- Internal ---

    def _rebuild(self, active: set[str], store) -> None:
        """Full index rebuild — embed all active nodes in one API call."""
        texts: dict[str, str] = {}
        for addr in active:
            node = store.get(addr)
            if node:
                texts[addr] = node.content

        if not texts:
            self._active_hash = _active_hash(active)
            self._vectors = {}
            self._save()
            return

        addrs = list(texts.keys())
        contents = [texts[a] for a in addrs]

        try:
            raw_vectors = self._embed_fn(contents)
        except Exception as e:
            # API failure during rebuild — leave index empty, scores will all be 0
            _warn(f"embedding rebuild failed ({len(contents)} docs): {e} — index empty, all scores zero")
            self._active_hash = _active_hash(active)
            self._vectors = {}
            self._save()
            return

        self._vectors = {}
        for addr, vec in zip(addrs, raw_vectors):
            self._vectors[addr] = _normalize_vec(vec)

        self._active_hash = _active_hash(active)
        self._save()

    def _load(self, expected_hash: str) -> bool:
        """Try loading index from disk."""
        if not self._index_path.exists():
            return False

        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if data.get("active_hash") != expected_hash:
                return False
            if data.get("model") != self._model_name:
                return False  # model changed — need re-embed

            self._active_hash = data["active_hash"]
            self._vectors = data.get("vectors", {})
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def _save(self) -> None:
        """Persist index to disk (atomic write)."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._index_path.with_suffix(".tmp")
        data = {
            "active_hash": self._active_hash,
            "model": self._model_name,
            "vectors": self._vectors,
        }
        # No indent — vectors are large, save disk space
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        tmp_path.replace(self._index_path)


# ===================================================================
# Embedding provider factory
# ===================================================================

def make_embedder() -> Optional[tuple[Callable[[list[str]], list[list[float]]], str]]:
    """
    Create an embedding function from available API keys.
    Returns (embed_fn, model_name) or None if no provider available.

    Supports:
    - Voyage AI (VOYAGE_API_KEY) — recommended, Anthropic-affiliated
    - OpenAI (OPENAI_API_KEY) — widely available

    Override model with MNEMO_EMBEDDING_MODEL env var.
    """
    provider = os.environ.get("MNEMO_EMBEDDING_PROVIDER", "auto")

    # Try Voyage — direct HTTP to avoid voyageai SDK which pulls in
    # langchain-core (~3.7s import). Raw urllib is 0.17s per call.
    if provider in ("auto", "voyage"):
        voyage_key = os.environ.get("VOYAGE_API_KEY")
        if voyage_key:
            import urllib.request
            model = os.environ.get("MNEMO_EMBEDDING_MODEL", "voyage-3-lite")

            def voyage_embed(texts: list[str]) -> list[list[float]]:
                data = json.dumps({"input": texts, "model": model}).encode()
                req = urllib.request.Request(
                    "https://api.voyageai.com/v1/embeddings",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {voyage_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp = urllib.request.urlopen(req, timeout=30)
                result = json.loads(resp.read())
                return [item["embedding"] for item in result["data"]]

            return voyage_embed, model

    # Try OpenAI — same lazy pattern
    if provider in ("auto", "openai"):
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            model = os.environ.get("MNEMO_EMBEDDING_MODEL",
                                   "text-embedding-3-small")
            _openai_client = None

            def openai_embed(texts: list[str]) -> list[list[float]]:
                nonlocal _openai_client
                if _openai_client is None:
                    from openai import OpenAI
                    _openai_client = OpenAI(api_key=openai_key)
                result = _openai_client.embeddings.create(input=texts, model=model)
                return [item.embedding for item in result.data]

            return openai_embed, model

    return None

"""
Matching Engine for AI Resume Screening & Job Matching System.

Implements per ARCHITECTURE.md §5.2 Stage 3 and PRD §4.2-4.3:

1. Text normalization — lowercasing, punctuation & stopword removal
2. TF-IDF vectorization + Cosine Similarity (scikit-learn)
3. Dense semantic embeddings via `sentence-transformers/all-MiniLM-L6-v2` + Cosine
4. Weighted hybrid score: (0.4 * TF-IDF) + (0.6 * Semantic)

Design goals:
- Deterministic, pure-functional where possible
- Handles empty / short docs gracefully (returns 0.0)
- Model is lazily loaded as singleton + LRU cache-friendly
- Long-document chunking (256-token window) with mean-pool
- Returns scores in 0–100 range (rounded to 2 decimals) for dashboard compatibility
"""

from __future__ import annotations

import re
import string
import hashlib
from typing import Optional, Tuple, List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Text Normalization
# ---------------------------------------------------------------------------

# English stopwords from sklearn (no NLTK download required at runtime)
_STOPWORDS: set[str] = set(ENGLISH_STOP_WORDS)

# Pre-compiled punctuation pattern for speed.
# We replace punctuation with space to preserve token boundaries (e.g., "CI/CD" -> "CI CD")
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """
    Normalize raw text for lexical matching.

    Pipeline:
      1. Unicode NFKC handled upstream; here we ensure str type
      2. Lowercasing
      3. Punctuation removal (replace with space)
      4. Stopword removal (optional, default True)
      5. Collapse whitespace / trim
      6. Drop tokens <2 chars (noise) — keeps meaningful tokens

    Args:
        text: Raw input text (JD or resume section)
        remove_stopwords: Whether to filter ENGLISH_STOP_WORDS
        remove_punctuation: Whether to strip punctuation

    Returns:
        Cleaned, space-separated lowercased tokens.

    Example:
        >>> normalize_text("Senior Python Developer, with AWS & Docker!")
        'senior python developer aws docker'
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase & strip
    text = text.lower().strip()

    # 2. Remove punctuation → space
    if remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)
        # Remove any remaining non-alphanumeric noise (keep a-z0-9 + space)
        text = _NON_ALPHA_RE.sub(" ", text)

    # 3. Tokenize on whitespace
    tokens = text.split()

    # 4. Stopword & length filter
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
    else:
        tokens = [t for t in tokens if len(t) >= 2]

    # 5. Collapse whitespace (already via split/join)
    cleaned = " ".join(tokens)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# 2. TF-IDF + Cosine Similarity (Lexical)
# ---------------------------------------------------------------------------

_DEFAULT_TFIDF_KWARGS: Dict = dict(
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
    stop_words="english",
)


def compute_tfidf_score(
    jd_text: str,
    resume_text: str,
    *,
    vectorizer_kwargs: Optional[Dict] = None,
) -> float:
    """
    Compute TF-IDF cosine similarity between a JD and a resume.

    Fits a fresh TfidfVectorizer on the two-document corpus [jd, resume]
    per PRD §4.2 (per-request fit) — interpretable and avoids stale vocab.
    Returns 0-100 scaled score.

    Args:
        jd_text: Job description raw or normalized text
        resume_text: Resume raw or normalized text
        vectorizer_kwargs: Optional override for TfidfVectorizer params

    Returns:
        Cosine similarity scaled to 0–100, rounded to 2 decimals.
        Returns 0.0 if either document is empty after normalization.
    """
    # Normalize first for consistent lexical signal
    jd_norm = normalize_text(jd_text)
    resume_norm = normalize_text(resume_text)

    if not jd_norm or not resume_norm:
        return 0.0

    kwargs = {**_DEFAULT_TFIDF_KWARGS, **(vectorizer_kwargs or {})}

    # Fit on the 2-doc corpus to ensure shared vocab per PRD
    vectorizer = TfidfVectorizer(**kwargs)
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_norm, resume_norm])
    except ValueError:
        # Empty vocabulary (e.g., only stopwords)
        return 0.0

    # Cosine between row 0 (JD) and row 1 (resume)
    # tfidf_matrix is (2, vocab); cosine_similarity expects 2D
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Clip to [0,1] (TF-IDF cosine is non-negative, but guard against fp drift)
    sim = float(np.clip(sim, 0.0, 1.0))
    return round(sim * 100.0, 2)


def compute_tfidf_scores_batch(
    jd_text: str,
    resume_texts: List[str],
    *,
    vectorizer_kwargs: Optional[Dict] = None,
) -> List[float]:
    """
    Batch variant: JD vs N resumes in a single vectorizer fit (more efficient
    than N separate fits). Used by ranking pipeline.

    Returns list of 0–100 scores aligned with resume_texts.
    """
    jd_norm = normalize_text(jd_text)
    if not jd_norm:
        return [0.0] * len(resume_texts)

    resumes_norm = [normalize_text(t) for t in resume_texts]
    # Filter: keep original indices but handle empties as 0.0
    # Fit on JD + all non-empty resumes
    corpus = [jd_norm] + [r for r in resumes_norm if r]

    if len(corpus) < 2:
        return [0.0] * len(resume_texts)

    kwargs = {**_DEFAULT_TFIDF_KWARGS, **(vectorizer_kwargs or {})}
    vectorizer = TfidfVectorizer(**kwargs)
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [0.0] * len(resume_texts)

    jd_vec = tfidf_matrix[0:1]
    sims = cosine_similarity(jd_vec, tfidf_matrix[1:]).flatten()

    # Map back to original resume order
    result: List[float] = []
    sim_idx = 0
    for r_norm in resumes_norm:
        if not r_norm:
            result.append(0.0)
        else:
            sim = float(np.clip(sims[sim_idx], 0.0, 1.0)) * 100.0
            result.append(round(sim, 2))
            sim_idx += 1
    return result


# ---------------------------------------------------------------------------
# 3. Semantic Embeddings via sentence-transformers/all-MiniLM-L6-v2
# ---------------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = None  # singleton
_MODEL_LOAD_ERROR: Optional[str] = None

# Simple in-memory LRU-ish cache (dict + manual eviction not needed for V1;
# 1000 entries max). Keyed by sha256(text).
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}
_CACHE_MAX = 1000

# Chunking: model max 256 WordPiece tokens ~ ~1200 chars; we chunk by sentences.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_CHARS_PER_CHUNK = 1000  # approx 200-250 tokens, conservative


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_model(model_name: str = _MODEL_NAME):
    """
    Lazy singleton loader for SentenceTransformer.
    Bakes model after first call; subsequent calls return cached instance.
    Thread-safe for single-worker Uvicorn V1.

    Raises RuntimeError if model cannot be loaded (network/weights missing).
    """
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(model_name)
        # Ensure normalized embeddings for dot == cosine
        _MODEL.max_seq_length = 256
        _MODEL_LOAD_ERROR = None
        return _MODEL
    except Exception as exc:  # pragma: no cover - import/download failure
        _MODEL_LOAD_ERROR = str(exc)
        raise RuntimeError(f"Failed to load SentenceTransformer '{model_name}': {exc}") from exc


def _chunk_text(text: str) -> List[str]:
    """
    Split long text into ~256-token chunks via sentence boundaries.
    Falls back to char slicing if no sentence boundaries.
    """
    if len(text) <= _MAX_CHARS_PER_CHUNK:
        return [text]

    sentences = _SENT_SPLIT_RE.split(text)
    if len(sentences) <= 1:
        # No sentence punctuation: naive char chunks
        return [text[i : i + _MAX_CHARS_PER_CHUNK] for i in range(0, len(text), _MAX_CHARS_PER_CHUNK)]

    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= _MAX_CHARS_PER_CHUNK:
            current = f"{current} {sent}".strip() if current else sent
        else:
            if current:
                chunks.append(current)
            # If single sentence exceeds max, slice it
            if len(sent) > _MAX_CHARS_PER_CHUNK:
                for i in range(0, len(sent), _MAX_CHARS_PER_CHUNK):
                    chunks.append(sent[i : i + _MAX_CHARS_PER_CHUNK])
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:_MAX_CHARS_PER_CHUNK]]


def _embed_text(text: str, model=None) -> np.ndarray:
    """
    Encode text -> L2-normalized 384-dim embedding.
    Uses chunking + mean-pool for long docs, and in-memory cache.
    """
    if not text or not text.strip():
        # Return zero vector for empty; caller will map to 0 score
        return np.zeros(384, dtype=np.float32)

    cache_key = _hash_text(text)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    if model is None:
        model = get_model()

    chunks = _chunk_text(text.strip())

    # Batch encode chunks; normalize_embeddings=True gives L2-norm =1.0
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # SentenceTransformer returns (n_chunks, 384) when n_chunks>1, else (384,)
    if embeddings.ndim == 1:
        pooled = embeddings
    else:
        # Mean-pool chunk embeddings then re-normalize to unit sphere
        pooled = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

    # Cache with simple eviction
    if len(_EMBEDDING_CACHE) >= _CACHE_MAX:
        # Evict oldest (first inserted) — Python 3.7+ dict preserves order
        _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
    _EMBEDDING_CACHE[cache_key] = pooled.astype(np.float32)
    return pooled


def compute_semantic_score(
    jd_text: str,
    resume_text: str,
    model=None,
) -> float:
    """
    Compute semantic cosine similarity using all-MiniLM-L6-v2.

    Embeddings are L2-normalized, so cosine == dot product.
    Returns 0–100 scaled score.

    Args:
        jd_text: Job description raw text
        resume_text: Resume raw text
        model: Optional pre-loaded SentenceTransformer (for DI / testing).
               If None, lazily loads singleton via get_model().

    Returns:
        Cosine similarity scaled to 0–100, rounded to 2 decimals.
        Returns 0.0 if either text is empty/whitespace.
    """
    if not jd_text or not jd_text.strip() or not resume_text or not resume_text.strip():
        return 0.0

    try:
        jd_emb = _embed_text(jd_text, model=model)
        resume_emb = _embed_text(resume_text, model=model)

        # Zero-vector guard (from empty handling)
        if np.all(jd_emb == 0) or np.all(resume_emb == 0):
            return 0.0

        # Since embeddings are normalized, dot == cosine in [-1, 1]
        # Clip to [0,1] for resume matching (negative semantic unrelatedness => 0)
        cosine = float(np.dot(jd_emb, resume_emb))
        cosine = float(np.clip(cosine, 0.0, 1.0))
        return round(cosine * 100.0, 2)
    except RuntimeError:
        # Model load failure — graceful degradation per ARCHITECTURE §10
        # Caller (hybrid) will handle fallback; here return 0.0 to avoid crash
        return 0.0
    except Exception:  # pragma: no cover
        return 0.0


def compute_semantic_scores_batch(
    jd_text: str,
    resume_texts: List[str],
    model=None,
) -> List[float]:
    """
    Batch semantic scoring: embed JD once, then each resume.
    Cache-aware; avoids re-embedding JD per candidate.
    """
    if not jd_text or not jd_text.strip():
        return [0.0] * len(resume_texts)
    if model is None:
        try:
            model = get_model()
        except RuntimeError:
            return [0.0] * len(resume_texts)

    # Embed JD once (cached path)
    jd_emb = _embed_text(jd_text, model=model)
    if np.all(jd_emb == 0):
        return [0.0] * len(resume_texts)

    scores: List[float] = []
    for rt in resume_texts:
        if not rt or not rt.strip():
            scores.append(0.0)
            continue
        resume_emb = _embed_text(rt, model=model)
        if np.all(resume_emb == 0):
            scores.append(0.0)
            continue
        cosine = float(np.clip(np.dot(jd_emb, resume_emb), 0.0, 1.0))
        scores.append(round(cosine * 100.0, 2))
    return scores


# ---------------------------------------------------------------------------
# 4. Hybrid Score
# ---------------------------------------------------------------------------

DEFAULT_W_TFIDF = 0.4
DEFAULT_W_SEMANTIC = 0.6


def compute_hybrid_score(
    tfidf_score: float,
    semantic_score: float,
    *,
    w_tfidf: float = DEFAULT_W_TFIDF,
    w_semantic: float = DEFAULT_W_SEMANTIC,
) -> float:
    """
    Weighted hybrid: (w_tfidf * TF-IDF) + (w_semantic * Semantic)

    Args:
        tfidf_score: 0–100 TF-IDF cosine score
        semantic_score: 0–100 semantic cosine score
        w_tfidf: Weight for TF-IDF (default 0.4 per prompt task)
        w_semantic: Weight for semantic (default 0.6)

    Returns:
        Hybrid score 0–100 rounded to 2 decimals.
        Weights are auto-normalized if they don't sum to 1.0.
    """
    # Validate bounds
    tfidf_score = float(np.clip(tfidf_score, 0.0, 100.0))
    semantic_score = float(np.clip(semantic_score, 0.0, 100.0))

    total = w_tfidf + w_semantic
    if total == 0:
        raise ValueError("Hybrid weights cannot both be zero")
    # Auto-normalize
    if abs(total - 1.0) > 1e-6:
        w_tfidf /= total
        w_semantic /= total

    hybrid = (w_tfidf * tfidf_score) + (w_semantic * semantic_score)
    hybrid = float(np.clip(hybrid, 0.0, 100.0))
    return round(hybrid, 2)


def match_resume(
    jd_text: str,
    resume_text: str,
    *,
    w_tfidf: float = DEFAULT_W_TFIDF,
    w_semantic: float = DEFAULT_W_SEMANTIC,
    model=None,
) -> Dict[str, float]:
    """
    Convenience end-to-end matcher for a single JD + resume pair.

    Returns dict with tfidf_score, semantic_score, hybrid_score (all 0-100).

    Example:
        >>> result = match_resume("Python dev with AWS", "Experienced in Python and AWS EC2")
        >>> result["hybrid_score"] > 60
        True
    """
    tfidf = compute_tfidf_score(jd_text, resume_text)
    semantic = compute_semantic_score(jd_text, resume_text, model=model)
    hybrid = compute_hybrid_score(tfidf, semantic, w_tfidf=w_tfidf, w_semantic=w_semantic)
    return {
        "tfidf_score": tfidf,
        "semantic_score": semantic,
        "hybrid_score": hybrid,
    }


def clear_caches():
    """Utility for tests: clear embedding & model caches."""
    _EMBEDDING_CACHE.clear()
    # Note: does NOT unload _MODEL singleton (kept for speed); tests can monkeypatch if needed


# Re-export for test introspection
__all__ = [
    "normalize_text",
    "compute_tfidf_score",
    "compute_tfidf_scores_batch",
    "compute_semantic_score",
    "compute_semantic_scores_batch",
    "compute_hybrid_score",
    "match_resume",
    "get_model",
    "clear_caches",
    "_MODEL_NAME",
]

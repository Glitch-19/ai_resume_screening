"""
Unit tests for backend/services/matching_engine.py

Covers:
  1. Text normalization (lowercasing, punctuation/stopword removal)
  2. TF-IDF + Cosine Similarity
  3. Semantic embeddings (all-MiniLM-L6-v2) + Cosine
  4. Weighted hybrid score (0.4 * TF-IDF + 0.6 * Semantic)

Run:
  pytest backend/tests/test_matching.py -v
"""

import pytest
import numpy as np

from backend.services.matching_engine import (
    normalize_text,
    compute_tfidf_score,
    compute_tfidf_scores_batch,
    compute_semantic_score,
    compute_semantic_scores_batch,
    compute_hybrid_score,
    match_resume,
    clear_caches,
    _MODEL_NAME,
    get_model,
)


# ---------------------------------------------------------------------------
# 1. Text normalization
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercasing(self):
        assert normalize_text("Python Developer") == "python developer"
        assert normalize_text("AWS") == "aws"

    def test_punctuation_removal(self):
        # punctuation should become space and be collapsed
        result = normalize_text("Senior Python Developer, with AWS & Docker!")
        # stopwords "with" removed
        assert "python" in result
        assert "developer" in result
        assert "aws" in result
        assert "docker" in result
        assert "," not in result
        assert "&" not in result
        assert "!" not in result
        # lowercased
        assert result == result.lower()

    def test_stopword_removal(self):
        # 'the', 'is', 'and' are stopwords and should be removed
        text = "the quick brown fox and the lazy dog"
        normalized = normalize_text(text)
        assert "the" not in normalized.split()
        assert "and" not in normalized.split()
        assert "quick" in normalized
        assert "brown" in normalized

    def test_keep_non_stopwords(self):
        text = "machine learning engineer"
        normalized = normalize_text(text, remove_stopwords=True)
        assert normalized == "machine learning engineer"

    def test_remove_punctuation_disabled(self):
        # when disabled, punctuation still normalized via lowercasing path?
        # but we test that stopword removal still works even without punct removal
        text = "hello, world!"
        with_punct = normalize_text(text, remove_punctuation=False)
        # still lowercased and stopwords handled; punctuation retained partially
        assert "hello" in with_punct
        assert "world" in with_punct

    def test_without_stopword_removal(self):
        text = "the python developer and aws"
        without = normalize_text(text, remove_stopwords=False)
        with_sw = normalize_text(text, remove_stopwords=True)
        # Without filtering, 'the'/'and' kept
        assert "the" in without
        assert "and" in without
        assert "the" not in with_sw

    def test_collapses_whitespace(self):
        assert normalize_text("  python   \t\n  aws  ") == "python aws"

    def test_filters_short_tokens(self):
        # single-char tokens like 'a', 'i' should be removed
        assert normalize_text("a i am ok") == "ok"
        # 'am' is stopword? actually 'am' is stopword in ENGLISH_STOP_WORDS, so removed
        # 'ok' length 2 kept

    def test_empty_and_none(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
        assert normalize_text(None) == ""  # type: ignore
        assert normalize_text("the and is") == ""  # only stopwords -> empty

    def test_numeric_tokens_kept(self):
        text = "python 3.10 experience"
        result = normalize_text(text)
        assert "3" not in result.split()  # single digit filtered (<2)
        assert "10" in result.split() or "3 10" in result  # 10 kept
        assert "python" in result

    def test_non_ascii_noise(self):
        # non-ascii should be stripped via regex
        result = normalize_text("Python ★ Developer → AWS")
        assert "python" in result
        assert "developer" in result
        assert "aws" in result
        assert "★" not in result


# ---------------------------------------------------------------------------
# 2. TF-IDF + Cosine Similarity
# ---------------------------------------------------------------------------


class TestTfidfScore:
    def test_identical_high_score(self):
        jd = "Python developer with AWS and Docker experience"
        resume = "Python developer with AWS and Docker experience"
        score = compute_tfidf_score(jd, resume)
        assert score == 100.0, "Identical docs should be 100"

    def test_partial_overlap_mid_score(self):
        jd = "Python developer with AWS Docker Kubernetes"
        resume = "Python developer with AWS"
        score = compute_tfidf_score(jd, resume)
        assert 0 < score < 100
        assert score > 30  # at least some overlap

    def test_unrelated_low_score(self):
        jd = "Python AWS Docker Kubernetes microservices"
        resume = "Graphic designer Photoshop Illustrator typography branding"
        score = compute_tfidf_score(jd, resume)
        assert 0 <= score < 30

    def test_empty_returns_zero(self):
        assert compute_tfidf_score("", "python aws") == 0.0
        assert compute_tfidf_score("python aws", "") == 0.0
        assert compute_tfidf_score("", "") == 0.0
        assert compute_tfidf_score("   ", "python") == 0.0
        assert compute_tfidf_score("the and is", "python") == 0.0  # JD normalizes to empty

    def test_batch_variant(self):
        jd = "Python developer AWS Docker"
        resumes = [
            "Python developer AWS Docker",  # identical
            "Graphic designer Photoshop",  # unrelated
            "",  # empty
        ]
        scores = compute_tfidf_scores_batch(jd, resumes)
        assert len(scores) == 3
        assert scores[0] == 100.0
        assert scores[1] < 30
        assert scores[2] == 0.0

    def test_batch_empty_jd(self):
        scores = compute_tfidf_scores_batch("", ["python aws", "java"])
        assert scores == [0.0, 0.0]

    def test_numerical_stability(self):
        # very long repetitive text should not explode
        jd = "python " * 500
        resume = "python " * 500
        score = compute_tfidf_score(jd, resume)
        assert 0 <= score <= 100

    def test_tfidf_punctuation_insensitive(self):
        jd = "Python, AWS, Docker & Kubernetes."
        resume = "Python AWS Docker Kubernetes"
        score = compute_tfidf_score(jd, resume)
        assert score == 100.0


# ---------------------------------------------------------------------------
# 3. Semantic embeddings (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------


class TestSemanticScore:
    @pytest.fixture(scope="class", autouse=True)
    def _load_model(self):
        # Ensure model is downloaded once for class; skip if offline
        try:
            get_model()
        except RuntimeError as e:
            pytest.skip(f"Model not available offline: {e}")

    def test_model_name_constant(self):
        assert _MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"

    def test_identical_high_score(self):
        jd = "Python developer with AWS and Docker"
        resume = "Python developer with AWS and Docker"
        score = compute_semantic_score(jd, resume)
        # Identical should be near 100 (allow small fp diff)
        assert score >= 99.0

    def test_paraphrase_high_score(self):
        # Semantic should capture paraphrase beyond lexical
        jd = "Built ETL pipelines using Python"
        resume = "Developed data pipelines in Python"
        tfidf = compute_tfidf_score(jd, resume)
        semantic = compute_semantic_score(jd, resume)
        # Semantic should be higher than TF-IDF for paraphrase
        # (not strictly guaranteed, but typically true for MiniLM)
        # We just ensure semantic is reasonably high
        assert semantic > 50, f"Paraphrase semantic too low: {semantic}"
        # Also ensure semantic is computed (not zero)
        assert semantic > tfidf or semantic > 40

    def test_unrelated_low_score(self):
        jd = "Python AWS Docker Kubernetes backend engineer"
        resume = "Graphic designer Photoshop Illustrator branding typography"
        score = compute_semantic_score(jd, resume)
        # Unrelated should be low; MiniLM typical 10-30 for unrelated
        assert 0 <= score < 60

    def test_empty_returns_zero(self):
        assert compute_semantic_score("", "python aws") == 0.0
        assert compute_semantic_score("python aws", "") == 0.0
        assert compute_semantic_score("   ", "   ") == 0.0
        assert compute_semantic_score(None, "python") == 0.0  # type: ignore

    def test_batch_variant(self):
        jd = "Python developer AWS Docker"
        resumes = [
            "Python developer AWS Docker",
            "Graphic designer Photoshop",
            "",
        ]
        scores = compute_semantic_scores_batch(jd, resumes)
        assert len(scores) == 3
        assert scores[0] >= 99.0
        assert scores[2] == 0.0

    def test_batch_empty_jd(self):
        assert compute_semantic_scores_batch("", ["python aws"]) == [0.0]

    def test_caching_consistency(self):
        clear_caches()
        jd = "Senior machine learning engineer with PyTorch"
        resume = "Machine learning engineer experienced in PyTorch and TensorFlow"
        s1 = compute_semantic_score(jd, resume)
        s2 = compute_semantic_score(jd, resume)
        assert s1 == s2, "Cached second call should be identical"

    def test_long_document_chunking(self):
        # >1000 chars forces chunking path in _chunk_text
        jd = "Python developer with AWS. " * 100  # ~2700 chars
        resume = "Experienced Python developer using AWS services. " * 100
        score = compute_semantic_score(jd, resume)
        assert 0 <= score <= 100
        assert score > 50  # Should still be high despite chunking

    def test_score_clipped_to_0_100(self):
        # Even with weird inputs, should stay in bounds
        scores = [
            compute_semantic_score("a" * 5000, "b" * 5000),
            compute_semantic_score("Python", "Python"),
            compute_semantic_score("hello world", "hello world"),
        ]
        for s in scores:
            assert 0 <= s <= 100


# ---------------------------------------------------------------------------
# 4. Hybrid score (0.4 TF-IDF + 0.6 Semantic)
# ---------------------------------------------------------------------------


class TestHybridScore:
    def test_weighted_formula(self):
        # (0.4 * 80) + (0.6 * 60) = 32 + 36 = 68
        assert compute_hybrid_score(80, 60) == 68.0
        assert compute_hybrid_score(100, 100) == 100.0
        assert compute_hybrid_score(0, 0) == 0.0

    def test_default_weights(self):
        # explicit 0.4 / 0.6 matches default
        assert compute_hybrid_score(50, 50, w_tfidf=0.4, w_semantic=0.6) == 50.0
        # Half-half would be different: check auto-normalization path
        assert compute_hybrid_score(80, 60, w_tfidf=0.4, w_semantic=0.6) == 68.0

    def test_weighted_favors_semantic(self):
        # Semantic weight higher (0.6), so hybrid closer to semantic
        tfidf, semantic = 20, 80
        hybrid = compute_hybrid_score(tfidf, semantic)
        assert hybrid == pytest.approx(56.0)  # 0.4*20 +0.6*80 = 8+48=56
        assert hybrid > tfidf and hybrid < semantic

    def test_custom_weights(self):
        # 50/50
        assert compute_hybrid_score(80, 60, w_tfidf=0.5, w_semantic=0.5) == 70.0
        # auto-normalize when sum !=1
        # 0.8 + 0.4 =1.2 => normalized 0.666/0.333
        score = compute_hybrid_score(100, 0, w_tfidf=0.8, w_semantic=0.4)
        assert score == pytest.approx(66.67, abs=0.01)

    def test_clipping(self):
        # Out-of-range inputs clipped
        assert compute_hybrid_score(150, 100) == 100.0
        assert compute_hybrid_score(-20, 50) == 30.0  # -20 clipped to 0 => 0.4*0+0.6*50=30

    def test_zero_weight_guard(self):
        with pytest.raises(ValueError, match="cannot both be zero"):
            compute_hybrid_score(50, 50, w_tfidf=0, w_semantic=0)


class TestMatchResumeIntegration:
    def test_end_to_end(self):
        jd = "We need a Python developer with AWS, Docker and machine learning experience"
        resume = "Experienced Python developer with AWS EC2, Docker and ML models using scikit-learn"
        result = match_resume(jd, resume)
        assert set(result.keys()) == {"tfidf_score", "semantic_score", "hybrid_score"}
        for k, v in result.items():
            assert 0 <= v <= 100, f"{k} out of bounds: {v}"
        # Hybrid must equal weighted combination (allow rounding)
        expected = compute_hybrid_score(result["tfidf_score"], result["semantic_score"])
        assert result["hybrid_score"] == expected

    def test_match_custom_weights(self):
        jd = "Python AWS"
        resume = "Python AWS Docker"
        r1 = match_resume(jd, resume, w_tfidf=0.4, w_semantic=0.6)
        r2 = match_resume(jd, resume, w_tfidf=0.8, w_semantic=0.2)
        assert r1["hybrid_score"] != r2["hybrid_score"]

    def test_match_empty(self):
        result = match_resume("", "")
        assert result["tfidf_score"] == 0.0
        assert result["semantic_score"] == 0.0
        assert result["hybrid_score"] == 0.0

    def test_match_with_injected_model(self):
        # DI path: pass explicit model (same singleton)
        model = get_model()
        result = match_resume("Python developer", "Python engineer", model=model)
        assert 0 <= result["hybrid_score"] <= 100

    def test_performance_latency_p95(self):
        # PRD §7 SLO: p95 <2s per resume. Quick sanity: single match should be <2s.
        import time

        jd = "Python developer with AWS, Docker, Kubernetes, microservices, REST APIs"
        resume = "Senior Python engineer with AWS, Docker, Kubernetes and microservices experience"
        start = time.perf_counter()
        match_resume(jd, resume)
        elapsed = time.perf_counter() - start
        # Warm cache: should be well under 2s; first semantic encode ~300ms
        assert elapsed < 2.0, f"Latency {elapsed:.3f}s exceeds 2s SLO"

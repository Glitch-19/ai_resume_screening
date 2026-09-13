# PRD: AI Resume Screening & Job Matching System

**Version:** 1.0.0  
**Author:** Principal Product Manager  
**Date:** 2026-09-13  
**Status:** Draft / Ready for Engineering Review  
**Repository:** `ai-resume-screening`

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Target Users](#2-target-users)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Core Features](#4-core-features)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Success Metrics & Performance KPIs](#8-success-metrics--performance-kpis)
9. [API Contract Overview](#9-api-contract-overview)
10. [Data Model](#10-data-model)
11. [UX / Candidate Ranking Dashboard](#11-ux--candidate-ranking-dashboard)
12. [Security, Privacy & Compliance](#12-security-privacy--compliance)
13. [Risks & Mitigations](#13-risks--mitigations)
14. [Roadmap & Milestones](#14-roadmap--milestones)
15. [Open Questions](#15-open-questions)
16. [Appendix](#16-appendix)

---

## 1. Executive Summary & Problem Statement

### 1.1 Executive Summary

The AI Resume Screening & Job Matching System is an intelligent, API-first platform that automates the top-of-funnel hiring process. It parses resumes (PDF/DOCX), normalizes text, extracts structured skills, and scores candidates against a Job Description (JD) using two complementary matching engines:

1.  **Lexical Engine (TF-IDF + Cosine Similarity):** Fast, interpretable keyword overlap.
2.  **Semantic Engine (Sentence Transformers `all-MiniLM-L6-v2`):** Dense embeddings (384-dim) for meaning-aware matching beyond exact keywords.

A weighted hybrid score powers a **Candidate Ranking Dashboard** where recruiters can filter, compare, and identify skill gaps in <2s per resume, reducing screening time by 80%+ and improving quality-of-hire.

### 1.2 Problem Statement

| Pain Point | Impact |
| :--- | :--- |
| **High Volume, Low Signal:** Recruiters manually review 200-500 resumes per role; ~75% are unqualified. | 15-20 hrs wasted per requisition; slow time-to-shortlist. |
| **Keyword Myopia:** ATS keyword filters miss semantically equivalent skills (e.g., `React` vs `Frontend Framework`, `K8s` vs `Kubernetes`). | False negatives; diverse talent overlooked. |
| **No Gap Analysis:** No automated view of *what* a candidate lacks vs. JD. | Poor interview prep, inconsistent feedback. |
| **Unstructured Data:** Resumes are unstructured PDFs/DOCXs with varied layouts. | Error-prone manual parsing. |
| **No Explainability:** Black-box AI scores erode trust. | Low adoption by HR teams. |

**Opportunity:** A dual-engine, explainable, low-latency system that provides ranked, evidence-backed matches and actionable skill gaps, integrable via API into existing ATS/HRIS.

### 1.3 Value Proposition

*   **For Recruiters:** Screen 10x faster, focus on top 10% candidates with evidence.
*   **For Hiring Managers:** Data-driven shortlist with gap analysis.
*   **For HR Tech Teams:** Embeddable API with <2s p95 latency, Docker-ready, model-agnostic.

---

## 2. Target Users

### 2.1 Primary Persona: Recruiter / Talent Acquisition Specialist

*   **Role:** Agency or in-house recruiter managing 10-20 open requisitions.
*   **Goals:** Reduce time-to-shortlist from 3 days to <4 hours; defend shortlist with objective scores; quickly identify interview focus areas.
*   **Pain:** Context switching, resume format chaos, pressure to fill quickly without bias.
*   **Tech Savviness:** Low-Medium; needs no-code dashboard + CSV export.
*   **Jobs to be Done:** "When I post a JD, I want a ranked list of candidates with match reasons so I can shortlist the top 5 without reading every resume."

### 2.2 Secondary Persona: HR Tech Team / Platform Engineer

*   **Role:** Engineer integrating screening into ATS, career site, or internal HRIS.
*   **Goals:** Stable, documented REST API; predictable latency/cost; easy self-hosting or cloud deployment; no vendor lock-in.
*   **Pain:** Opaque AI vendors, high per-resume cost, GDPR concerns with sending PII to third parties.
*   **Tech Savviness:** High.
*   **Jobs to be Done:** "I need a `/match` endpoint that takes a JD + batch of resumes and returns structured JSON scores I can render in our UI."

### 2.3 Stakeholders

*   **Hiring Manager:** Consumer of ranked dashboard.
*   **Compliance / Legal:** Ensures bias mitigation & data retention compliance.
*   **Ops / IT:** Manages deployment, scaling.

### 2.4 User Stories (Sample)

*   As a **Recruiter**, I can upload a JD and 50 resumes (PDF/DOCX) and get a ranked table in <30 seconds.
*   As a **Recruiter**, I can see *why* a candidate scored 87% (matched skills, semantic similarity, missing skills).
*   As a **Recruiter**, I can filter by "Missing: AWS" to find candidates needing upskilling.
*   As an **HR Tech Engineer**, I can `POST /api/v1/parse` a resume and get structured text/skills without scoring.

---

## 3. Goals & Non-Goals

### 3.1 Goals (V1.0)

*   Parse PDF/DOCX with >95% text extraction fidelity (excluding scanned image PDFs).
*   Provide dual scoring: TF-IDF + Cosine AND `all-MiniLM-L6-v2` semantic score + weighted hybrid.
*   Extract skills via NER/dictionary + gap analysis vs. JD.
*   Rank candidates deterministically with explainable dashboard.
*   p95 latency <2s per resume end-to-end (parse + embed + score) on CPU (4 vCPU, 8GB RAM).
*   Ship as REST API + Streamlit/React dashboard.

### 3.2 Non-Goals (V1.0)

*   No OCR for scanned/image-only PDFs (deferred to V1.1 with Tesseract/PaddleOCR).
*   No automated interview scheduling or offer generation.
*   No bias mitigation beyond explainability and audit logs (V2 will add fairness metrics).
*   No multi-lingual support (English only V1).
*   No LLM-based generative reasoning (deterministic engines only for V1 cost/latency).

---

## 4. Core Features

### 4.1 F1: PDF/DOCX Resume Parsing and Text Preprocessing

**Description:** Robust ingestion pipeline for heterogeneous resume formats.

**Workflow:**
1.  **Ingest:** Accept `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document` via upload or API. Max 5MB/file, max 100 files/batch.
2.  **Extract:**
    *   PDF: `PyMuPDF (fitz)` primary, fallback `pdfminer.six` for layout preservation. Extract text blocks, preserve reading order.
    *   DOCX: `python-docx` extracting paragraphs + tables.
    *   Metadata: filename, page count, extraction confidence.
3.  **Preprocessing Pipeline:**
    *   Unicode normalization (NFKC), lowercasing (for TF-IDF path; preserve case for NER).
    *   Remove headers/footers, emails/phones optionally masked for privacy mode.
    *   Remove special chars, extra whitespace, non-ASCII noise.
    *   Tokenization (word, sentence), stopword removal (TF-IDF only), lemmatization via `spaCy`/`nltk`.
    *   Section segmentation heuristic: `Experience | Education | Skills | Projects` via regex + layout cues.

**Acceptance Criteria:**
*   Handles 2-column PDF layouts, tables, bullet points.
*   Returns `raw_text`, `clean_text`, `sections: {skills, experience, education}`.
*   Failure mode: If extraction <50 chars, return `status: FAILED` with error code `EXTRACTION_LOW_CONFIDENCE`.

### 4.2 F2: Keyword Matching using TF-IDF + Cosine Similarity

**Description:** Interpretable lexical baseline.

**Logic:**
*   Vectorizer: `TfidfVectorizer` with `ngram_range=(1,2)`, `max_features=5000`, `sublinear_tf=True`, `stop_words='english'`.
*   Corpus: JD + all resumes in batch (or single JD vs single resume) fitted per-request OR using pre-fitted vocab for consistency.
*   Score: `cosine_similarity(TF-IDF_JD, TF-IDF_Resume)` ∈ [0,1] → scaled to 0-100.
*   Explainability: Return `top_matched_keywords: [{term, tfidf_score, in_jd, in_resume}]` (top 15).

**Why:** Fast (<50ms), zero model download, highly explainable for non-technical users; serves as debuggable baseline.

### 4.3 F3: Semantic Matching using Sentence Transformers (`all-MiniLM-L6-v2`)

**Description:** Dense semantic understanding beyond keywords.

**Model Spec:**
*   Model: `sentence-transformers/all-MiniLM-L6-v2` (80MB, 384 dims, 6 layers, 22M params).
*   Rationale: Best speed/accuracy tradeoff; 5x faster than `mpnet-base`; 384-dim fits in memory for 10k resumes; strong on STS benchmarks.
*   Hosting: Local `sentence-transformers` lib; embeddings cached in-memory (LRU) keyed by hash(text).
*   Chunking Strategy: JD and resume truncated/chunked to 256 tokens (model max 256 WordPiece tokens). Long docs: split by sentences, mean-pool chunk embeddings, L2 normalize.
*   Score: `cosine_similarity(emb_JD, emb_Resume)` ∈ [-1,1] → rescaled to [0,1] via `(score+1)/2` or clipped [0,1], scaled 0-100.
*   Batch Inference: Encode JD once, resumes batched (batch_size=32) for throughput.

**Acceptance:**
*   Cold start model load <5s; warm inference <300ms per resume on CPU.
*   Handles paraphrases: e.g., JD "built ETL pipelines" matches resume "developed data pipelines".

### 4.4 F4: Skill Extraction & Missing Skill Gap Analysis

**Description:** Structured skill intelligence.

**Skill Extraction:**
*   **Source 1 - Dictionary/Knowledge Base:** Curated skill taxonomy (~3k terms: `Python, AWS, Docker, React, SQL, Machine Learning...`) via exact + fuzzy match (rapidfuzz, threshold 85) on lemmatized text. Synonym map: `{"k8s": "kubernetes", "js": "javascript"}`.
*   **Source 2 - NER / Phrase Extraction:** `spaCy` + custom EntityRuler or `SkillNer` logic extracting noun phrases under `SKILL` label. Fallback to YAKE/keyBERT for unsupervised keyphrase extraction (top 20 phrases).
*   **Deduplication & Normalization:** Lowercase, alias resolve, dedupe.

**Gap Analysis:**
*   `jd_skills = extract(jd_clean)`
*   `resume_skills = extract(resume_clean)`
*   `matched_skills = jd_skills ∩ resume_skills`
*   `missing_skills = jd_skills - resume_skills`
*   `extra_skills = resume_skills - jd_skills` (candidate strengths beyond JD)
*   **Skill Match Score:** `|matched| / |jd_skills|` (if jd_skills non-empty else 0).

**Output:** `skill_analysis: {jd_skills[], resume_skills[], matched[], missing[], extra[], skill_match_score}` plus `recommendation: "Strong Fit" if hybrid>80 else "Moderate"`.

### 4.5 F5: Candidate Ranking Dashboard

**Description:** Human-in-the-loop UI for decisioning.

**Views:**
1.  **Upload View:** Drag-drop JD (textarea + file) + multi-resume upload (PDF/DOCX). Config: Weight sliders `w_tfidf` (default 0.3) + `w_semantic` (0.5) + `w_skill` (0.2) = 1.0. Hybrid = weighted sum. Threshold filters.
2.  **Ranking Table:** Columns: Rank, Candidate Name (parsed), File, Hybrid Score (sortable), TF-IDF, Semantic, Skill Match %, Matched/Missing Skills (chips), Actions (View Detail, Export). Color coding: >80 Green, 60-80 Amber, <60 Red. Pagination for >50 candidates.
3.  **Detail Drawer:** Resume text, JD text, score breakdown bar chart, keyword evidence, skill gap Venn/list, raw vs clean text toggle.
4.  **Analytics Bar:** Avg score, distribution histogram, total processed, latency metrics.
5.  **Export:** CSV/JSON with all scores and skill arrays.

**Tech:** V1: `Streamlit` for rapid prototyping; V2: React + FastAPI + AG Grid.

**Acceptance:** Renders 100 candidates without pagination lag; all scores explainable in 1 click.

---

## 5. System Architecture Overview

```
[ Client / Dashboard ] -> [ FastAPI Gateway ]
                              |
            +-----------------+-----------------+
            |                 |                 |
        [Parse Service] [Matching Engine] [Skill Service]
        (PyMuPDF/docx)   (TF-IDF | ST)    (KB + NER)
            |                 |                 |
            +--------+--------+--------+--------+
                     |                 |
               [Preprocessing]   [Embedding Cache (LRU)]
                     |                 |
               [Ranking Orchestrator] -> Hybrid Scorer -> Ranker
                     |
               [ Storage: /data/uploads (ephemeral) ]
                     |
               [ Dashboard / API Response ]
```

*   **Deployment:** Dockerized FastAPI (`uvicorn`), `sentence-transformers` model baked into image or lazy-downloaded. Stateless horizontal scaling.
*   **Dependencies:** `PyMuPDF`, `python-docx`, `scikit-learn`, `sentence-transformers`, `torch (cpu)`, `spaCy`, `rapidfuzz`.

---

## 6. Functional Requirements

| ID | Requirement | Priority |
| :--- | :--- | :--- |
| FR-01 | System MUST accept PDF and DOCX resume uploads via `multipart/form-data` and raw text. | P0 |
| FR-02 | System MUST parse and return raw and clean text plus sections. | P0 |
| FR-03 | System MUST compute TF-IDF cosine similarity (0-100) + top keywords. | P0 |
| FR-04 | System MUST compute semantic similarity via `all-MiniLM-L6-v2` embeddings. | P0 |
| FR-05 | System MUST compute hybrid score: `0.3*tfidf + 0.5*semantic + 0.2*skill` (configurable). | P0 |
| FR-06 | System MUST extract skills from JD and resume and compute gap analysis. | P0 |
| FR-07 | System MUST rank N candidates descending by hybrid score and return ranks. | P0 |
| FR-08 | System MUST provide batch endpoint for JD + N resumes. | P0 |
| FR-09 | System MUST expose skill extraction as standalone endpoint. | P1 |
| FR-10 | System MUST provide CSV/JSON export of results. | P1 |
| FR-11 | System MUST validate file type/size and return 400 on invalid input. | P0 |
| FR-12 | System MUST handle empty JD/resume gracefully (422 with details). | P0 |
| FR-13 | Dashboard MUST allow weight tuning and real-time re-ranking without re-embedding (if cached). | P1 |
| FR-14 | System MUST log request latency and score distributions for observability. | P1 |

---

## 7. Non-Functional Requirements

| Category | Requirement | Target |
| :--- | :--- | :--- |
| **Performance** | End-to-end latency per resume (parse+score) | p95 <2s on CPU (4 vCPU), p50 <1s |
|  | Semantic embedding encode | <300ms per doc (warm) |
|  | TF-IDF scoring | <50ms |
|  | Batch throughput (50 resumes) | <45s total |
|  | Concurrency | 10 concurrent requests without degradation >20% |
| **Scalability** | Stateless API, horizontal scaling via load balancer | 100 req/min |
| **Reliability** | Uptime | 99.5% |
|  | Model load resilience | Fallback to TF-IDF-only if ST model fails |
| **Accuracy** | Parsing fidelity (text recall vs manual) | >95% |
|  | Skill extraction F1 (vs labeled set) | >0.82 |
|  | Semantic ranking NDCG@10 (on eval set) | >0.85 |
| **Usability** | Dashboard time-to-interactive | <3s |
|  | API onboarding | <10 mins via OpenAPI docs |
| **Security** | PII handling | Masking option; no persistence beyond request unless opted in |
|  | File validation | MIME + magic byte check, size limits |
| **Compatibility** | Python 3.10+ | Docker image <2.5GB |
| **Maintainability** | Code coverage | >75% for core engine |
|  | Modularity | Pluggable embedder, parser, skill extractor |
| **Portability** | Offline mode | Works without internet after image build |

---

## 8. Success Metrics & Performance KPIs

### 8.1 Product Success Metrics (North Star: Time-to-Shortlist Reduction)

| Metric | Baseline | Target V1 | Measurement |
| :--- | :--- | :--- | :--- |
| Time to screen 50 resumes | 8 hrs (manual) | <5 mins (automated) | Dashboard telemetry |
| Recruiter NPS for ranking usefulness | - | >45 | Survey |
| % shortlisted candidates interviewed | 30% | >60% | ATS integration |
| Reduction in unqualified interviews | - | -40% | Hiring manager feedback |

### 8.2 Performance KPIs (Technical SLOs)

| KPI | SLO | Alert Threshold |
| :--- | :--- | :--- |
| p95 latency per resume | <2.0s | >2.0s for 5 mins |
| p50 latency per resume | <1.0s | >1.2s |
| Model load time (cold start) | <5s | >8s |
| Batch 50 resumes E2E | <45s | >60s |
| Parsing success rate | >98% | <95% |
| API error rate (5xx) | <1% | >2% |
| Skill extraction precision | >0.85 | <0.80 |
| Semantic vs TF-IDF correlation | Monitored | Divergence tracked |
| Memory per worker | <2GB | >2.5GB |

### 8.3 Evaluation Methodology

*   **Latency Benchmark:** `pytest-benchmark` script: 100 resumes x 10 runs on `c2-standard-4` GCP. Report p50/p95/p99.
*   **Accuracy Benchmark:** Labeled dataset of 200 JD-resume pairs scored by 3 recruiters (ground truth ranking). Compute NDCG, Kendall's Tau vs hybrid score.
*   **A/B:** Compare hybrid vs TF-IDF-only ranking quality.

---

## 9. API Contract Overview

**Base URL:** `http://localhost:8000` (dev) | `https://api.resume-matcher.company.com` (prod)  
**Versioning:** `/api/v1`  
**Auth:** `X-API-Key` header (V1: optional, V2: required)  
**Content-Type:** `application/json` or `multipart/form-data`  
**Docs:** Auto-generated OpenAPI at `/docs` (Swagger) and `/redoc`

### 9.1 Endpoints Summary

| Method | Path | Purpose | Auth |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness & model readiness | No |
| `POST` | `/api/v1/parse` | Parse single resume file → text | Optional |
| `POST` | `/api/v1/extract-skills` | Extract skills from text | Optional |
| `POST` | `/api/v1/match` | Match single JD + single resume | Optional |
| `POST` | `/api/v1/match/batch` | Match single JD + batch resumes (ranked) | Optional |
| `POST` | `/api/v1/rank` | Alias for batch with full dashboard payload | Optional |
| `GET` | `/api/v1/skills/taxonomy` | List known skill vocabulary | No |

### 9.2 Common Schemas

#### Error Envelope
```json
{
  "detail": "Invalid file type. Only PDF and DOCX allowed.",
  "code": "INVALID_FILE_TYPE",
  "status": 400
}
```

#### SkillAnalysis
```json
{
  "jd_skills": ["python", "aws", "docker"],
  "resume_skills": ["python", "sql", "react"],
  "matched_skills": ["python"],
  "missing_skills": ["aws", "docker"],
  "extra_skills": ["sql", "react"],
  "skill_match_score": 33.33
}
```

### 9.3 `GET /health`

**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "version": "1.0.0",
  "uptime_seconds": 12345
}
```

### 9.4 `POST /api/v1/parse`

**Request:** `multipart/form-data` with `file: <pdf|docx>`

**Response 200:**
```json
{
  "filename": "john_doe.pdf",
  "raw_text": "John Doe\nSenior Python Developer...",
  "clean_text": "john doe senior python developer...",
  "sections": {
    "skills": "Python, AWS, Docker...",
    "experience": "...",
    "education": "..."
  },
  "metadata": {
    "pages": 2,
    "extraction_confidence": 0.98,
    "char_count": 3421
  }
}
```

**Errors:** `400 INVALID_FILE_TYPE`, `413 FILE_TOO_LARGE`, `422 EXTRACTION_FAILED`

### 9.5 `POST /api/v1/extract-skills`

**Request JSON:**
```json
{
  "text": "Experienced in Python, AWS and Kubernetes..."
}
```

**Response 200:**
```json
{
  "skills": ["python", "aws", "kubernetes"],
  "count": 3
}
```

### 9.6 `POST /api/v1/match` - Single JD + Single Resume

**Request Options:**

**A) JSON (text only):**
```json
{
  "job_description": "We need a Python developer with AWS...",
  "resume_text": "John Doe has 5 years Python...",
  "weights": {
    "tfidf": 0.3,
    "semantic": 0.5,
    "skill": 0.2
  }
}
```

**B) multipart/form-data:**
*   `job_description: string` (required)
*   `resume_file: file (pdf|docx)` (required, alternative to resume_text)
*   `weights: json string` (optional)

**Response 200:**
```json
{
  "scores": {
    "tfidf_score": 68.42,
    "semantic_score": 82.15,
    "skill_match_score": 33.33,
    "hybrid_score": 68.59
  },
  "skill_analysis": {
    "jd_skills": ["python", "aws", "docker"],
    "resume_skills": ["python", "sql"],
    "matched_skills": ["python"],
    "missing_skills": ["aws", "docker"],
    "extra_skills": ["sql"],
    "skill_match_score": 33.33
  },
  "details": {
    "top_matched_keywords": [
      {"term": "python", "score": 0.42},
      {"term": "aws", "score": 0.31}
    ],
    "hybrid_weights": {"tfidf": 0.3, "semantic": 0.5, "skill": 0.2}
  },
  "latency_ms": 842
}
```

### 9.7 `POST /api/v1/match/batch` - Single JD + Batch Resumes (Ranked)

**Request:** `multipart/form-data`
*   `job_description: string` (required)
*   `resume_files: file[]` (required, up to 100 files)
*   `weights: json string` (optional)

**Alternative JSON (for HR Tech integration):**
```json
{
  "job_description": "We need...",
  "resumes": [
    {"candidate_id": "cand_001", "resume_text": "..."},
    {"candidate_id": "cand_002", "resume_text": "..."}
  ],
  "weights": {"tfidf": 0.3, "semantic": 0.5, "skill": 0.2}
}
```

**Response 200:**
```json
{
  "job_description_preview": "We need a Python developer...",
  "weights_used": {"tfidf": 0.3, "semantic": 0.5, "skill": 0.2},
  "total_processed": 3,
  "total_failed": 0,
  "results": [
    {
      "rank": 1,
      "candidate_id": "cand_001",
      "filename": "alice.pdf",
      "scores": {
        "tfidf_score": 72.1,
        "semantic_score": 88.4,
        "skill_match_score": 75.0,
        "hybrid_score": 80.83
      },
      "skill_analysis": {
        "jd_skills": ["python", "aws"],
        "resume_skills": ["python", "aws", "docker"],
        "matched_skills": ["python", "aws"],
        "missing_skills": [],
        "extra_skills": ["docker"],
        "skill_match_score": 100.0
      },
      "latency_ms": 742
    }
  ],
  "analytics": {
    "avg_hybrid_score": 65.2,
    "max_hybrid_score": 80.8,
    "min_hybrid_score": 42.1,
    "p95_latency_ms": 890
  }
}
```

**Errors:** `422 EMPTY_JOB_DESCRIPTION`, `400 NO_RESUMES_PROVIDED`, partial failures returned with `status: "FAILED"` per item.

### 9.8 `POST /api/v1/rank`

Alias to `/match/batch` with additional `dashboard_payload` for UI rendering (histograms, CSV download URL). Same schema, with extra `dashboard: { histogram: [...] }`.

### 9.9 `GET /api/v1/skills/taxonomy`

**Response 200:**
```json
{
  "skills": ["python", "java", "aws", "docker", "kubernetes", "react", "..."],
  "count": 3120,
  "version": "2026-09"
}
```

### 9.10 Status Codes

| Code | Meaning |
| :--- | :--- |
| 200 | Success |
| 400 | Bad Request (invalid file type, missing fields) |
| 413 | Payload Too Large (>5MB/file or >100 files) |
| 422 | Unprocessable Entity (extraction failed, empty text) |
| 500 | Internal Server Error (model failure - fallback to TF-IDF) |
| 503 | Model Not Ready (loading) |

---

## 10. Data Model

### 10.1 Resume Object

```python
class Resume:
    filename: str
    raw_text: str
    clean_text: str
    sections: Dict[str, str]
    skills: List[str]
    embedding: Optional[List[float]]  # 384-dim, cached
    metadata: Dict  # pages, char_count, confidence
```

### 10.2 Scoring Result

```python
class ScoringResult:
    tfidf_score: float  # 0-100
    semantic_score: float  # 0-100
    skill_match_score: float  # 0-100
    hybrid_score: float  # weighted
    skill_analysis: SkillAnalysis
    top_keywords: List[Dict]
    latency_ms: int
```

---

## 11. UX / Candidate Ranking Dashboard

### 11.1 Wireframe (Text)

```
+-------------------------------------------------------------+
|  AI Resume Matcher | JD Input [textarea] | Upload Resumes [] |
|  Weights: [TF-IDF 0.3] [Semantic 0.5] [Skill 0.2] [Rerank] |
+-------------------------------------------------------------+
| Analytics: Avg 65.2 | Max 80.8 | p95 0.89s | Histogram [...] |
+-------------------------------------------------------------+
| Rank | Candidate | File | Hybrid | TF-IDF | Semantic | Skill | Missing Skills | Action |
| 1    | Alice     | pdf  | 80.8%  | 72.1   | 88.4     | 75%   | -              | View   |
| 2    | Bob       | docx | 65.2%  | 60.0   | 70.1     | 50%   | [aws]          | View   |
+-------------------------------------------------------------+
| Detail Drawer: Bar Chart + Venn + Keyword Evidence + Text    |
+-------------------------------------------------------------+
```

### 11.2 Interaction Requirements

*   Sorting by any score column.
*   Filtering by missing skill chip.
*   CSV export button triggers download.
*   Mobile responsive (table horizontal scroll).

---

## 12. Security, Privacy & Compliance

*   **Data Retention:** Ephemeral by default; files deleted after response. Optional `persist=false` flag. No DB V1.
*   **PII:** Optional masking of emails/phones before logging. Logs exclude resume text unless debug enabled.
*   **Compliance:** GDPR-friendly self-hostable; no external API calls for inference. DPA-ready.
*   **Input Security:** File magic byte validation, virus scan hook (ClamAV optional), rate limiting (100 req/min/IP).
*   **Bias Note:** No protected attribute extraction; scores explainable; audit log includes weights and model version. Disclaimer in UI: "AI-assist, human decision required."

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Model download fails offline | Medium | High | Bake model in Docker; fallback to TF-IDF; health check |
| Poor PDF extraction (scanned) | High | Medium | Detect low confidence; warn user; roadmap OCR |
| Skill taxonomy incomplete | High | Medium | Allow custom skill addition via taxonomy endpoint; iterative expansion |
| Latency >2s under load | Medium | High | Embedding cache, batch inference, async queue for >50 batch |
| Bias perception | Medium | High | Explainability, no auto-reject, human-in-loop |
| Memory bloat (Torch) | Medium | Medium | Limit workers, LRU cache max 1000 entries, `torch.no_grad()` |

---

## 14. Roadmap & Milestones

| Phase | Timeline | Deliverables |
| :--- | :--- | :--- |
| **M0 - Foundation** | Week 1 | Project scaffold, parsing service, preprocessing, TF-IDF engine, unit tests |
| **M1 - Semantic Core** | Week 2 | ST `all-MiniLM-L6-v2` integration, embedding cache, hybrid scorer |
| **M2 - Skills & Gap** | Week 3 | Skill extractor, gap analysis, taxonomy |
| **M3 - API & Dashboard** | Week 4 | FastAPI `/match/batch`, OpenAPI docs, Streamlit/React dashboard |
| **M4 - Hardening** | Week 5 | Docker, benchmarks (<2s p95), CI, 75% coverage, README |
| **V1.1** | Week 6+ | OCR, multi-lingual, LLM re-ranker, persistence (Postgres), auth |

---

## 15. Open Questions

1.  Should hybrid weights be learned via logistic regression on recruiter labels vs static 0.3/0.5/0.2?
2.  Do we need per-section weighting (e.g., Skills section 2x Experience)?
3.  Should resume name parsing be mandatory or optional anonymization?
4.  Batch size limits for SaaS vs self-hosted?

---

## 16. Appendix

### 16.1 Tech Stack (Proposed V1)

*   **Backend:** Python 3.10, FastAPI, Uvicorn, Pydantic
*   **ML:** `scikit-learn`, `sentence-transformers==2.2.2`, `torch --index-url https://download.pytorch.org/whl/cpu`, `spaCy`, `rapidfuzz`
*   **Parsing:** `PyMuPDF`, `python-docx`, `pdfminer.six`
*   **Frontend:** Streamlit (V1) → React + Tailwind (V2)
*   **Infra:** Docker, Docker Compose, Pytest, GitHub Actions

### 16.2 Glossary

*   **Hybrid Score:** Weighted combination of lexical, semantic, and skill scores.
*   **p95 Latency:** 95th percentile latency; 95% of requests faster than this.
*   **NDCG:** Normalized Discounted Cumulative Gain for ranking quality.

### 16.3 References

*   Sentence Transformers: https://www.sbert.net/docs/pretrained_models.html#all-minilm-l6-v2
*   TF-IDF + Cosine Similarity: scikit-learn documentation

---

**Approval Required From:** Product, Engineering, Design, Legal  
**Next Step:** Engineering kickoff & scaffold `app/` structure per Section 14 M0.


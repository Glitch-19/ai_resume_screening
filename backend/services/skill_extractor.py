"""
Skill Extraction & Gap Detection Module

Per PRD §4.4 and ARCHITECTURE.md §5.2 Stage 4 / §7.5:

- Defines a comprehensive tech skill ontology (languages, frameworks, cloud, databases, etc.)
- Implements extraction via tokens + n-grams (up to 3) with alias resolution
- Implements gap detection: get_missing_skills(resume_skills, job_skills)

Design principles:
- Deterministic, case-insensitive, punctuation-aware
- No heavy NER model required V1; dictionary + alias map is sufficient and offline
- Supports multi-word skills ("machine learning", "ci/cd") and short aliases ("k8s" -> "kubernetes")
- Returns canonical forms for consistent ranking
"""

from __future__ import annotations

import re
import string
from typing import List, Set, Dict, Iterable, Optional

# ---------------------------------------------------------------------------
# 1. Comprehensive Tech Skill Ontology
# ---------------------------------------------------------------------------

# Canonical skill -> metadata (category + aliases)
# Categories mirror PRD skill taxonomy grouping for dashboard filtering.
_SKILL_ONTOLOGY_RAW: Dict[str, Dict] = {
    # --- Languages ---
    "python": {"aliases": ["py", "python3"], "category": "Languages"},
    "java": {"aliases": [], "category": "Languages"},
    "javascript": {"aliases": ["js", "ecmascript"], "category": "Languages"},
    "typescript": {"aliases": ["ts"], "category": "Languages"},
    "go": {"aliases": ["golang"], "category": "Languages"},
    "rust": {"aliases": [], "category": "Languages"},
    "c++": {"aliases": ["cpp", "c plus plus"], "category": "Languages"},
    "c#": {"aliases": ["csharp", "c sharp"], "category": "Languages"},
    "c": {"aliases": [], "category": "Languages"},
    "ruby": {"aliases": [], "category": "Languages"},
    "php": {"aliases": [], "category": "Languages"},
    "swift": {"aliases": [], "category": "Languages"},
    "kotlin": {"aliases": [], "category": "Languages"},
    "scala": {"aliases": [], "category": "Languages"},
    "r": {"aliases": [], "category": "Languages"},
    "perl": {"aliases": [], "category": "Languages"},
    "shell": {"aliases": ["bash", "sh", "shell scripting"], "category": "Languages"},
    "dart": {"aliases": [], "category": "Languages"},
    "elixir": {"aliases": [], "category": "Languages"},
    # --- Frontend Frameworks / Libraries ---
    "react": {"aliases": ["reactjs", "react.js"], "category": "Frameworks"},
    "angular": {"aliases": ["angularjs"], "category": "Frameworks"},
    "vue": {"aliases": ["vuejs", "vue.js"], "category": "Frameworks"},
    "next.js": {"aliases": ["nextjs", "next"], "category": "Frameworks"},
    "nuxt": {"aliases": [], "category": "Frameworks"},
    "svelte": {"aliases": [], "category": "Frameworks"},
    "jquery": {"aliases": [], "category": "Frameworks"},
    "ember": {"aliases": [], "category": "Frameworks"},
    "bootstrap": {"aliases": [], "category": "Frameworks"},
    "tailwind css": {"aliases": ["tailwind", "tailwindcss"], "category": "Frameworks"},
    "material ui": {"aliases": ["mui", "material-ui"], "category": "Frameworks"},
    # --- Backend Frameworks ---
    "django": {"aliases": [], "category": "Frameworks"},
    "flask": {"aliases": [], "category": "Frameworks"},
    "fastapi": {"aliases": [], "category": "Frameworks"},
    "spring": {"aliases": ["spring boot", "springboot"], "category": "Frameworks"},
    "spring boot": {"aliases": ["springboot"], "category": "Frameworks"},
    "express": {"aliases": ["expressjs", "express.js"], "category": "Frameworks"},
    "nest.js": {"aliases": ["nestjs", "nest"], "category": "Frameworks"},
    "laravel": {"aliases": [], "category": "Frameworks"},
    "rails": {"aliases": ["ruby on rails"], "category": "Frameworks"},
    "asp.net": {"aliases": ["aspnet", "asp.net core"], "category": "Frameworks"},
    ".net": {"aliases": ["dotnet"], "category": "Frameworks"},
    # --- AI / ML / Data ---
    "machine learning": {"aliases": ["ml"], "category": "AI/ML"},
    "deep learning": {"aliases": ["dl"], "category": "AI/ML"},
    "natural language processing": {"aliases": ["nlp"], "category": "AI/ML"},
    "computer vision": {"aliases": ["cv"], "category": "AI/ML"},
    "data science": {"aliases": [], "category": "AI/ML"},
    "data analysis": {"aliases": [], "category": "AI/ML"},
    "tensorflow": {"aliases": ["tf"], "category": "AI/ML"},
    "pytorch": {"aliases": ["torch"], "category": "AI/ML"},
    "keras": {"aliases": [], "category": "AI/ML"},
    "scikit-learn": {"aliases": ["sklearn", "scikit learn"], "category": "AI/ML"},
    "pandas": {"aliases": [], "category": "AI/ML"},
    "numpy": {"aliases": [], "category": "AI/ML"},
    "opencv": {"aliases": ["cv2"], "category": "AI/ML"},
    "hugging face": {"aliases": ["huggingface", "transformers"], "category": "AI/ML"},
    "langchain": {"aliases": [], "category": "AI/ML"},
    "llm": {"aliases": ["large language models", "generative ai", "gen ai"], "category": "AI/ML"},
    "etl": {"aliases": ["extract transform load"], "category": "Data"},
    "data engineering": {"aliases": [], "category": "Data"},
    "big data": {"aliases": [], "category": "Data"},
    "hadoop": {"aliases": [], "category": "Data"},
    "spark": {"aliases": ["apache spark", "pyspark"], "category": "Data"},
    "airflow": {"aliases": ["apache airflow"], "category": "Data"},
    # --- Cloud & DevOps ---
    "aws": {"aliases": ["amazon web services", "ec2", "s3", "lambda"], "category": "Cloud"},
    "gcp": {"aliases": ["google cloud", "google cloud platform"], "category": "Cloud"},
    "azure": {"aliases": ["microsoft azure"], "category": "Cloud"},
    "docker": {"aliases": [], "category": "DevOps"},
    "kubernetes": {"aliases": ["k8s", "kube", "k8"], "category": "DevOps"},
    "terraform": {"aliases": [], "category": "DevOps"},
    "ansible": {"aliases": [], "category": "DevOps"},
    "jenkins": {"aliases": [], "category": "DevOps"},
    "github actions": {"aliases": ["gha"], "category": "DevOps"},
    "gitlab ci": {"aliases": ["gitlab"], "category": "DevOps"},
    "ci/cd": {"aliases": ["cicd", "ci cd"], "category": "DevOps"},
    "linux": {"aliases": [], "category": "DevOps"},
    "nginx": {"aliases": [], "category": "DevOps"},
    "prometheus": {"aliases": [], "category": "DevOps"},
    "grafana": {"aliases": [], "category": "DevOps"},
    "serverless": {"aliases": [], "category": "Cloud"},
    # --- Databases ---
    "mysql": {"aliases": [], "category": "Databases"},
    "postgresql": {"aliases": ["postgres", "psql"], "category": "Databases"},
    "mongodb": {"aliases": ["mongo"], "category": "Databases"},
    "redis": {"aliases": [], "category": "Databases"},
    "elasticsearch": {"aliases": ["elastic search", "es"], "category": "Databases"},
    "cassandra": {"aliases": [], "category": "Databases"},
    "dynamodb": {"aliases": ["dynamo db"], "category": "Databases"},
    "oracle": {"aliases": ["oracle db"], "category": "Databases"},
    "sql server": {"aliases": ["mssql", "microsoft sql server"], "category": "Databases"},
    "sqlite": {"aliases": [], "category": "Databases"},
    "neo4j": {"aliases": [], "category": "Databases"},
    "snowflake": {"aliases": [], "category": "Databases"},
    "bigquery": {"aliases": ["bq"], "category": "Databases"},
    "redshift": {"aliases": [], "category": "Databases"},
    "sql": {"aliases": [], "category": "Databases"},
    "nosql": {"aliases": ["no sql"], "category": "Databases"},
    # --- Testing / Tools / Others ---
    "git": {"aliases": [], "category": "Tools"},
    "github": {"aliases": [], "category": "Tools"},
    "gitlab": {"aliases": [], "category": "Tools"},
    "jira": {"aliases": [], "category": "Tools"},
    "agile": {"aliases": ["scrum", "kanban"], "category": "Methodology"},
    "rest api": {"aliases": ["rest", "restful api"], "category": "Tools"},
    "graphql": {"aliases": [], "category": "Tools"},
    "grpc": {"aliases": [], "category": "Tools"},
    "microservices": {"aliases": ["micro services"], "category": "Architecture"},
    "system design": {"aliases": [], "category": "Architecture"},
    "figma": {"aliases": [], "category": "Tools"},
    "selenium": {"aliases": [], "category": "Testing"},
    "pytest": {"aliases": [], "category": "Testing"},
    "jest": {"aliases": [], "category": "Testing"},
    "cypress": {"aliases": [], "category": "Testing"},
    "unit testing": {"aliases": [], "category": "Testing"},
}

# Build normalized lookup: alias -> canonical and canonical -> canonical (self)
# All keys lowercased and punctuation-normalized for matching
_PUNCT_MAP = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _normalize_phrase(phrase: str) -> str:
    """Lowercase, punctuation->space, collapse whitespace for lookup."""
    phrase = phrase.lower().strip()
    phrase = phrase.translate(_PUNCT_MAP)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


# Precompute two structures:
# 1. ALIAS_TO_CANONICAL: normalized alias -> canonical (original casing lower)
# 2. CANONICAL_SET: set of normalized canonical skills for direct match
# 3. MAX_NGRAM: longest skill token length (for n-gram generation)
ALIAS_TO_CANONICAL: Dict[str, str] = {}
CANONICAL_TO_CATEGORY: Dict[str, str] = {}
CANONICAL_SET: Set[str] = set()

for canonical, meta in _SKILL_ONTOLOGY_RAW.items():
    norm_canon = _normalize_phrase(canonical)
    CANONICAL_SET.add(norm_canon)
    ALIAS_TO_CANONICAL[norm_canon] = canonical.lower()
    CANONICAL_TO_CATEGORY[canonical.lower()] = meta["category"]
    for alias in meta.get("aliases", []):
        norm_alias = _normalize_phrase(alias)
        if norm_alias:
            # Alias maps to canonical lowercased form
            ALIAS_TO_CANONICAL[norm_alias] = canonical.lower()

# For quick introspection, expose ontology list (canonical lowercased)
SKILL_ONTOLOGY: List[str] = sorted({v for v in ALIAS_TO_CANONICAL.values()})
# Also expose category-grouped view
SKILL_BY_CATEGORY: Dict[str, List[str]] = {}
for canon_lower, cat in CANONICAL_TO_CATEGORY.items():
    SKILL_BY_CATEGORY.setdefault(cat, []).append(canon_lower)
for cat in SKILL_BY_CATEGORY:
    SKILL_BY_CATEGORY[cat] = sorted(SKILL_BY_CATEGORY[cat])

MAX_NGRAM_LEN = max(len(s.split()) for s in ALIAS_TO_CANONICAL.keys()) if ALIAS_TO_CANONICAL else 3
# Cap at 3 for performance but allow 4 if ontology needs (e.g., "natural language processing"=3)
MAX_NGRAM_LEN = max(1, min(MAX_NGRAM_LEN, 4))


# ---------------------------------------------------------------------------
# 2. Extraction Function
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9\+\#\.\-]+", re.IGNORECASE)


def _tokenize_for_extraction(text: str) -> List[str]:
    """
    Tokenize for ontology matching:
    - Lowercase
    - Keep alphanumerics + +# . - (for c++, c#, .net, node.js)
    - Split on whitespace/punctuation otherwise
    """
    if not text or not isinstance(text, str):
        return []
    # Normalize punctuation that separates tokens: keep +#. for skill chars,
    # but replace other punct with space. We handle c++ / c# / .net specially.
    # Approach: lower, then extract via regex that includes those chars.
    text_lower = text.lower()
    # Temporarily protect "c++" and "c#" and ".net" / "asp.net"
    # Regex [a-z0-9\+\#\.\-]+ will capture them as tokens including symbols
    tokens = _TOKEN_RE.findall(text_lower)
    # Filter very short noise except known single-char skills handled via alias?
    # Keep tokens length >=1 but later n-gram will handle.
    return tokens


def extract_skills(
    text: str,
    ontology: Optional[Set[str]] = None,
    *,
    use_aliases: bool = True,
    ngram_range: tuple[int, int] = (1, 3),
) -> List[str]:
    """
    Extract canonical skills from free-form text by matching tokens and n-grams
    against the ontology.

    Args:
        text: Raw JD or resume text
        ontology: Optional custom set of canonical skills (lowercased).
                  If None, uses global SKILL_ONTOLOGY.
        use_aliases: If True (default), alias -> canonical resolution enabled.
                     If False, only direct canonical match.
        ngram_range: Range of n-grams to consider (min, max). Default (1,3).

    Returns:
        Sorted list of unique canonical skills (lowercased) found in text.
        Empty list if no matches or empty input.

    Example:
        >>> extract_skills("Experienced in Python, React and AWS EC2 with k8s")
        ['aws', 'kubernetes', 'python', 'react']
        >>> extract_skills("We need someone with machine learning and ml exposure")
        ['machine learning']
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    tokens = _tokenize_for_extraction(text)
    if not tokens:
        return []

    # Build lookup choice
    if use_aliases:
        lookup = ALIAS_TO_CANONICAL
    else:
        # Only canonical self-mapping
        lookup = {c: c for c in CANONICAL_SET}

    # Generate n-grams (1 to max). We do greedy longest-match first to prefer
    # multi-word skills ("machine learning") over unigrams ("machine")
    min_n, max_n = ngram_range
    max_n = min(max_n, MAX_NGRAM_LEN, len(tokens))
    found: Set[str] = set()
    matched_positions: Set[int] = set()  # to avoid subsumed matches, but not strictly required

    # Create normalized n-gram strings to match alias dict.
    # Note: tokens already lowercased; we join with space for normalized form.
    # For alias dict, we normalized punctuation->space, so joining with space aligns.
    for n in range(max_n, min_n - 1, -1):
        for i in range(len(tokens) - n + 1):
            # Greedy longest-match: skip n-gram if all its token positions
            # were already covered by a longer match (prevents "spring" from
            # also firing when "spring boot" was already matched)
            if all(pos in matched_positions for pos in range(i, i + n)):
                continue
            ngram = " ".join(tokens[i : i + n])
            norm_ngram = _normalize_phrase(ngram)
            if not norm_ngram:
                continue
            canonical = lookup.get(norm_ngram)
            if canonical:
                # For cases like "python3" tokenized as "python3" -> normalized "python3"
                # alias map should have it.
                found.add(canonical)
                # Mark positions as matched (so smaller n won't create noise)
                for pos in range(i, i + n):
                    matched_positions.add(pos)

    # Additional direct substring fallback for skills that contain punctuation
    # not captured by token regex perfectly (e.g., "ci/cd" -> tokens "ci", "cd")
    # Alias "ci/cd" normalized to "ci cd", so "ci cd" ngram (2) will match above.
    # Similarly "c++" vs "cpp": tokens "c" - not ideal. Handle special case:
    text_norm_full = _normalize_phrase(text)
    for alias_norm, canonical in lookup.items():
        if " " not in alias_norm and canonical in found:
            continue
        # If alias is a single word that is already a token inside a longer
        # found skill that appears in the text (e.g., "spring" inside
        # "spring boot"), skip fallback to avoid double-counting subsumed matches
        if " " not in alias_norm and found:
            is_subsumed = False
            for found_skill in found:
                # found_skill is canonical lower; check if alias is a word in it
                if alias_norm in found_skill.split():
                    # and that longer skill actually appears in normalized text
                    if _normalize_phrase(found_skill) in text_norm_full:
                        is_subsumed = True
                        break
                # also check alias==found_skill token case handled above
            if is_subsumed:
                continue
        # For multi-word or symbol skills that might have been missed due to tokenization edge
        # Check if alias phrase appears as substring in normalized full text
        # This helps for "c++" (normalized "c") vs "cpp" alias? Actually "c++" normalized to "c"
        # So we keep alias list for c++ including "cpp".
        if alias_norm in text_norm_full:
            # Ensure it's a whole-word/phrase match, not partial inside unrelated word
            # Use regex word boundaries
            pattern = r"(^| )" + re.escape(alias_norm) + r"( |$)"
            if re.search(pattern, text_norm_full):
                found.add(canonical)

    return sorted(found)


# ---------------------------------------------------------------------------
# 3. Gap Detection
# ---------------------------------------------------------------------------


def get_missing_skills(
    resume_skills: Iterable[str],
    job_skills: Iterable[str],
) -> List[str]:
    """
    Gap detection: skills required by job but missing from resume.

    Both inputs are assumed to be canonical lowercased lists (as returned by
    extract_skills), but the function normalizes casing/whitespace defensively.

    Args:
        resume_skills: Skills extracted from resume (canonical)
        job_skills: Skills extracted from job description (canonical)

    Returns:
        Sorted list of missing canonical skills (job - resume).

    Example:
        >>> get_missing_skills(['python', 'sql'], ['python', 'aws', 'docker'])
        ['aws', 'docker']
    """
    if job_skills is None or resume_skills is None:
        return sorted({s.lower().strip() for s in (job_skills or []) if isinstance(s, str) and s.strip()})

    # Normalize to lower + strip + deduplicate
    resume_set = {s.lower().strip() for s in resume_skills if isinstance(s, str) and s.strip()}
    job_set = {s.lower().strip() for s in job_skills if isinstance(s, str) and s.strip()}

    # Alias-resolve both to canonical for fair comparison (e.g., "k8s" vs "kubernetes")
    def resolve_to_canonical(s: str) -> str:
        norm = _normalize_phrase(s)
        return ALIAS_TO_CANONICAL.get(norm, s.lower().strip())

    resume_canonical = {resolve_to_canonical(s) for s in resume_set}
    job_canonical = {resolve_to_canonical(s) for s in job_set}

    missing = job_canonical - resume_canonical
    return sorted(missing)


def get_matched_skills(
    resume_skills: Iterable[str],
    job_skills: Iterable[str],
) -> List[str]:
    """Intersection: skills present in both resume and job."""
    resume_set = {s.lower().strip() for s in resume_skills if isinstance(s, str) and s.strip()}
    job_set = {s.lower().strip() for s in job_skills if isinstance(s, str) and s.strip()}

    def resolve(s: str) -> str:
        return ALIAS_TO_CANONICAL.get(_normalize_phrase(s), s.lower().strip())

    resume_canonical = {resolve(s) for s in resume_set}
    job_canonical = {resolve(s) for s in job_set}
    return sorted(job_canonical & resume_canonical)


def get_extra_skills(
    resume_skills: Iterable[str],
    job_skills: Iterable[str],
) -> List[str]:
    """Skills in resume but not required by job (candidate strengths beyond JD)."""
    resume_set = {s.lower().strip() for s in resume_skills if isinstance(s, str) and s.strip()}
    job_set = {s.lower().strip() for s in job_skills if isinstance(s, str) and s.strip()}

    def resolve(s: str) -> str:
        return ALIAS_TO_CANONICAL.get(_normalize_phrase(s), s.lower().strip())

    resume_canonical = {resolve(s) for s in resume_set}
    job_canonical = {resolve(s) for s in job_set}
    return sorted(resume_canonical - job_canonical)


def skill_gap_analysis(
    resume_text: str,
    job_text: str,
) -> Dict[str, List[str] | float]:
    """
    Convenience wrapper: extract from raw texts and compute full gap analysis.

    Returns dict with:
      - job_skills, resume_skills, matched_skills, missing_skills, extra_skills
      - skill_match_score: |matched| / |job_skills| *100 (0 if job has no skills)

    Example:
        >>> analysis = skill_gap_analysis("Python and AWS", "Python AWS Docker")
        >>> analysis["missing_skills"]
        ['docker']
    """
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    matched = get_matched_skills(resume_skills, job_skills)
    missing = get_missing_skills(resume_skills, job_skills)
    extra = get_extra_skills(resume_skills, job_skills)

    if job_skills:
        score = round(len(matched) / len(job_skills) * 100.0, 2)
    else:
        score = 0.0

    return {
        "job_skills": job_skills,
        "resume_skills": resume_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "skill_match_score": score,
    }


# ---------------------------------------------------------------------------
# Ontology helpers for API / taxonomy endpoint
# ---------------------------------------------------------------------------


def get_ontology() -> List[str]:
    """Return sorted list of all canonical skills (lowercased)."""
    return SKILL_ONTOLOGY.copy()


def get_ontology_by_category() -> Dict[str, List[str]]:
    """Return category -> skills mapping."""
    return {k: v.copy() for k, v in SKILL_BY_CATEGORY.items()}


def get_taxonomy_metadata() -> Dict:
    """Return taxonomy metadata for API (count, version)."""
    return {
        "skills": get_ontology(),
        "count": len(SKILL_ONTOLOGY),
        "by_category": get_ontology_by_category(),
        "version": "2026-09",
    }


__all__ = [
    "SKILL_ONTOLOGY",
    "SKILL_BY_CATEGORY",
    "ALIAS_TO_CANONICAL",
    "CANONICAL_SET",
    "MAX_NGRAM_LEN",
    "extract_skills",
    "get_missing_skills",
    "get_matched_skills",
    "get_extra_skills",
    "skill_gap_analysis",
    "get_ontology",
    "get_ontology_by_category",
    "get_taxonomy_metadata",
]

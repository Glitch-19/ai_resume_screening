"""
Screening router — AI Resume Screening & Job Matching System

Endpoints per task:
  POST /api/upload-resume  — PDF/DOCX -> raw text extraction
  POST /api/match          — JD + list of resumes -> ranked scores

Also aliases:
  POST /api/upload-resumes (plural) for convenience
  POST /api/screen         (legacy alias)

Wires into backend/main.py and uses services:
  - matching_engine (TF-IDF + semantic hybrid 0.4/0.6)
  - skill_extractor (ontology + gap detection)
"""

from __future__ import annotations

import io
import re
from typing import List, Optional, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

# --- Internal helpers for PDF/DOCX extraction ---

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _validate_file(file: UploadFile, file_bytes: bytes):
    """Validate mime/extension/size; raise HTTPException on failure."""
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = (file.content_type or "").lower()

    if ext not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{content_type or ext}'. Only PDF and DOCX are allowed.",
        )
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Max {MAX_FILE_SIZE_MB}MB allowed.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes via pypdf with pdfplumber fallback."""
    text_parts: List[str] = []
    # Attempt 1: pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
            except Exception:
                continue
        text = "\n".join(text_parts).strip()
        # If extraction is suspiciously short but PDF has pages, try pdfplumber
        if len(text) < 50 and len(text_parts) == 0 or (len(text) < 50 and len(reader.pages) > 0):
            # fall through to pdfplumber attempt below
            pass
        else:
            if text:
                return text
    except Exception:
        # Ignore and try fallback
        pass

    # Attempt 2: pdfplumber fallback (better for layout/2-col PDFs)
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            fallback_parts = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    fallback_parts.append(t)
            fallback_text = "\n".join(fallback_parts).strip()
            if fallback_text:
                # Prefer longer extraction
                if len(fallback_text) > len("\n".join(text_parts).strip()):
                    return fallback_text
    except Exception:
        pass

    # Return whatever we got from pypdf (may be empty)
    return "\n".join(text_parts).strip()


def _extract_docx_text(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes via python-docx (paragraphs + tables)."""
    try:
        from docx import Document  # type: ignore

        doc = Document(io.BytesIO(file_bytes))
        parts: List[str] = []
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                parts.append(para.text.strip())
        # Include tables text
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    parts.append(row_text)
        return "\n".join(parts).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse DOCX: {exc}") from exc


def _infer_candidate_name(filename: str, raw_text: str) -> str:
    """
    Heuristic candidate name inference:
    - Use filename stem (without extension) as fallback
    - Try to extract first line that looks like a name (2 words, capitalized)
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    # Clean stem: replace _ - with space, title case
    stem_clean = re.sub(r"[_\-]+", " ", stem).strip()
    # If raw_text first non-empty line looks like a name (e.g., "John Doe"), use it
    if raw_text:
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        if lines:
            first = lines[0]
            # Heuristic: 2-3 words, mostly letters, each capitalized
            words = first.split()
            if 1 < len(words) <= 4 and all(w[0].isalpha() and w[0].isupper() for w in words if w):
                # Avoid generic headers like "Resume" / "Curriculum Vitae"
                if first.lower() not in {"resume", "curriculum vitae", "cv"}:
                    # Keep first line if it contains a name-like pattern and not too long
                    if len(first) < 50:
                        return first.strip()
    # Fallback to filename stem title-cased
    if stem_clean:
        return stem_clean.title()
    return "Unknown Candidate"


# --- Pydantic schemas for /match ---


class CandidateInput(BaseModel):
    candidate_name: Optional[str] = Field(default=None, description="Candidate display name")
    resume_text: str = Field(..., description="Plain text of parsed resume")
    filename: Optional[str] = Field(default=None, description="Original filename if available")


class MatchRequest(BaseModel):
    job_description: str = Field(..., min_length=1, description="Job description full text")
    # Primary field per prompt: list of parsed candidate resumes
    candidates: Optional[List[CandidateInput]] = Field(default=None, description="List of candidates with resume_text")
    # Compatibility aliases
    resumes: Optional[List[CandidateInput]] = None
    resume_texts: Optional[List[str]] = None  # simple list of texts

    def get_candidates(self) -> List[CandidateInput]:
        if self.candidates is not None:
            return self.candidates
        if self.resumes is not None:
            return self.resumes
        if self.resume_texts is not None:
            return [CandidateInput(resume_text=t) for t in self.resume_texts]
        return []


class CandidateResult(BaseModel):
    candidate_name: str
    total_score: float = Field(ge=0, le=100)
    tfidf_score: float = Field(ge=0, le=100)
    semantic_score: float = Field(ge=0, le=100)
    matching_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    # Extra helpful fields for dashboard (not required but useful)
    extra_skills: Optional[List[str]] = None


class MatchResponse(BaseModel):
    job_description_preview: str
    total_candidates: int
    results: List[CandidateResult]


# --- Router ---

router = APIRouter(prefix="/api", tags=["screening"])


@router.post("/upload-resume")
@router.post("/upload-resumes")  # alias plural
async def upload_resume(file: UploadFile = File(..., description="PDF or DOCX resume file")):
    """
    Accept PDF/DOCX and extract raw text.

    Returns:
        {
          "filename": "john_doe.pdf",
          "raw_text": "...",
          "char_count": 3421,
          "candidate_name": "John Doe",
          "extraction_confidence": 0.98
        }
    """
    file_bytes = await file.read()
    await file.close()

    _validate_file(file, file_bytes)

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Dispatch by extension (more reliable than mime)
    if ext == ".pdf":
        raw_text = _extract_pdf_text(file_bytes)
    elif ext == ".docx":
        raw_text = _extract_docx_text(file_bytes)
    else:
        # Fallback by mime
        ct = (file.content_type or "").lower()
        if "pdf" in ct:
            raw_text = _extract_pdf_text(file_bytes)
        elif "wordprocessingml" in ct or "docx" in ct:
            raw_text = _extract_docx_text(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or DOCX.")

    if not raw_text or len(raw_text.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="Failed to extract text from file. File may be scanned image PDF (OCR required) or empty.",
        )

    candidate_name = _infer_candidate_name(filename, raw_text)
    # Simple confidence heuristic: char count vs file size
    confidence = min(1.0, len(raw_text) / 2000) if raw_text else 0.0
    confidence = round(max(0.5, confidence), 2) if raw_text else 0.0

    return {
        "filename": filename,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "candidate_name": candidate_name,
        "extraction_confidence": confidence,
    }


@router.post("/match", response_model=MatchResponse)
@router.post("/screen", response_model=MatchResponse, include_in_schema=False)  # legacy alias
async def match_candidates(payload: MatchRequest):
    """
    Accept job_description + list of parsed candidate resumes.
    Return ranked list sorted by total_score descending.

    Request body example:
    {
      "job_description": "We need a Python developer with AWS, Docker...",
      "candidates": [
        {"candidate_name": "Alice", "resume_text": "Experienced in Python..."},
        {"candidate_name": "Bob", "resume_text": "Graphic designer..."}
      ]
    }

    Response: ranked results with candidate_name, total_score, tfidf_score,
              semantic_score, matching_skills, missing_skills (descending).
    """
    # Validate JD
    jd = (payload.job_description or "").strip()
    if not jd:
        raise HTTPException(status_code=422, detail="job_description is required and cannot be empty.")
    if len(jd) < 10:
        raise HTTPException(status_code=422, detail="job_description too short (min 10 characters).")

    candidates = payload.get_candidates()
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="No candidates provided. Supply 'candidates' (or 'resumes'/'resume_texts') with at least one resume_text.",
        )
    if len(candidates) > 100:
        raise HTTPException(status_code=413, detail="Too many candidates (max 100 per request).")

    # Lazy imports to avoid circular and keep cold start fast
    from backend.services.matching_engine import (
        compute_tfidf_score,
        compute_semantic_score,
        compute_hybrid_score,
    )
    from backend.services.skill_extractor import (
        extract_skills,
        get_missing_skills,
        get_matched_skills,
        get_extra_skills,
    )

    # Pre-extract JD skills once
    jd_skills = extract_skills(jd)

    results: List[CandidateResult] = []
    for idx, cand in enumerate(candidates):
        resume_text = (cand.resume_text or "").strip()
        if not resume_text:
            # Allow empty resume but score as 0
            tfidf = 0.0
            semantic = 0.0
            total = 0.0
            resume_skills: List[str] = []
            matched: List[str] = []
            missing = jd_skills  # all missing
            extra: List[str] = []
        else:
            # Hybrid scores: 0.4 TF-IDF + 0.6 Semantic per task spec
            tfidf = compute_tfidf_score(jd, resume_text)
            semantic = compute_semantic_score(jd, resume_text)
            total = compute_hybrid_score(tfidf, semantic, w_tfidf=0.4, w_semantic=0.6)

            resume_skills = extract_skills(resume_text)
            matched = get_matched_skills(resume_skills, jd_skills)
            missing = get_missing_skills(resume_skills, jd_skills)
            extra = get_extra_skills(resume_skills, jd_skills)

        # Resolve candidate_name with fallback
        name = (cand.candidate_name or "").strip()
        if not name:
            if cand.filename:
                name = _infer_candidate_name(cand.filename, resume_text)
            else:
                name = f"Candidate {idx + 1}"

        results.append(
            CandidateResult(
                candidate_name=name,
                total_score=total,
                tfidf_score=tfidf,
                semantic_score=semantic,
                matching_skills=matched,
                missing_skills=missing,
                extra_skills=extra,
            )
        )

    # Sort descending by total_score (hybrid)
    results.sort(key=lambda r: r.total_score, reverse=True)

    preview = jd[:200] + ("..." if len(jd) > 200 else "")

    return MatchResponse(
        job_description_preview=preview,
        total_candidates=len(results),
        results=results,
    )


# Health for router (optional)
@router.get("/screening/health")
async def screening_health():
    return {"status": "ok", "service": "screening"}

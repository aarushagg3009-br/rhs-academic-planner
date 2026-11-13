"""
recommender_api.py
FastAPI web server for the RHS course recommendation system.
"""
from __future__ import annotations

import json
import os
from difflib import get_close_matches
from pathlib import Path
from typing import Annotated, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from evaluator import _load, evaluate
from recommender import build_slate, build_two_year_plan, TIER_GRADE_TARGET

load_dotenv()

# ---------------------------------------------------------------------------
# Load rules once at import time
# ---------------------------------------------------------------------------

_rules_path = os.environ.get(
    "RHS_RULES_PATH",
    str(Path(__file__).parent / "RHS_POS_2026-2027_rules.json"),
)
try:
    _rules: dict = _load(_rules_path)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Rules file not found: {_rules_path}. Set RHS_RULES_PATH in .env"
    ) from exc


app = FastAPI(
    title="RHS Course Recommendation API",
    description="Prerequisite-validated course planning for Ridge High School students.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import Response, HTMLResponse

_APP_API_KEY = os.environ.get("APP_API_KEY", "")

@app.middleware("http")
async def _auth(request: Request, call_next):
    if _APP_API_KEY and request.url.path not in ("/", "/health"):
        if request.headers.get("X-API-Key") != _APP_API_KEY:
            return Response(
                '{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )
    return await call_next(request)

@app.middleware("http")
async def ngrok_skip_browser_warning(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html = (Path(__file__).parent / "frontend" / "index.html").read_text()
    return HTMLResponse(content=html)

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

CareerTrack = Literal[
    "business_finance",
    "computer_science",
    "medicine_premed",
    "engineering",
    "law_polisci",
    "humanities_history",
    "social_sciences",
    "creative_arts",
]

CollegeTier = Literal[
    "ivy_top10",
    "top20_30",
    "top50_regionals",
    "state_flagship",
    "liberal_arts",
    "community_college",
]

_VALID_GRADES = frozenset(
    {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"}
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TranscriptEntry(BaseModel):
    course_code: str = Field(..., min_length=1, description="RHS course code, e.g. '120C'")
    name: str = Field(default="", description="Course name (optional — used for display only)")
    grade: str = Field(..., description="Letter grade earned, e.g. 'B+'")
    grade_level: int = Field(..., ge=8, le=12, description="Grade level when the course was taken (8 for middle-school courses)")
    exam_taken: bool = Field(default=False, description="True if the AP exam was taken for this course")

    @field_validator("grade")
    @classmethod
    def _check_grade(cls, v: str) -> str:
        normalized = v.strip().upper()
        if normalized not in _VALID_GRADES:
            raise ValueError(
                f"'{v}' is not a valid letter grade. "
                f"Accepted values: {', '.join(sorted(_VALID_GRADES))}"
            )
        return normalized

    @field_validator("course_code")
    @classmethod
    def _normalize_code(cls, v: str) -> str:
        return v.strip().upper()


class StudentProfile(BaseModel):
    transcript: list[TranscriptEntry] = Field(
        default_factory=list,
        description="All courses the student has completed, with grades",
    )
    student_grade: int = Field(
        ...,
        ge=9,
        le=12,
        description="Grade the student is ENTERING (e.g. 10 means they are entering 10th grade)",
    )
    career_tracks: list[CareerTrack] = Field(
        default_factory=list,
        description="Up to 3 career interest areas that boost relevant course scores",
    )
    target_tier: CollegeTier = Field(
        default="state_flagship",
        description="College selectivity target — controls how heavily AP/Honors courses are weighted",
    )


class EvaluateRequest(BaseModel):
    transcript: list[TranscriptEntry] = Field(
        default_factory=list,
        description="Student's completed courses",
    )
    student_grade: int = Field(
        ...,
        ge=9,
        le=12,
        description="Grade the student is entering",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check. Returns the number of courses loaded from the rules file."""
    return {"status": "ok", "courses_loaded": len(_rules.get("courses", []))}


@app.get("/courses", tags=["catalog"])
def list_courses(
    department: Optional[str] = None,
    level: Optional[str] = None,
) -> list[dict]:
    """
    Return all courses in the catalog.

    Optional query params:
    - **department**: filter by department name (case-insensitive), e.g. `Science`
    - **level**: filter by level (case-insensitive), e.g. `AP`, `Honors`, `CP`
    """
    courses = _rules.get("courses", [])
    if department:
        courses = [c for c in courses if c.get("department", "").lower() == department.lower()]
    if level:
        courses = [c for c in courses if c.get("level", "").lower() == level.lower()]
    return [
        {
            "course_code": c["course_code"],
            "name": c.get("name", ""),
            "department": c.get("department", ""),
            "level": c.get("level", ""),
            "credits": c.get("credits", 0),
            "grade_levels": c.get("grade_levels", []),
        }
        for c in courses
    ]


@app.post("/evaluate/{course_code}", tags=["eligibility"])
def evaluate_course(
    course_code: Annotated[str, FPath(description="RHS course code, e.g. '120C'")],
    body: EvaluateRequest,
) -> dict:
    """
    Check whether a student is eligible for a specific course.

    Returns:
    - **eligible**: whether the student can take this course
    - **prereqs_met**: whether all prerequisites are satisfied
    - **grade_level_ok**: whether the course is offered at the student's grade
    - **reasons**: blocking reasons if not eligible
    - **warnings**: non-blocking notes (biannual availability, already taken, etc.)
    - **corequisites**: courses that must be taken concurrently
    - **constraints**: scheduling flags (study hall, summer work, dual enrollment, etc.)
    """
    code = course_code.strip().upper()
    tx_dicts = [e.model_dump() for e in body.transcript]
    result = evaluate(_rules, tx_dicts, code, body.student_grade)
    if result["course_name"] == "UNKNOWN":
        raise HTTPException(status_code=404, detail=f"Course code '{code}' not found in the catalog")
    return result


@app.post("/recommend", tags=["planning"])
def recommend(profile: StudentProfile) -> dict:
    """
    Build a personalized course plan for the given student profile.

    Returns the top 3 **eligible** courses per schedule slot
    (english, math, science, social_studies, world_language, elective_1, elective_2, pe_health)
    for the entering grade. If the student is in grades 9–11, a projected Year 2 slate
    is also included based on simulated completion of Year 1 core courses at a B+ grade.

    Courses are ranked by a weighted score: base level score + career track bonus × college tier multiplier.
    """
    tx_dicts = [e.model_dump() for e in profile.transcript]

    target_grade = TIER_GRADE_TARGET.get(profile.target_tier, "B+")

    if profile.student_grade == 12:
        slate = build_slate(
            _rules, tx_dicts, 12, profile.career_tracks, profile.target_tier
        )
        result: dict[str, list[dict]] = {}
        for slot, recs in slate.items():
            eligible = [r for r in recs if r.eligible]
            annotated = []
            for r in eligible[:3]:
                d = r.to_dict()
                if annotated == []:  # first = rank 1
                    d["suggested_grade"] = target_grade
                annotated.append(d)
            result[slot] = annotated
        return {"grade_12": result}

    plan = build_two_year_plan(
        _rules,
        tx_dicts,
        profile.student_grade,
        profile.career_tracks,
        profile.target_tier,
    )
    # plan already has suggested_grade on rank-1 Year 1 courses from build_two_year_plan.
    # Filter to eligible only (top 3) for each slot and year.
    first_year, *rest_years = plan.keys()
    filtered: dict = {}
    for year, slots in plan.items():
        filtered[year] = {
            slot: [c for c in courses if c["eligible"]][:3]
            for slot, courses in slots.items()
        }
    return filtered


@app.post("/recommend/full", tags=["planning"])
def recommend_full(profile: StudentProfile) -> dict:
    """
    Like `/recommend` but returns **all** courses per slot — both eligible and blocked —
    sorted by score descending. Blocked courses include their `blocking_reasons` so a UI
    can show the student what they need to unlock each course.
    """
    tx_dicts = [e.model_dump() for e in profile.transcript]

    if profile.student_grade == 12:
        slate = build_slate(
            _rules, tx_dicts, 12, profile.career_tracks, profile.target_tier
        )
        return {
            "grade_12": {
                slot: [r.to_dict() for r in recs]
                for slot, recs in slate.items()
            }
        }

    return build_two_year_plan(
        _rules,
        tx_dicts,
        profile.student_grade,
        profile.career_tracks,
        profile.target_tier,
    )


# ---------------------------------------------------------------------------
# Transcript PDF parsing
# ---------------------------------------------------------------------------

def _match_code(name: str, rules: dict) -> str:
    """Fuzzy-match a course name to the closest RHS course code."""
    catalog = {c.get("name", "").lower(): c["course_code"] for c in rules.get("courses", [])}
    hits = get_close_matches(name.lower(), catalog.keys(), n=1, cutoff=0.55)
    return catalog[hits[0]] if hits else ""


_PARSE_PROMPT = (
    "The attached document is a student transcript. "
    "Extract ONLY the courses that are explicitly printed in this document — "
    "do NOT add any course that you do not see on the page.\n\n"
    "Return a JSON array. Each element must have exactly these fields:\n"
    '  "name": the course name exactly as printed,\n'
    '  "grade": the final letter grade earned (e.g. "A", "B+", "C-"),\n'
    '  "grade_level": the grade year shown in the transcript section header '
    '(e.g. "9th Grade" → 9). Use the printed heading only — never guess from the course name,\n'
    '  "exam_taken": true only if an AP exam score is explicitly shown for this course.\n\n'
    "Return ONLY a valid JSON array with no markdown and no explanation."
)


class _PDFFile(BaseModel):
    name: str
    data: str  # base64-encoded PDF bytes


class ParseTranscriptRequest(BaseModel):
    files: list[_PDFFile] = Field(..., min_length=1)


@app.post("/parse-transcript", tags=["transcript"])
def parse_transcript(body: ParseTranscriptRequest) -> list[dict]:
    """
    Parse one or more base64-encoded PDF transcripts with Claude and return
    a deduplicated list of TranscriptEntry-compatible dicts.

    Requires ANTHROPIC_API_KEY in the environment.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set. Add it to your .env file.",
        )

    all_courses: list[dict] = []

    for f in body.files:
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": f.data,
                                    },
                                },
                                {"type": "text", "text": _PARSE_PROMPT},
                            ],
                        }
                    ],
                },
                timeout=60.0,
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Network error calling Claude API: {exc}")

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Claude API returned {resp.status_code} for '{f.name}': {resp.text[:300]}",
            )

        raw = resp.json()["content"][0]["text"].strip()
        # Strip markdown code fences if Claude includes them
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        try:
            courses: list[dict] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not parse Claude response for '{f.name}': {exc}",
            )

        # Server-side code matching — Claude only extracted names, we resolve codes
        for c in courses:
            printed_code = str(c.get("course_code", "") or "").strip().upper()
            matched = _match_code(c.get("name", ""), _rules)
            c["course_code"] = matched.upper() if matched else printed_code

        all_courses.extend(courses)

    # Deduplicate by course code, last-write-wins
    seen: dict[str, dict] = {}
    for c in all_courses:
        code = str(c.get("course_code", "")).strip().upper()
        if code:
            c["course_code"] = code
            seen[code] = c

    return list(seen.values())

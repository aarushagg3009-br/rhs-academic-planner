"""
recommender.py
Course recommendation engine for RHS academic planning.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from evaluator import evaluate, _load, resolve_course_ref, Transcript

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get(
    "RHS_RECOMMENDER_CONFIG",
    str(Path(__file__).parent / "recommender_config.json"),
)


def _load_config() -> dict:
    try:
        return json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


_CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CourseRecommendation:
    course_code: str
    name: str
    level: str
    credits: float
    department: str
    slot: str
    score: float
    reason: str
    eligible: bool
    blocking_reasons: list[str] = field(default_factory=list)
    corequisites: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "course_code": self.course_code,
            "name": self.name,
            "level": self.level,
            "credits": self.credits,
            "department": self.department,
            "slot": self.slot,
            "score": round(self.score, 2),
            "reason": self.reason,
            "eligible": self.eligible,
            "blocking_reasons": self.blocking_reasons,
            "corequisites": self.corequisites,
            "warnings": self.warnings,
            "constraints": self.constraints,
        }


# ---------------------------------------------------------------------------
# Slot / department mapping
# ---------------------------------------------------------------------------

SLOTS = [
    "english",
    "math",
    "science",
    "social_studies",
    "world_language",
    "elective_1",
    "elective_2",
    "pe_health",
]

# fulfills value → slot (used for courses that satisfy a graduation requirement)
_FULFILLS_TO_SLOT: dict[str, str] = {
    "english":                  "english",
    "math":                     "math",
    "biology_requirement":      "science",
    "chemistry_requirement":    "science",
    "physics_requirement":      "science",
    "social_studies_global":    "social_studies",
    "social_studies_us_history":"social_studies",
    "social_studies_gov":       "social_studies",
    "world_language":           "world_language",
    "visual_performing_arts":   "elective_1",
    "career_education_21c":     "elective_2",
    "financial_literacy":       "elective_2",
    "pe_health":                "pe_health",
}

# Fallback by department for courses with no graduation-requirement fulfills.
# ELA/Math electives (AP Seminar, AP CS A, etc.) go to elective slots.
# Advanced Science/SS still occupy that year's subject slot.
_NO_FULFILLS_DEPT_SLOT: dict[str, str] = {
    "English Language Arts": "elective_1",
    "Mathematics":           "elective_2",
    "Science":               "science",
    "Social Studies":        "social_studies",
    "Career Education":      "elective_2",
    "Visual Arts":           "elective_1",
    "Performing Arts":       "elective_1",
    "World Languages":       "world_language",
}


def _slot_for_course(course: dict) -> str:
    """Return the schedule slot for a course using fulfills first, then department."""
    for f in course.get("constraints", {}).get("fulfills", []):
        slot = _FULFILLS_TO_SLOT.get(f)
        if slot:
            return slot
    return _NO_FULFILLS_DEPT_SLOT.get(course.get("department", ""), "elective_1")


# ---------------------------------------------------------------------------
# Sequencing helpers
# ---------------------------------------------------------------------------

_LEVEL_RANK: dict[str, int] = {"AP": 4, "Honors": 3, "Enriched": 2, "CP": 1}

# Only LATERAL fulfills categories are deduplicated by level (pick one from
# the group).  Sequential categories (math, english, world_language) span the
# full course progression and must NOT be skipped by this mechanism.
_LATERAL_FULFILLS: frozenset[str] = frozenset({
    "biology_requirement",
    "chemistry_requirement",
    "physics_requirement",
    "social_studies_global",
    "social_studies_us_history",
    "social_studies_gov",
    "visual_performing_arts",
    "career_education_21c",
    "financial_literacy",
    "pe_health",
})


def _fulfilled_categories(rules: dict, completed_codes: set[str]) -> dict[str, int]:
    """
    Return {fulfills_category: max_level_rank} for every LATERAL graduation-
    requirement category the student has already satisfied.  Used to skip
    lower-level variants (e.g. CP Biology when Honors Biology is done).
    Sequential categories (math/english/world_language) are excluded.
    """
    course_map = {c["course_code"]: c for c in rules.get("courses", [])}
    result: dict[str, int] = {}
    for code in completed_codes:
        course = course_map.get(code, {})
        rank = _LEVEL_RANK.get(course.get("level", "CP"), 1)
        for f in course.get("constraints", {}).get("fulfills", []):
            if f in _LATERAL_FULFILLS and rank > result.get(f, 0):
                result[f] = rank
    return result


def _implicit_completions(rules: dict, completed_codes: set[str]) -> set[str]:
    """
    Walk prerequisite trees of completed courses and return codes of courses
    the student must have already passed.

    Handles:
    - Direct completed / min_grade nodes  (e.g. Latin II → infers Latin I)
    - completed_group / min_grade_group   (e.g. Algebra II requires
      geometry_or_higher → infers CP Geometry → infers Algebra I)
    - all_of chains recursively
    - any_of is skipped (can't know which branch was satisfied)

    Multiple passes resolve transitive chains.
    """
    course_map    = {c["course_code"]: c for c in rules.get("courses", [])}
    course_groups = rules.get("course_groups", {})
    implied: set[str] = set()

    current_known = set(completed_codes)
    for _ in range(4):
        new_implied: set[str] = set()
        for code in current_known:
            course = course_map.get(code)
            if course:
                _collect_prereq_codes(
                    course.get("prerequisites", {}),
                    new_implied,
                    course_map,
                    course_groups,
                )
        added = new_implied - current_known
        if not added:
            break
        implied       |= added
        current_known |= added

    return implied


def _collect_prereq_codes(
    node: dict,
    result: set[str],
    course_map: dict,
    course_groups: dict,
) -> None:
    t = node.get("type", "none")

    if t in ("completed", "min_grade"):
        result.update(node.get("course_codes", []))

    elif t in ("completed_group", "min_grade_group"):
        group_codes = course_groups.get(node.get("group", ""), [])
        if group_codes:
            # Take the lowest-level course in the group as a conservative proxy.
            # e.g. "geometry_or_higher" → CP Geometry → its prereq Algebra I
            min_code = min(
                group_codes,
                key=lambda c: _LEVEL_RANK.get(course_map.get(c, {}).get("level", "CP"), 1),
            )
            result.add(min_code)
            min_course = course_map.get(min_code, {})
            if min_course:
                _collect_prereq_codes(
                    min_course.get("prerequisites", {}),
                    result,
                    course_map,
                    course_groups,
                )

    elif t == "all_of":
        for child in node.get("conditions", []):
            _collect_prereq_codes(child, result, course_map, course_groups)


# ---------------------------------------------------------------------------
# Language continuity
# ---------------------------------------------------------------------------

# Maps the first two digits of a world-language course code to a family tag.
_LANG_PREFIX: dict[str, str] = {
    "41": "french",
    "43": "latin",
    "44": "spanish",
    "49": "asl",
}


def _language_family(course_code: str) -> str | None:
    """Return the language family for a world-language course code, or None."""
    digits = "".join(c for c in course_code if c.isdigit())
    return _LANG_PREFIX.get(digits[:2]) if len(digits) >= 2 else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

LEVEL_BASE: dict[str, float] = {
    "AP": 100.0,
    "Honors": 80.0,
    "Enriched": 60.0,
    "CP": 40.0,
}

TIER_GRADE_TARGET: dict[str, str] = {
    "ivy_top10":        "A",
    "top20_30":         "A-",
    "top50_regionals":  "B+",
    "state_flagship":   "B+",
    "liberal_arts":     "B+",
    "community_college":"B",
}


def _career_bonus(course_code: str, course: dict, career_tracks: list[str]) -> float:
    bonuses: dict = _CONFIG.get("career_track_bonuses", {})
    total = 0.0
    for track in career_tracks:
        track_map = bonuses.get(track, {})
        total += track_map.get(course_code, 0.0)
        dept_key = f"_dept_{course.get('department', '')}"
        total += track_map.get(dept_key, 0.0)
    return total


def _tier_multiplier(level: str, target_tier: str) -> float:
    multipliers: dict = _CONFIG.get("tier_ap_multiplier", {})
    base = multipliers.get(target_tier, 1.0)
    if level in ("AP", "Honors"):
        return base
    return 1.0


def _score_course(course: dict, career_tracks: list[str], target_tier: str) -> float:
    base = LEVEL_BASE.get(course.get("level", "CP"), 40.0)
    career_b = _career_bonus(course["course_code"], course, career_tracks)
    mult = _tier_multiplier(course.get("level", "CP"), target_tier)
    return (base + career_b) * mult


# ---------------------------------------------------------------------------
# Human-readable rationale
# ---------------------------------------------------------------------------

def _build_reason(
    course: dict,
    career_tracks: list[str],
    eval_result: dict,
) -> str:
    bonuses: dict = _CONFIG.get("career_track_bonuses", {})
    code = course["course_code"]
    matched = [t for t in career_tracks if code in bonuses.get(t, {})]

    parts: list[str] = []
    if matched:
        track_labels = [t.replace("_", " ").title() for t in matched]
        parts.append(f"Directly supports your {', '.join(track_labels)} goals")
    level = course.get("level", "CP")
    if level == "AP":
        parts.append("AP rigor signals readiness for college-level work")
    elif level == "Honors":
        parts.append("Honors level strengthens your academic profile")

    dept = course.get("department", "")
    if not parts:
        parts.append(f"Required {dept} course for graduation or academic progression")

    if eval_result.get("constraints", {}).get("dual_enrollment"):
        inst = eval_result["constraints"]["dual_enrollment"]
        parts.append(f"Dual enrollment credit available ({inst})")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# AP limit enforcement
# ---------------------------------------------------------------------------

def _ap_count_in_schedule(schedule: list[CourseRecommendation]) -> int:
    return sum(1 for r in schedule if r.level == "AP")


def _study_hall_needed(schedule: list[CourseRecommendation], rules: dict) -> bool:
    triggers = set(rules.get("scheduling_rules", {}).get("study_hall_triggers", []))
    return any(r.course_code in triggers for r in schedule)


# ---------------------------------------------------------------------------
# Core recommendation functions
# ---------------------------------------------------------------------------

def build_slate(
    rules: dict,
    transcript: list[dict],
    student_grade: int,
    career_tracks: list[str],
    target_tier: str,
    current_schedule: list[str] | None = None,
) -> dict[str, list[CourseRecommendation]]:
    """
    Build a prioritised course slate for *student_grade*.

    Returns {slot: [CourseRecommendation, ...]} sorted by score descending.
    Each slot contains BOTH eligible and ineligible courses so the caller
    can show alternatives. Filter by .eligible == True for the top picks.
    """
    if current_schedule is None:
        current_schedule = []

    tx = Transcript(transcript)
    results: dict[str, list[CourseRecommendation]] = {s: [] for s in SLOTS}

    ap_limit = rules.get("scheduling_rules", {}).get("ap_limits_by_grade", {})
    grade_ap_cap = ap_limit.get(str(student_grade))  # None means no cap

    # Pre-compute sequencing helpers once, not per-course
    completed = tx.completed_codes()
    implied   = _implicit_completions(rules, completed)
    fulfilled = _fulfilled_categories(rules, completed)
    all_done  = completed | implied

    # Detect which language families the student has already started
    started_languages: set[str] = set()
    for code in all_done:
        fam = _language_family(code)
        if fam:
            started_languages.add(fam)

    for course in rules.get("courses", []):
        code = course["course_code"]

        # Skip already-completed or implicitly completed (e.g. Latin I when Latin II is done)
        if code in all_done:
            continue

        # Skip courses not available at this grade level
        if student_grade not in course.get("grade_levels", [9, 10, 11, 12]):
            continue

        # Skip lateral/lower-level variants already covered by a completed course
        # (e.g. CP Biology when Honors Biology is done — same fulfills category, lower rank)
        course_fulfills = course.get("constraints", {}).get("fulfills", [])
        course_rank = _LEVEL_RANK.get(course.get("level", "CP"), 1)
        if any(fulfilled.get(f, 0) >= course_rank for f in course_fulfills):
            continue

        eval_result = evaluate(rules, transcript, code, student_grade)
        slot = _slot_for_course(course)
        score = _score_course(course, career_tracks, target_tier)

        # Language continuity bonus — strongly prefer continuing a started language
        # over starting a new one (50 pts sits between CP=40 and Honors=80 bases)
        fam = _language_family(code)
        if fam and started_languages and fam in started_languages:
            score += 50
        elif fam and started_languages and fam not in started_languages:
            score -= 20

        # Downrank ineligible courses
        if not eval_result["eligible"]:
            score *= 0.1

        rec = CourseRecommendation(
            course_code=code,
            name=course.get("name", code),
            level=course.get("level", "CP"),
            credits=course.get("credits", 5),
            department=course.get("department", ""),
            slot=slot,
            score=score,
            reason=_build_reason(course, career_tracks, eval_result),
            eligible=eval_result["eligible"],
            blocking_reasons=eval_result["reasons"],
            corequisites=eval_result["corequisites"],
            warnings=eval_result["warnings"],
            constraints=eval_result["constraints"],
        )
        results[slot].append(rec)

    # Sort each slot descending by score
    for slot in results:
        results[slot].sort(key=lambda r: r.score, reverse=True)

    # Enforce per-grade AP cap across slots (core slots get priority)
    if grade_ap_cap is not None:
        ap_used = 0
        for slot in SLOTS:
            for rec in results[slot]:
                if not rec.eligible or rec.level != "AP":
                    continue
                if ap_used < grade_ap_cap:
                    ap_used += 1
                else:
                    rec.eligible = False
                    rec.blocking_reasons = [
                        f"Grade {student_grade} AP course limit of {grade_ap_cap} already reached"
                    ] + rec.blocking_reasons

    return results


def build_two_year_plan(
    rules: dict,
    transcript: list[dict],
    student_grade: int,
    career_tracks: list[str],
    target_tier: str,
) -> dict:
    """
    Return recommended slates for *student_grade* AND *student_grade + 1*.

    Year 1 top courses are annotated with a suggested_grade drawn from
    TIER_GRADE_TARGET so the student knows what to aim for.
    Year 2 is simulated by assuming the student took the #1 eligible course
    per core slot and achieved that suggested grade.
    """
    target_grade = TIER_GRADE_TARGET.get(target_tier, "B+")
    year1 = build_slate(rules, transcript, student_grade, career_tracks, target_tier)

    # Simulate Year 2 using tier-appropriate grade for ALL slots so
    # courses taken in Year 1 (including electives) are not repeated.
    simulated = list(transcript)
    for slot in SLOTS:
        eligible = [r for r in year1.get(slot, []) if r.eligible]
        if eligible:
            top = eligible[0]
            simulated.append({
                "course_code": top.course_code,
                "name": top.name,
                "grade": target_grade,
                "grade_level": student_grade,
            })

    year2 = build_slate(rules, simulated, student_grade + 1, career_tracks, target_tier)

    # Annotate the #1 eligible course in each Year 1 slot with suggested_grade
    year1_dicts: dict[str, list[dict]] = {}
    for slot, recs in year1.items():
        eligible = [r for r in recs if r.eligible]
        annotated: list[dict] = []
        for i, r in enumerate(recs):
            d = r.to_dict()
            if r.eligible and eligible and r is eligible[0]:
                d["suggested_grade"] = target_grade
            annotated.append(d)
        year1_dicts[slot] = annotated

    return {
        f"grade_{student_grade}": year1_dicts,
        f"grade_{student_grade + 1}": {
            slot: [r.to_dict() for r in recs]
            for slot, recs in year2.items()
        },
    }


def top_picks(
    slate: dict[str, list[CourseRecommendation]],
    n: int = 3,
) -> dict[str, list[dict]]:
    """Return the top *n* eligible courses per slot as plain dicts."""
    return {
        slot: [r.to_dict() for r in recs if r.eligible][:n]
        for slot, recs in slate.items()
    }

"""
evaluator.py
Prerequisite evaluation engine for RHS course recommendations.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Grade utilities
# ---------------------------------------------------------------------------

GRADE_POINTS: dict[str, float] = {
    "A+": 4.3, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}

PASSING_GPA = 1.7  # C- or better constitutes "completion"


def gpa(grade_str: str) -> float:
    """Convert letter grade string to GPA points."""
    return GRADE_POINTS.get(grade_str.strip().upper(), 0.0)


def _gpa_to_letter(gpa_val: float) -> str:
    for letter, pts in sorted(GRADE_POINTS.items(), key=lambda x: x[1], reverse=True):
        if gpa_val >= pts - 0.01:
            return letter
    return "F"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: str | Path) -> dict:
    """Load and return the rules JSON from *path*."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_course_ref(rules: dict, ref: str) -> list[dict]:
    """Return all courses whose code or name matches *ref* (case-insensitive)."""
    ref_lower = ref.lower()
    return [
        c for c in rules.get("courses", [])
        if c.get("course_code", "").lower() == ref_lower
        or ref_lower in c.get("name", "").lower()
    ]


# ---------------------------------------------------------------------------
# Transcript wrapper
# ---------------------------------------------------------------------------

class Transcript:
    """Wrapper around a list of completed-course dicts."""

    def __init__(self, entries: list[dict]):
        # entries: [{"course_code": str, "name": str, "grade": str, "grade_level": int}, ...]
        self._entries = entries
        self._by_code: dict[str, list[dict]] = {}
        for e in entries:
            self._by_code.setdefault(e["course_code"], []).append(e)

    def completed_codes(self) -> set[str]:
        return set(self._by_code.keys())

    def best_grade(self, code: str) -> float:
        """Highest GPA earned for a given course code. 0.0 if not taken."""
        entries = self._by_code.get(code, [])
        return max((gpa(e["grade"]) for e in entries), default=0.0)

    def best_grade_in_group(self, rules: dict, group_name: str) -> tuple[float, str | None]:
        """Best (gpa, course_code) among all courses in a named group. (0.0, None) if none taken."""
        group = rules.get("course_groups", {}).get(group_name, [])
        best, best_code = 0.0, None
        for code in group:
            g = self.best_grade(code)
            if g > best:
                best, best_code = g, code
        return best, best_code

    def took_any_in_group(self, rules: dict, group_name: str) -> bool:
        group = rules.get("course_groups", {}).get(group_name, [])
        return any(c in self.completed_codes() for c in group)

    def cumulative_weighted_gpa(self) -> float:
        if not self._entries:
            return 0.0
        return round(sum(gpa(e["grade"]) for e in self._entries) / len(self._entries), 3)

    def exam_taken(self, course_code: str) -> bool:
        for e in self._entries:
            if e.get("course_code") == course_code and e.get("exam_taken"):
                return True
        return False


# ---------------------------------------------------------------------------
# Prerequisite node evaluation (recursive)
# ---------------------------------------------------------------------------

def _codes_to_names(codes: list[str], rules: dict) -> list[str]:
    code_map = {c["course_code"]: c.get("name", c["course_code"]) for c in rules.get("courses", [])}
    return [code_map.get(code, code) for code in codes]


def _eval_node(
    node: dict,
    transcript: Transcript,
    rules: dict,
    student_grade: int,
) -> tuple[bool, list[str]]:
    """
    Recursively evaluate a single prerequisite node.
    Returns (passed: bool, blocking_reasons: list[str]).
    Concurrent/corequisite nodes always pass here — collected separately.
    """
    t = node.get("type", "none")

    if t == "none":
        return True, []

    if t == "grade_level":
        lo, hi = node.get("min", 9), node.get("max", 12)
        if lo <= student_grade <= hi:
            return True, []
        return False, [f"Course only available grades {lo}–{hi} (student entering grade {student_grade})"]

    if t == "completed":
        codes = node.get("course_codes", [])
        for code in codes:
            if transcript.best_grade(code) >= PASSING_GPA:
                return True, []
        names = _codes_to_names(codes, rules)
        return False, [f"Must complete one of: {', '.join(names)}"]

    if t == "min_grade":
        codes = node.get("course_codes", [])
        min_g = node.get("min_gpa", 0.0)
        label = node.get("label", _gpa_to_letter(min_g))
        for code in codes:
            if transcript.best_grade(code) >= min_g:
                return True, []
        names = _codes_to_names(codes, rules)
        return False, [f"Need {label} or better in: {', '.join(names)}"]

    if t == "completed_group":
        group = node.get("group", "")
        if transcript.took_any_in_group(rules, group):
            return True, []
        label = node.get("group_label", group.replace("_", " "))
        return False, [f"Must complete any {label} course"]

    if t == "min_grade_group":
        group = node.get("group", "")
        min_g = node.get("min_gpa", 0.0)
        label = node.get("label", _gpa_to_letter(min_g))
        group_label = node.get("group_label", group.replace("_", " "))
        best_g, _ = transcript.best_grade_in_group(rules, group)
        if best_g >= min_g:
            return True, []
        return False, [f"Need {label} or better in any {group_label} course"]

    if t in ("concurrent", "concurrent_group"):
        # Soft requirement — always passes here, collected as corequisite
        return True, []

    if t == "cumulative_gpa":
        min_g = node.get("min_gpa", 0.0)
        actual = transcript.cumulative_weighted_gpa()
        if actual >= min_g:
            return True, []
        return False, [f"Cumulative GPA {actual:.2f} below required {min_g:.1f}"]

    if t == "exam_taken":
        codes = node.get("course_codes", [])
        for code in codes:
            if transcript.exam_taken(code):
                return True, []
        names = _codes_to_names(codes, rules)
        return False, [f"Must have taken the AP exam for: {', '.join(names)}"]

    if t == "all_of":
        all_reasons: list[str] = []
        for child in node.get("conditions", []):
            ok, reasons = _eval_node(child, transcript, rules, student_grade)
            if not ok:
                all_reasons.extend(reasons)
        return len(all_reasons) == 0, all_reasons

    if t == "any_of":
        child_reasons: list[str] = []
        for child in node.get("conditions", []):
            ok, reasons = _eval_node(child, transcript, rules, student_grade)
            if ok:
                return True, []
            child_reasons.extend(reasons)
        return False, [f"Need one of: [{' | '.join(child_reasons)}]"]

    return False, [f"Unknown prerequisite type: '{t}'"]


def _collect_coreqs(node: dict, rules: dict) -> list[str]:
    """Walk prereq tree and return all concurrent-enrollment requirements as strings."""
    result: list[str] = []
    t = node.get("type", "none")
    if t == "concurrent":
        codes = node.get("course_codes", [])
        names = _codes_to_names(codes, rules)
        result.append(f"Concurrent enrollment required in: {', '.join(names)}")
    elif t == "concurrent_group":
        label = node.get("group_label", node.get("group", "").replace("_", " "))
        result.append(f"Concurrent enrollment required in: {label} or higher")
    elif t in ("all_of", "any_of"):
        for child in node.get("conditions", []):
            result.extend(_collect_coreqs(child, rules))
    return result


# ---------------------------------------------------------------------------
# Public evaluate() function
# ---------------------------------------------------------------------------

def evaluate(
    rules: dict,
    transcript: list[dict],
    course_code: str,
    student_grade: int,
) -> dict:
    """
    Evaluate whether a student is eligible for *course_code* in *student_grade*.

    Parameters
    ----------
    rules         : loaded rules JSON (from _load())
    transcript    : list of dicts [{course_code, name, grade, grade_level}]
                    Add {"exam_taken": True} to mark AP exam completion.
    course_code   : target course code (e.g. "230A")
    student_grade : grade the student will be *entering* (10, 11, or 12)

    Returns
    -------
    dict with keys:
      eligible         : bool — can the student take this course?
      course_code      : str
      course_name      : str
      prereqs_met      : bool
      grade_level_ok   : bool
      reasons          : list[str]  — blocking reasons if not eligible
      warnings         : list[str]  — non-blocking notes
      corequisites     : list[str]  — concurrent enrollment requirements
      constraints      : dict       — scheduling flags
    """
    tx = Transcript(transcript)
    course_map = {c["course_code"]: c for c in rules.get("courses", [])}
    course = course_map.get(course_code)

    if course is None:
        return {
            "eligible": False,
            "course_code": course_code,
            "course_name": "UNKNOWN",
            "prereqs_met": False,
            "grade_level_ok": False,
            "reasons": [f"Course code '{course_code}' not found in rules file"],
            "warnings": [],
            "corequisites": [],
            "constraints": {},
        }

    reasons: list[str] = []
    warnings: list[str] = []

    # Grade level
    grade_levels = course.get("grade_levels", [9, 10, 11, 12])
    grade_level_ok = student_grade in grade_levels
    if not grade_level_ok:
        reasons.append(f"Only available in grades {min(grade_levels)}–{max(grade_levels)}")

    # AP Grade 9 hard block
    if course.get("level") == "AP" and student_grade == 9:
        reasons.append("No AP courses permitted in Grade 9")

    # Already taken
    if course_code in tx.completed_codes():
        warnings.append("Already completed — credit cannot be earned twice")

    # Prerequisites
    prereq_node = course.get("prerequisites", {"type": "none"})
    prereqs_met, prereq_reasons = _eval_node(prereq_node, tx, rules, student_grade)
    reasons.extend(prereq_reasons)

    # Corequisites
    corequisites = _collect_coreqs(prereq_node, rules)
    for coq in course.get("corequisites", []):
        names = _codes_to_names(coq.get("course_codes", []), rules)
        corequisites.append(f"Must also enroll in: {', '.join(names)}")

    # Constraints
    raw = course.get("constraints", {})
    constraints = {
        "requires_study_hall": raw.get("requires_study_hall", False),
        "summer_work": raw.get("summer_work", False),
        "biannual": raw.get("biannual", False),
        "ap_course": course.get("level") == "AP",
        "dual_enrollment": raw.get("dual_enrollment"),
        "fulfills": raw.get("fulfills", []),
    }

    biannual_parity = raw.get("biannual_parity")
    if biannual_parity:
        constraints["biannual_parity"] = biannual_parity
        warnings.append(
            f"Biannual course (runs {biannual_parity} school-year starts) — "
            "confirm availability with counselor"
        )

    eligible = (
        grade_level_ok
        and prereqs_met
        and not any("Grade 9" in r for r in reasons)
    )

    return {
        "eligible": eligible,
        "course_code": course_code,
        "course_name": course.get("name", ""),
        "prereqs_met": prereqs_met,
        "grade_level_ok": grade_level_ok,
        "reasons": reasons,
        "warnings": warnings,
        "corequisites": corequisites,
        "constraints": constraints,
    }

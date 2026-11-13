"""
Unit tests for evaluator.py.

Each test targets a specific node type, edge case, or documented policy rule
so that regressions in the prerequisite logic surface immediately.
"""
import pytest

from evaluator import Transcript, evaluate, gpa


# ---------------------------------------------------------------------------
# GPA utilities
# ---------------------------------------------------------------------------

def test_gpa_known_grades():
    assert gpa("A+") == 4.3
    assert gpa("A") == 4.0
    assert gpa("B+") == 3.3
    assert gpa("C-") == 1.7
    assert gpa("F") == 0.0


def test_gpa_case_insensitive():
    assert gpa("a-") == gpa("A-")
    assert gpa("b+") == gpa("B+")


def test_gpa_unknown_returns_zero():
    assert gpa("Z") == 0.0
    assert gpa("") == 0.0
    assert gpa("Pass") == 0.0


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def test_transcript_best_grade_single(rules):
    tx = Transcript([{"course_code": "120C", "name": "CP Bio", "grade": "B+", "grade_level": 9}])
    assert tx.best_grade("120C") == gpa("B+")
    assert tx.best_grade("999X") == 0.0


def test_transcript_best_grade_repeated_takes_highest(rules):
    tx = Transcript([
        {"course_code": "120C", "name": "CP Bio", "grade": "C", "grade_level": 9},
        {"course_code": "120C", "name": "CP Bio", "grade": "B+", "grade_level": 10},
    ])
    assert tx.best_grade("120C") == gpa("B+")


def test_transcript_best_grade_in_group(rules):
    tx = Transcript([
        {"course_code": "120C", "name": "CP Bio", "grade": "B", "grade_level": 9},
        {"course_code": "120H", "name": "Honors Bio", "grade": "A", "grade_level": 9},
    ])
    best_gpa, best_code = tx.best_grade_in_group(rules, "biology")
    assert best_gpa == gpa("A")
    assert best_code == "120H"


def test_transcript_took_any_in_group_true(rules):
    tx = Transcript([{"course_code": "120M", "name": "Enriched Bio", "grade": "B", "grade_level": 9}])
    assert tx.took_any_in_group(rules, "biology") is True


def test_transcript_took_any_in_group_false(rules):
    tx = Transcript([])
    assert tx.took_any_in_group(rules, "biology") is False


def test_transcript_exam_taken_flag(rules):
    tx = Transcript([
        {"course_code": "250A", "name": "AP Seminar", "grade": "B", "grade_level": 10, "exam_taken": True},
    ])
    assert tx.exam_taken("250A") is True
    assert tx.exam_taken("260A") is False


# ---------------------------------------------------------------------------
# Course-not-found
# ---------------------------------------------------------------------------

def test_unknown_course_returns_ineligible(rules):
    result = evaluate(rules, [], "FAKE99", 10)
    assert result["eligible"] is False
    assert result["course_name"] == "UNKNOWN"
    assert any("not found" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# No prerequisites
# ---------------------------------------------------------------------------

def test_no_prereqs_eligible(rules):
    # CP Biology has no prerequisites and is available grades 9-12
    result = evaluate(rules, [], "120C", 9)
    assert result["eligible"] is True
    assert result["reasons"] == []
    assert result["prereqs_met"] is True
    assert result["grade_level_ok"] is True


# ---------------------------------------------------------------------------
# Grade-level restriction
# ---------------------------------------------------------------------------

def test_grade_level_blocks_too_early(rules):
    # AP Biology is grades 11-12 only
    result = evaluate(rules, [], "120A", 10)
    assert result["grade_level_ok"] is False
    assert result["eligible"] is False


def test_grade_level_passes_when_in_range(rules):
    # AP Biology in grade 11 — prereqs will still block, but grade_level_ok should pass
    result = evaluate(rules, [], "120A", 11)
    assert result["grade_level_ok"] is True


# ---------------------------------------------------------------------------
# AP Grade 9 hard block (policy rule, not a prereq node)
# ---------------------------------------------------------------------------

def test_ap_hard_blocked_grade_9(rules):
    # No AP courses permitted for incoming 9th graders regardless of transcript
    result = evaluate(rules, [], "140A", 9)
    assert result["eligible"] is False
    assert any("Grade 9" in r for r in result["reasons"])


def test_ap_not_blocked_grade_10(rules):
    # Grade-level gate passes for AP Physics 1 in grade 10; prereqs may still fail
    result = evaluate(rules, [], "140A", 10)
    assert result["grade_level_ok"] is True
    assert not any("Grade 9" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# completed_group node
# ---------------------------------------------------------------------------

def test_completed_group_met(rules):
    # CP Chemistry requires any Biology course
    tx = [{"course_code": "120C", "name": "CP Bio", "grade": "B", "grade_level": 9}]
    result = evaluate(rules, tx, "130C", 10)
    assert result["prereqs_met"] is True
    assert result["eligible"] is True


def test_completed_group_not_met(rules):
    result = evaluate(rules, [], "130C", 10)
    assert result["prereqs_met"] is False
    assert any("Biology" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# min_grade node (any_of with two branches)
# ---------------------------------------------------------------------------

def test_min_grade_met_via_first_branch(rules):
    # AP Calc BC (311A): B- in AP Calc AB OR A in Honors Precalc
    tx = [{"course_code": "310A", "name": "AP Calc AB", "grade": "B", "grade_level": 10}]
    result = evaluate(rules, tx, "311A", 11)
    assert result["prereqs_met"] is True


def test_min_grade_met_via_second_branch(rules):
    tx = [{"course_code": "308H", "name": "Honors Precalc", "grade": "A", "grade_level": 10}]
    result = evaluate(rules, tx, "311A", 11)
    assert result["prereqs_met"] is True


def test_min_grade_not_met_below_threshold(rules):
    # C in AP Calc AB — below the B- (2.7) threshold
    tx = [{"course_code": "310A", "name": "AP Calc AB", "grade": "C", "grade_level": 10}]
    result = evaluate(rules, tx, "311A", 11)
    assert result["prereqs_met"] is False


# ---------------------------------------------------------------------------
# concurrent_group node — treated as soft requirement
# ---------------------------------------------------------------------------

def test_concurrent_passes_and_collected_as_coreq(rules):
    # Honors Chemistry (130H): requires A- in Enriched Bio OR B- in Honors Bio,
    # AND concurrent enrollment in Geometry or higher (soft).
    tx = [{"course_code": "120M", "name": "Enriched Bio", "grade": "A-", "grade_level": 9}]
    result = evaluate(rules, tx, "130H", 10)
    assert result["prereqs_met"] is True
    assert any("Geometry" in c for c in result["corequisites"])


# ---------------------------------------------------------------------------
# APUSH (540A) — complex any_of with grade-conditional branches
# ---------------------------------------------------------------------------

def test_apush_eligible_via_honors_global_path(rules):
    # Branch 1: A- in Honors Global History 2 (510H) → eligible at grade 10
    tx = [{"course_code": "510H", "name": "Honors Global 2", "grade": "A-", "grade_level": 9}]
    result = evaluate(rules, tx, "540A", 10)
    assert result["prereqs_met"] is True
    assert result["eligible"] is True


def test_apush_eligible_via_cp_global_a_plus(rules):
    # Branch 2: A+ in CP Global History 2 (5100)
    tx = [{"course_code": "5100", "name": "CP Global 2", "grade": "A+", "grade_level": 9}]
    result = evaluate(rules, tx, "540A", 10)
    assert result["prereqs_met"] is True


def test_apush_grade_level_branch_unlocks_at_grade_11(rules):
    # Branch 3: grade 11+ AND A- in any required CP SS → eligible
    tx = [{"course_code": "5100", "name": "CP Global 2", "grade": "A-", "grade_level": 9}]
    result = evaluate(rules, tx, "540A", 11)
    assert result["prereqs_met"] is True
    assert result["eligible"] is True


def test_apush_grade_level_branch_blocked_at_grade_10(rules):
    # Grade 10 with only B in CP Global 2 — none of the four branches pass
    tx = [{"course_code": "5100", "name": "CP Global 2", "grade": "B", "grade_level": 9}]
    result = evaluate(rules, tx, "540A", 10)
    assert result["eligible"] is False


# ---------------------------------------------------------------------------
# AP Research (260A) — exam_taken node inside all_of
# ---------------------------------------------------------------------------

def test_ap_research_blocked_without_exam(rules):
    # AP Seminar completed but no exam flag → exam_taken node fails
    tx = [
        {"course_code": "250A", "name": "AP Seminar", "grade": "B", "grade_level": 10},
        {"course_code": "230A", "name": "AP Language", "grade": "B-", "grade_level": 10},
    ]
    result = evaluate(rules, tx, "260A", 11)
    assert result["prereqs_met"] is False
    assert any("exam" in r.lower() for r in result["reasons"])


def test_ap_research_eligible_with_exam_and_english(rules):
    # exam_taken=True for AP Seminar + B- in AP Lang (honors_or_ap_english_11 group) → eligible
    tx = [
        {"course_code": "250A", "name": "AP Seminar", "grade": "B", "grade_level": 10, "exam_taken": True},
        {"course_code": "230A", "name": "AP Language", "grade": "B-", "grade_level": 10},
    ]
    result = evaluate(rules, tx, "260A", 11)
    assert result["prereqs_met"] is True
    assert result["eligible"] is True


# ---------------------------------------------------------------------------
# Already-completed warning
# ---------------------------------------------------------------------------

def test_already_completed_adds_warning(rules):
    tx = [{"course_code": "120C", "name": "CP Bio", "grade": "B", "grade_level": 9}]
    result = evaluate(rules, tx, "120C", 10)
    assert any("Already completed" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Constraints flags
# ---------------------------------------------------------------------------

def test_study_hall_trigger_flag(rules):
    # AP Bio (120A) is a study hall trigger
    result = evaluate(rules, [], "120A", 11)
    assert result["constraints"]["requires_study_hall"] is True
    assert result["constraints"]["ap_course"] is True


def test_non_trigger_course_no_study_hall(rules):
    # CP Biology never triggers study hall
    result = evaluate(rules, [], "120C", 9)
    assert result["constraints"]["requires_study_hall"] is False


def test_biannual_parity_warning(rules):
    # Honors Latin Literature (430H) runs even years only
    result = evaluate(rules, [], "430H", 11)
    assert result["constraints"]["biannual"] is True
    assert result["constraints"]["biannual_parity"] == "even"
    assert any("Biannual" in w for w in result["warnings"])

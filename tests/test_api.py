"""
Integration tests for recommender_api.py.

Each test sends a real HTTP request through the ASGI transport so the full
request/response cycle — routing, Pydantic validation, evaluator, recommender — is exercised.
"""
import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Shared fixtures (defined in conftest.py)
# ---------------------------------------------------------------------------

# A complete grade-10 incoming transcript used across several tests
GRADE_10_TRANSCRIPT = [
    {"course_code": "120H", "name": "Honors Biology", "grade": "A", "grade_level": 9},
    {"course_code": "2100", "name": "CP English 9", "grade": "B+", "grade_level": 9},
    {"course_code": "3020", "name": "Algebra I", "grade": "A", "grade_level": 9},
    {"course_code": "510H", "name": "Honors Global History 2", "grade": "A-", "grade_level": 9},
]

SLOTS = {"english", "math", "science", "social_studies", "world_language", "elective_1", "elective_2", "pe_health"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

async def test_health_ok(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["courses_loaded"] == 119


# ---------------------------------------------------------------------------
# /courses
# ---------------------------------------------------------------------------

async def test_courses_returns_all(client: AsyncClient):
    r = await client.get("/courses")
    assert r.status_code == 200
    assert len(r.json()) == 119


async def test_courses_filter_by_level(client: AsyncClient):
    r = await client.get("/courses", params={"level": "AP"})
    assert r.status_code == 200
    courses = r.json()
    assert len(courses) > 0
    assert all(c["level"] == "AP" for c in courses)


async def test_courses_filter_by_department(client: AsyncClient):
    r = await client.get("/courses", params={"department": "Science"})
    assert r.status_code == 200
    courses = r.json()
    assert len(courses) > 0
    assert all(c["department"] == "Science" for c in courses)


async def test_courses_combined_filter(client: AsyncClient):
    r = await client.get("/courses", params={"department": "Mathematics", "level": "AP"})
    assert r.status_code == 200
    courses = r.json()
    assert all(c["department"] == "Mathematics" and c["level"] == "AP" for c in courses)


# ---------------------------------------------------------------------------
# /evaluate/{course_code} — happy paths
# ---------------------------------------------------------------------------

async def test_evaluate_eligible_no_prereqs(client: AsyncClient):
    # CP Biology has no prerequisites
    r = await client.post("/evaluate/120C", json={"transcript": [], "student_grade": 9})
    assert r.status_code == 200
    data = r.json()
    assert data["eligible"] is True
    assert data["prereqs_met"] is True
    assert data["grade_level_ok"] is True
    assert data["reasons"] == []


async def test_evaluate_not_eligible_missing_prereq(client: AsyncClient):
    # CP Chemistry requires any Biology — empty transcript → blocked
    r = await client.post("/evaluate/130C", json={"transcript": [], "student_grade": 10})
    assert r.status_code == 200
    data = r.json()
    assert data["eligible"] is False
    assert data["prereqs_met"] is False
    assert len(data["reasons"]) > 0


async def test_evaluate_response_shape(client: AsyncClient):
    r = await client.post("/evaluate/120C", json={"transcript": [], "student_grade": 9})
    data = r.json()
    for key in ("eligible", "course_code", "course_name", "prereqs_met", "grade_level_ok",
                "reasons", "warnings", "corequisites", "constraints"):
        assert key in data, f"missing key: {key}"


async def test_evaluate_constraints_study_hall_flag(client: AsyncClient):
    r = await client.post("/evaluate/120A", json={"transcript": [], "student_grade": 11})
    data = r.json()
    assert data["constraints"]["requires_study_hall"] is True
    assert data["constraints"]["ap_course"] is True


async def test_evaluate_ap_grade_9_block(client: AsyncClient):
    r = await client.post("/evaluate/140A", json={"transcript": [], "student_grade": 9})
    assert r.status_code == 200
    data = r.json()
    assert data["eligible"] is False
    assert any("Grade 9" in reason for reason in data["reasons"])


# ---------------------------------------------------------------------------
# /evaluate/{course_code} — error handling
# ---------------------------------------------------------------------------

async def test_evaluate_unknown_course_404(client: AsyncClient):
    r = await client.post("/evaluate/FAKE99", json={"transcript": [], "student_grade": 10})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


async def test_evaluate_invalid_grade_422(client: AsyncClient):
    payload = {
        "transcript": [{"course_code": "120C", "name": "Bio", "grade": "X", "grade_level": 9}],
        "student_grade": 10,
    }
    r = await client.post("/evaluate/130C", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(e["loc"][-1] == "grade" for e in detail)


async def test_evaluate_grade_out_of_range_422(client: AsyncClient):
    r = await client.post("/evaluate/120C", json={"transcript": [], "student_grade": 13})
    assert r.status_code == 422


async def test_evaluate_missing_student_grade_422(client: AsyncClient):
    r = await client.post("/evaluate/120C", json={"transcript": []})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /recommend — structure
# ---------------------------------------------------------------------------

async def test_recommend_returns_two_year_keys(client: AsyncClient):
    payload = {
        "transcript": GRADE_10_TRANSCRIPT,
        "student_grade": 10,
        "career_tracks": ["computer_science"],
        "target_tier": "top20_30",
    }
    r = await client.post("/recommend", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "grade_10" in data
    assert "grade_11" in data


async def test_recommend_grade_12_returns_single_year(client: AsyncClient):
    payload = {"transcript": [], "student_grade": 12}
    r = await client.post("/recommend", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"grade_12"}


async def test_recommend_all_slots_present(client: AsyncClient):
    payload = {"transcript": GRADE_10_TRANSCRIPT, "student_grade": 10}
    r = await client.post("/recommend", json=payload)
    data = r.json()
    for year in data.values():
        assert set(year.keys()) == SLOTS


async def test_recommend_only_eligible_courses(client: AsyncClient):
    payload = {"transcript": GRADE_10_TRANSCRIPT, "student_grade": 10}
    r = await client.post("/recommend", json=payload)
    data = r.json()
    for year in data.values():
        for slot_courses in year.values():
            for course in slot_courses:
                assert course["eligible"] is True, f"{course['course_code']} should be eligible in /recommend"


async def test_recommend_at_most_three_per_slot(client: AsyncClient):
    payload = {"transcript": GRADE_10_TRANSCRIPT, "student_grade": 10}
    r = await client.post("/recommend", json=payload)
    data = r.json()
    for year in data.values():
        for slot, courses in year.items():
            assert len(courses) <= 3, f"{slot} has {len(courses)} courses (max 3)"


async def test_recommend_default_target_tier(client: AsyncClient):
    # target_tier is optional — should default without error
    payload = {"transcript": [], "student_grade": 9}
    r = await client.post("/recommend", json=payload)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /recommend — validation errors
# ---------------------------------------------------------------------------

async def test_recommend_invalid_career_track_422(client: AsyncClient):
    payload = {"transcript": [], "student_grade": 10, "career_tracks": ["underwater_basket_weaving"]}
    r = await client.post("/recommend", json=payload)
    assert r.status_code == 422


async def test_recommend_invalid_target_tier_422(client: AsyncClient):
    payload = {"transcript": [], "student_grade": 10, "target_tier": "ultra_elite"}
    r = await client.post("/recommend", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /recommend/full — includes blocked courses
# ---------------------------------------------------------------------------

async def test_recommend_full_includes_blocked_courses(client: AsyncClient):
    # With an empty transcript, many prerequisites will fail — full endpoint should include them
    payload = {"transcript": [], "student_grade": 11}
    r = await client.post("/recommend/full", json=payload)
    assert r.status_code == 200
    data = r.json()
    blocked_found = False
    for year in data.values():
        for slot_courses in year.values():
            for course in slot_courses:
                if not course["eligible"]:
                    blocked_found = True
    assert blocked_found, "/recommend/full should include ineligible courses"


async def test_recommend_full_blocked_courses_have_reasons(client: AsyncClient):
    payload = {"transcript": [], "student_grade": 11}
    r = await client.post("/recommend/full", json=payload)
    data = r.json()
    for year in data.values():
        for slot_courses in year.values():
            for course in slot_courses:
                if not course["eligible"]:
                    assert len(course["blocking_reasons"]) > 0, (
                        f"{course['course_code']} is blocked but has no blocking_reasons"
                    )

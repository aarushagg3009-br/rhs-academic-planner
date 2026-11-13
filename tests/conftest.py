import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Set absolute paths before app import so load_dotenv() doesn't shadow them
os.environ.setdefault("RHS_RULES_PATH", str(ROOT / "RHS_POS_2026-2027_rules.json"))
os.environ.setdefault("RHS_RECOMMENDER_CONFIG", str(ROOT / "recommender_config.json"))

from recommender_api import app  # noqa: E402


@pytest.fixture(scope="session")
def rules() -> dict:
    return json.loads((ROOT / "RHS_POS_2026-2027_rules.json").read_text(encoding="utf-8"))


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

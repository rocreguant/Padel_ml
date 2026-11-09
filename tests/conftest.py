from pathlib import Path

import pytest

from padel_ml.db import Database

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()

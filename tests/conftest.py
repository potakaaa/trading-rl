import pytest

from src.config import load_settings


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.data.root = str(tmp_path / "data")
    return s

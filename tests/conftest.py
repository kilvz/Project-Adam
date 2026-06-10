import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from project_adam.config import set_memory_dir


@pytest.fixture(autouse=True)
def temp_memory_dir(tmp_path, monkeypatch):
    set_memory_dir(tmp_path)
    return tmp_path

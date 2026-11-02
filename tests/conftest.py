import json
from pathlib import Path

import pytest

from nhefs import data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw():
    return data.load_raw()


@pytest.fixture(scope="session")
def clean(raw):
    return data.clean(raw)


@pytest.fixture(scope="session")
def causal_json():
    return json.loads((ROOT / "models" / "causal.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pred_json():
    return json.loads((ROOT / "models" / "predictivo.json").read_text(encoding="utf-8"))

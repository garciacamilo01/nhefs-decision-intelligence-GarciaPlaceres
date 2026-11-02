"""Smoke test de la web app: los cuatro módulos renderizan sin excepciones."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.mark.parametrize("i", range(4))
def test_modulo_renderiza(i):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    radio = at.sidebar.radio[0]
    radio.set_value(radio.options[i]).run()
    assert not at.exception


def test_prediccion_es_probabilidad():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[1]).run()
    valor = float(at.metric[0].value.rstrip("%")) / 100
    assert 0 < valor < 1

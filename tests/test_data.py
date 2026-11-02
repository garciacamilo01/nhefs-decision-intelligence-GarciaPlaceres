"""Integridad de los datos y de la limpieza (rol Data Engineer)."""

from nhefs import data
from nhefs.config import CONFOUNDERS, TREATMENT


def test_dimensiones_y_clave(raw):
    assert raw.shape == (1629, 64)
    assert raw["seqn"].is_unique


def test_confusores_y_tratamiento_completos(raw):
    assert raw[CONFOUNDERS + [TREATMENT]].notna().all().all()


def test_bloque_faltante_por_diseno(clean):
    bloque = clean["bloque_no_informado"] == 1
    assert bloque.sum() == 791
    # el faltante oculto coincide exactamente con el cambio de cuestionario
    assert (clean.loc[bloque, "seqn"] >= 20751).all()
    assert (clean.loc[~bloque, "seqn"] < 20751).all()


def test_codigos_ocultos_recodificados(clean):
    assert not (clean["alcoholfreq"] == 5).any()
    assert set(clean["hbp_cat"].dropna()) <= {"no", "si", "no_informado"}


def test_censura(clean):
    assert clean["censurado"].sum() == 63
    dc = data.causal_sample(clean)
    assert len(dc) == 1566 and dc["wt82_71"].notna().all()


def test_auditoria_detecta_codigos_ocultos(raw):
    a = data.audit(raw)
    assert {"diabetes", "alcoholfreq", "wt82_71"} <= set(a["variable"])


def test_limpieza_no_modifica_el_original(raw):
    antes = raw.copy()
    data.clean(raw)
    assert raw.equals(antes)

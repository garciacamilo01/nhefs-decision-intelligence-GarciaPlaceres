"""Pruebas del DAG, del criterio backdoor y pruebas deterministas de los estimadores."""

import networkx as nx
import numpy as np
import pytest

from nhefs import causal, data
from nhefs.config import COLLIDER, CONFOUNDERS, INSTRUMENT


@pytest.fixture(scope="module")
def g():
    return causal.graph()


@pytest.fixture(scope="module")
def dc(clean):
    return data.causal_sample(clean)


def test_dag_aciclico(g):
    assert nx.is_directed_acyclic_graph(g)
    assert nx.is_directed_acyclic_graph(causal.graph(include_latent=True))


def test_backdoor_conjunto_de_confusores(g):
    assert causal.satisfies_backdoor(g, set(CONFOUNDERS))
    assert not causal.satisfies_backdoor(g, set())


def test_colisionador_y_descendientes_invalidan_el_ajuste(g):
    assert not causal.satisfies_backdoor(g, set(CONFOUNDERS) | {COLLIDER})


def test_instrumento_solo_afecta_al_tratamiento(g):
    assert set(g.successors(INSTRUMENT)) == {"qsmk"}


def test_conjunto_minimo_unico(g):
    sets = causal.minimal_adjustment_sets(g, CONFOUNDERS + [INSTRUMENT])
    assert sets == [set(CONFOUNDERS)]


@pytest.mark.parametrize("p,ok,esperado", [(0.9, True, "verde"), (0.07, True, "amarillo"),
                                           (0.01, True, "rojo"), (0.9, False, "rojo")])
def test_semaforo(p, ok, esperado):
    assert causal.refutation_light(p, ok) == esperado


def test_ate_ingenuo_y_ajustado_valores_de_referencia(dc):
    assert causal.naive(dc).params["qsmk"] == pytest.approx(2.540, abs=0.01)
    assert causal.ols_adjusted(dc).params["qsmk"] == pytest.approx(3.35, abs=0.02)


def test_ipw_determinista(dc):
    assert causal.ipw_ate(dc) == causal.ipw_ate(dc)
    a = causal.bootstrap(dc, causal.ipw_ate, n=20)
    b = causal.bootstrap(dc, causal.ipw_ate, n=20)
    np.testing.assert_array_equal(a, b)


def test_ipw_recupera_efecto_simulado(dc):
    """Prueba de control: con Y sintético y efecto conocido = 5, el IPW debe recuperarlo."""
    rng = np.random.default_rng(0)
    sim = dc.copy()
    sim["wt82_71"] = 5 * sim["qsmk"] - 0.2 * sim["age"] + 0.05 * sim["smokeintensity"] + rng.normal(0, 3, len(sim))
    assert causal.ipw_ate(sim) == pytest.approx(5, abs=0.6)


@pytest.mark.slow
def test_dml_reproducible(dc):
    a = causal.fit_dml(dc).ate(X=dc[["smokeintensity", "age"]])
    b = causal.fit_dml(dc).ate(X=dc[["smokeintensity", "age"]])
    assert a == pytest.approx(b, abs=1e-8)

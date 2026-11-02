"""DAG causal, criterio backdoor y estimación del efecto de dejar de fumar sobre el peso."""

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

from .config import COLLIDER, CONFOUNDERS, EFFECT_MODIFIERS, INSTRUMENT, OUTCOME_CAUSAL, SEED, TREATMENT

T, Y = TREATMENT, OUTCOME_CAUSAL

# Nodos latentes (no medidos): se dibujan en el DAG pero no entran en DoWhy
LATENT_MEDIATOR = "ingesta_metabolismo"
LATENT_CONFOUNDER = "conciencia_salud"
SELECTION = "peso82_observado"

ROLES = {**{z: "Confusor" for z in CONFOUNDERS}, T: "Tratamiento", Y: "Resultado",
         INSTRUMENT: "Instrumento", COLLIDER: "Colisionador", LATENT_MEDIATOR: "Mediador (latente)",
         LATENT_CONFOUNDER: "Confusor no observado", SELECTION: "Selección (censura)"}

LABELS = {"sex": "Sexo", "race": "Raza", "age": "Edad", "education": "Educación",
          "smokeintensity": "Cigarrillos/día", "smokeyrs": "Años fumando", "exercise": "Ejercicio",
          "active": "Actividad diaria", "wt71": "Peso 1971", T: "Dejó de fumar (T)",
          Y: "Cambio de peso (Y)", INSTRUMENT: "Precio tabaco 82", COLLIDER: "Presión sist. 82 (C)",
          LATENT_MEDIATOR: "Ingesta / metabolismo (M)", LATENT_CONFOUNDER: "Conciencia de salud (U)",
          SELECTION: "Peso 82 observado (S)"}


def edges(include_latent: bool = False) -> list[tuple[str, str]]:
    """Arcos del DAG. Cada arco está justificado en BITACORA.md (Entrada 4)."""
    e = [(z, T) for z in CONFOUNDERS] + [(z, Y) for z in CONFOUNDERS]
    e += [("age", "smokeyrs"), ("sex", "wt71"), ("age", "wt71")]
    e += [(INSTRUMENT, T), (T, Y), (T, COLLIDER), (Y, COLLIDER), ("age", COLLIDER)]
    if include_latent:
        e += [(T, LATENT_MEDIATOR), (LATENT_MEDIATOR, Y)]
        e += [(LATENT_CONFOUNDER, T), (LATENT_CONFOUNDER, "exercise")]
        e += [(T, SELECTION), ("age", SELECTION), ("wt71", SELECTION)]
    return e


def graph(include_latent: bool = False) -> nx.DiGraph:
    g = nx.DiGraph(edges(include_latent))
    assert nx.is_directed_acyclic_graph(g)
    return g


def dot(include_latent: bool = False) -> str:
    return "digraph {" + " ".join(f"{a} -> {b};" for a, b in edges(include_latent)) + "}"


def paths_table(g: nx.DiGraph) -> pd.DataFrame:
    """Enumera y clasifica todos los caminos simples entre T e Y."""
    u = g.to_undirected()
    filas = []
    for camino in nx.all_simple_paths(u, T, Y):
        flechas = [camino[0]]
        for a, b in zip(camino[:-1], camino[1:]):
            flechas.append(("→ " if g.has_edge(a, b) else "← ") + b)
        col = [camino[i] for i in range(1, len(camino) - 1)
               if g.has_edge(camino[i - 1], camino[i]) and g.has_edge(camino[i + 1], camino[i])]
        causal = all(g.has_edge(a, b) for a, b in zip(camino[:-1], camino[1:]))
        backdoor = g.has_edge(camino[1], camino[0])
        if causal:
            tipo = "Causal"
        elif col:
            tipo = "Bloqueado por colisionador"
        elif backdoor:
            tipo = "Puerta trasera (abierto)"
        else:
            tipo = "No causal (abierto)"
        filas.append({"camino": " ".join(flechas), "tipo": tipo, "colisionadores": ", ".join(col), "largo": len(camino) - 1})
    return pd.DataFrame(filas).sort_values(["tipo", "largo"]).reset_index(drop=True)


def satisfies_backdoor(g: nx.DiGraph, s: set[str]) -> bool:
    """Criterio backdoor: S sin descendientes de T y T ⫫ Y | S en el grafo sin las flechas que salen de T."""
    if s & nx.descendants(g, T):
        return False
    gb = g.copy()
    gb.remove_edges_from(list(g.out_edges(T)))
    dsep = nx.is_d_separator if hasattr(nx, "is_d_separator") else nx.d_separated
    return dsep(gb, {T}, {Y}, s)


def minimal_adjustment_sets(g: nx.DiGraph, candidates: list[str]) -> list[set[str]]:
    """Todos los conjuntos válidos de tamaño mínimo (búsqueda exhaustiva por tamaño creciente)."""
    for k in range(len(candidates) + 1):
        ok = [set(s) for s in itertools.combinations(candidates, k) if satisfies_backdoor(g, set(s))]
        if ok:
            return ok
    return []


# --------------------------------------------------------------------- Estimadores
FORMULA_Z = " + ".join(CONFOUNDERS)


def naive(d: pd.DataFrame):
    return smf.ols(f"{Y} ~ {T}", data=d).fit()


def ols_adjusted(d: pd.DataFrame, extra: str = ""):
    return smf.ols(f"{Y} ~ {T} + {FORMULA_Z}{extra}", data=d).fit()


def ipw_ate(d: pd.DataFrame) -> float:
    """ATE por ponderación de propensión inversa estabilizada (Hájek)."""
    ps = LogisticRegression(max_iter=2000).fit(d[CONFOUNDERS], d[T]).predict_proba(d[CONFOUNDERS])[:, 1]
    w1, w0 = d[T] / ps, (1 - d[T]) / (1 - ps)
    return float(np.sum(w1 * d[Y]) / np.sum(w1) - np.sum(w0 * d[Y]) / np.sum(w0))


def bootstrap(d: pd.DataFrame, fn, n: int = 300, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([fn(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n)])


def ipcw_ate(d_full: pd.DataFrame) -> float:
    """Sensibilidad a la censura: OLS ajustado ponderado por 1 / P(no censurado | T, Z) (supuesto MAR)."""
    X = d_full[[T] + CONFOUNDERS]
    p_obs = LogisticRegression(max_iter=2000).fit(X, 1 - d_full["censurado"]).predict_proba(X)[:, 1]
    obs = d_full["censurado"] == 0
    dd = d_full[obs].assign(w=1 / p_obs[obs.values])
    return float(smf.wls(f"{Y} ~ {T} + {FORMULA_Z}", data=dd, weights=dd["w"]).fit().params[T])


def fit_dml(d: pd.DataFrame, cv: int = 5):
    """Double ML lineal: τ(X) = θ0 + θ'X con X = modificadores del efecto y W = resto de confusores."""
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    W = [z for z in CONFOUNDERS if z not in EFFECT_MODIFIERS]
    dml = LinearDML(model_y=RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=SEED),
                    model_t=RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=SEED),
                    discrete_treatment=True, cv=cv, random_state=SEED)
    dml.fit(Y=d[Y], T=d[T], X=d[EFFECT_MODIFIERS], W=d[W])
    return dml


def cate_grid(dml, d: pd.DataFrame, n_int: int = 41, n_age: int = 26) -> pd.DataFrame:
    """CATE e IC 95% sobre una grilla (cigarrillos/día × edad) dentro del soporte observado."""
    si = np.linspace(d["smokeintensity"].min(), d["smokeintensity"].quantile(0.99), n_int).round(1)
    ag = np.linspace(d["age"].min(), d["age"].max(), n_age).round(1)
    g = pd.DataFrame([(a, b) for a in si for b in ag], columns=EFFECT_MODIFIERS)
    lo, hi = dml.effect_interval(g.values, alpha=0.05)
    return g.assign(cate=dml.effect(g.values), ic_inf=lo, ic_sup=hi)


def refutation_light(p_value: float, ok_cond: bool = True) -> str:
    """Semáforo: verde si p > 0,10; amarillo si 0,05 < p ≤ 0,10; rojo si p ≤ 0,05 o falla la condición."""
    if not ok_cond or p_value <= 0.05:
        return "rojo"
    return "verde" if p_value > 0.10 else "amarillo"

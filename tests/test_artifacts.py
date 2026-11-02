"""Contrato entre el pipeline y la web app: los artefactos existen y son coherentes."""

import joblib
import pandas as pd

from nhefs import features
from nhefs.config import MODELS_DIR


def test_artefactos_presentes():
    for f in ["modelo_predictivo.joblib", "pca.joblib", "predictivo.json", "causal.json", "datos_app.csv"]:
        assert (MODELS_DIR / f).exists(), f


def test_metricas_predictivas_minimas(pred_json):
    m = pred_json["metricas_test"]
    assert m["auc_roc"] > 0.80
    assert m["brier"] < 0.158  # mejor que predecir siempre la prevalencia


def test_modelo_reproduce_metricas(clean, pred_json):
    from nhefs.pipeline import predictive_metrics
    _, Xte, _, yte = features.split(clean)
    m = predictive_metrics(joblib.load(MODELS_DIR / "modelo_predictivo.joblib"), Xte, yte)
    assert m["auc_roc"] == pred_json["metricas_test"]["auc_roc"]


def test_resultados_causales_coherentes(causal_json):
    c = causal_json
    for k in ["ols", "ipw", "dml"]:
        lo, hi = c[k]["ic"]
        assert lo < c[k]["ate"] < hi
        assert lo > 0  # efecto positivo y significativo
    assert c["naive"]["ate"] < c["ols"]["ate"]
    assert len(c["refutaciones"]) == 3
    assert {r["semaforo"] for r in c["refutaciones"].values()} <= {"verde", "amarillo", "rojo"}


def test_grilla_cate(causal_json):
    g = pd.DataFrame(causal_json["cate_grid"])
    assert {"smokeintensity", "age", "cate", "ic_inf", "ic_sup"} <= set(g.columns)
    assert (g["ic_inf"] <= g["cate"]).all() and (g["cate"] <= g["ic_sup"]).all()

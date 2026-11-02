"""Pruebas anti-fuga de datos (rol QA/MLOps)."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nhefs import features
from nhefs.config import MODELS_DIR, PCA_VARS

POST_TRATAMIENTO = {"death", "yrdth", "modth", "dadth", "wt82", "wt82_71", "sbp", "dbp", "price82", "tax82",
                    "censurado", "seqn"}


def test_split_disjunto_y_estratificado(clean):
    Xtr, Xte, ytr, yte = features.split(clean)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert len(Xtr) + len(Xte) == len(clean)
    assert abs(ytr.mean() - yte.mean()) < 0.01


def test_split_reproducible(clean):
    a = features.split(clean)[0].index
    b = features.split(clean)[0].index
    assert a.equals(b)


def test_sin_variables_posteriores_al_resultado(clean, pred_json):
    assert POST_TRATAMIENTO.isdisjoint(pred_json["features"])
    Xtr = features.split(clean)[0]
    assert POST_TRATAMIENTO.isdisjoint(Xtr.columns)


def test_imputacion_y_escalado_aprendidos_solo_en_train(clean):
    Xtr, Xte, ytr, _ = features.split(clean)
    pipe = features.make_pipeline(LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    num = pipe.named_steps["prep"].named_transformers_["num"]
    medianas = num.named_steps[next(iter(num.named_steps))].statistics_
    np.testing.assert_allclose(medianas, Xtr[features.NUM_PRED].median().values)
    escalador = num.steps[-1][1]
    imputado = Xtr[features.NUM_PRED].fillna(Xtr[features.NUM_PRED].median())
    np.testing.assert_allclose(escalador.mean_, imputado.mean().values)


def test_pca_ajustado_solo_en_train(clean):
    pca = joblib.load(MODELS_DIR / "pca.joblib")
    Xtr = features.split(clean)[0]
    P = clean.loc[Xtr.index, PCA_VARS]
    np.testing.assert_allclose(pca.named_steps["escalar"].mean_, P.fillna(P.median()).mean().values)
    full = clean[PCA_VARS].fillna(clean[PCA_VARS].median()).mean().values
    assert not np.allclose(pca.named_steps["escalar"].mean_, full)


def test_categoria_desconocida_no_rompe_el_modelo(clean, pred_json):
    modelo = joblib.load(MODELS_DIR / "modelo_predictivo.joblib")
    Xte = features.split(clean)[1].head(3).copy()
    Xte["marital_cat"] = "categoria_nueva"
    p = modelo.predict_proba(Xte)[:, 1]
    assert np.all((p > 0) & (p < 1))
    assert isinstance(Xte, pd.DataFrame)

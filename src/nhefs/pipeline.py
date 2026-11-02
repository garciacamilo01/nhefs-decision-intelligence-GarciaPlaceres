"""Entrenamiento de los modelos finales y generación de artefactos para la web app.

El notebook maestro y `scripts/build_artifacts.py` usan estas mismas funciones, así que los
números de la app son exactamente los del notebook.
"""

import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import causal, data, features
from .config import COLLIDER, CONFOUNDERS, EFFECT_MODIFIERS, INSTRUMENT, MODELS_DIR, OUTCOME_CAUSAL, PCA_VARS, SEED, TREATMENT

logging.getLogger("dowhy").setLevel(logging.ERROR)


def fit_pca(Xtr_num: pd.DataFrame, n_components: int | None = None) -> Pipeline:
    """Estandarización + PCA ajustados SOLO con train (la matriz de rotación se aplica a ciegas sobre test)."""
    return Pipeline([("escalar", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=SEED))]).fit(Xtr_num)


LOGIT_GRID = {"modelo__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], "modelo__l1_ratio": [0.0, 1.0]}


def logit_base() -> LogisticRegression:
    """Logística con penalización elástica (l1_ratio = 0 → Ridge/L2; l1_ratio = 1 → Lasso/L1)."""
    return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000)


def final_predictive_model(Xtr, ytr, C: float, l1_ratio: float) -> CalibratedClassifierCV:
    """Logística regularizada (hiperparámetros elegidos por CV) con calibración sigmoide por validación cruzada."""
    base = features.make_pipeline(logit_base().set_params(C=C, l1_ratio=l1_ratio))
    return CalibratedClassifierCV(base, method="sigmoid", cv=5).fit(Xtr, ytr)


def predictive_metrics(model, Xte, yte) -> dict:
    p = model.predict_proba(Xte)[:, 1]
    return {"auc_roc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)), "prevalencia_test": float(np.mean(yte))}


def run_causal(dc: pd.DataFrame, d_full: pd.DataFrame, n_sim: int = 100) -> dict:
    """Flujo DoWhy completo + DML. Devuelve un diccionario serializable con todos los resultados."""
    warnings.filterwarnings("ignore")
    from dowhy import CausalModel

    cols = CONFOUNDERS + [TREATMENT, OUTCOME_CAUSAL, INSTRUMENT, COLLIDER]
    model = CausalModel(data=dc[cols], treatment=TREATMENT, outcome=OUTCOME_CAUSAL, graph=causal.dot())
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    est_lr = model.estimate_effect(estimand, method_name="backdoor.linear_regression", confidence_intervals=True)
    est_ipw = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting", target_units="ate",
                                    method_params={"weighting_scheme": "ips_weight"})
    naive = causal.naive(dc)
    boot = causal.bootstrap(dc, causal.ipw_ate, n=300)
    ci_lr = np.ravel(est_lr.get_confidence_intervals())

    refs = {}
    specs = {"Causa común aleatoria": ("random_common_cause", {}),
             "Tratamiento placebo": ("placebo_treatment_refuter", {"placebo_type": "permute"}),
             "Subconjunto de datos (80%)": ("data_subset_refuter", {"subset_fraction": 0.8})}
    for nombre, (metodo, kw) in specs.items():
        r = model.refute_estimate(estimand, est_lr, method_name=metodo, num_simulations=n_sim, random_seed=SEED, **kw)
        p = float(r.refutation_result["p_value"])
        new = float(r.new_effect)
        if metodo == "placebo_treatment_refuter":
            ok = abs(new) < (ci_lr[1] - ci_lr[0]) / 2
        elif metodo == "data_subset_refuter":
            ok = ci_lr[0] <= new <= ci_lr[1]
        else:
            ok = abs(new - est_lr.value) < (ci_lr[1] - ci_lr[0]) / 4
        refs[nombre] = {"efecto_nuevo": new, "p_valor": p, "condicion_ok": bool(ok),
                        "semaforo": causal.refutation_light(p, ok)}

    dml = causal.fit_dml(dc)
    ate_dml = float(dml.ate(X=dc[EFFECT_MODIFIERS]))
    ate_dml_ci = [float(x) for x in dml.ate_interval(X=dc[EFFECT_MODIFIERS], alpha=0.05)]
    grid = causal.cate_grid(dml, dc)

    return {
        "n_causal": int(len(dc)), "n_total": int(len(d_full)),
        "naive": {"ate": float(naive.params[TREATMENT]), "ic": [float(x) for x in naive.conf_int().loc[TREATMENT]]},
        "ols": {"ate": float(est_lr.value), "ic": [float(x) for x in ci_lr]},
        "ipw": {"ate": float(causal.ipw_ate(dc)), "ic": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                "ate_dowhy": float(est_ipw.value)},
        "ipcw": {"ate": causal.ipcw_ate(d_full)},
        "dml": {"ate": ate_dml, "ic": ate_dml_ci, "intercepto": float(dml.intercept_),
                "coef": dict(zip(EFFECT_MODIFIERS, map(float, dml.coef_))),
                "coef_ic": {k: [float(a), float(b)] for k, a, b in zip(EFFECT_MODIFIERS, *dml.coef__interval(alpha=0.05))}},
        "refutaciones": refs,
        "cate_grid": grid.round(4).to_dict(orient="list"),
        "dag": {"edges": causal.edges(include_latent=True), "roles": causal.ROLES, "labels": causal.LABELS},
        "estimando": str(estimand).split("### Estimand : 2")[0],
    }


def save_artifacts(pred_model, pred_info: dict, pca_pipe, causal_res: dict, d_clean: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pred_model, MODELS_DIR / "modelo_predictivo.joblib")
    joblib.dump(pca_pipe, MODELS_DIR / "pca.joblib")
    (MODELS_DIR / "predictivo.json").write_text(json.dumps(pred_info, ensure_ascii=False, indent=1))
    (MODELS_DIR / "causal.json").write_text(json.dumps(causal_res, ensure_ascii=False, indent=1))
    keep = list(dict.fromkeys(PCA_VARS + features.NUM_PRED + features.CAT_ALL + features.BIN_PRED
                              + ["seqn", "death", OUTCOME_CAUSAL, "censurado"]))
    d_clean[keep].to_csv(MODELS_DIR / "datos_app.csv", index=False)


def build_all() -> dict:
    """Reconstruye todos los artefactos de la app desde cero (usado por scripts/build_artifacts.py)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    d = data.clean(data.load_raw())
    dc = data.causal_sample(d)
    Xtr, Xte, ytr, yte = features.split(d)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(features.make_pipeline(logit_base()), LOGIT_GRID, cv=skf, scoring="neg_brier_score").fit(Xtr, ytr)
    C, l1 = grid.best_params_["modelo__C"], grid.best_params_["modelo__l1_ratio"]
    model = final_predictive_model(Xtr, ytr, C, l1)
    info = {"modelo": "Regresión logística regularizada y calibrada (sigmoide)", "C": C, "l1_ratio": l1,
            "metricas_test": predictive_metrics(model, Xte, yte),
            "features": list(Xtr.columns), "medianas_train": Xtr.median(numeric_only=True).to_dict(),
            "modas_train": Xtr.mode().iloc[0].astype(str).to_dict()}
    pca_pipe = fit_pca(d.loc[Xtr.index, PCA_VARS].fillna(d.loc[Xtr.index, PCA_VARS].median()))
    res = run_causal(dc, d)
    save_artifacts(model, info, pca_pipe, res, d)
    return {"pred": info, "causal": res}

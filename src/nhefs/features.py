"""Partición de datos y preprocesamiento sin fuga de información (data leakage)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BIN_PRED, CAT_PRED, NUM_PRED, OUTCOME_PRED, SEED

CAT_ALL = CAT_PRED + ["diabetes_cat", "hbp_cat"]


def split(d: pd.DataFrame, test_size: float = 0.20):
    """Partición 80/20 estratificada por el resultado predictivo (muerte), con semilla fija."""
    X = d[NUM_PRED + CAT_ALL + BIN_PRED]
    y = d[OUTCOME_PRED]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Imputación + escalado + one-hot. Se ajusta SIEMPRE dentro del Pipeline (solo con datos de train)."""
    num_steps = [("imputar", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_PRED),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                                   sparse_output=False))]), CAT_ALL),
        ("bin", "passthrough", BIN_PRED),
    ], verbose_feature_names_out=False)


def make_pipeline(model, scale: bool = True) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(scale)), ("modelo", model)])

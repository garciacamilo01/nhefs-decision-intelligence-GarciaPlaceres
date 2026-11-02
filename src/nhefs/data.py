"""Carga, auditoría de calidad y limpieza del dataset NHEFS."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW, OUTCOME_CAUSAL, URL_NHEFS

MARITAL = {2: "casado", 3: "viudo", 4: "soltero", 5: "divorciado", 6: "separado"}
SI_NO_NI = {0: "no", 1: "si", 2: "no_informado"}


def load_raw(path: Path | str = DATA_RAW) -> pd.DataFrame:
    """Lee el CSV crudo; si no existe localmente, lo descarga de la fuente pública."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(URL_NHEFS).to_csv(path, index=False)
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de hallazgos de calidad: faltantes explícitos, códigos ocultos y faltantes estructurales."""
    filas = []
    for col, n in df.isna().sum().items():
        if n:
            filas.append({"variable": col, "tipo": "NaN explícito", "casos": int(n), "pct": 100 * n / len(df)})
    for col in ["diabetes", "hbp", "hbpmed", "pica", "boweltrouble", "birthcontrol", "alcoholpy"]:
        n = int((df[col] == 2).sum())
        filas.append({"variable": col, "tipo": "código 2 = 'MISSING' (oculto)", "casos": n, "pct": 100 * n / len(df)})
    filas.append({"variable": "alcoholfreq", "tipo": "código 5 = 'Unknown' (oculto)",
                  "casos": int((df["alcoholfreq"] == 5).sum()), "pct": 100 * (df["alcoholfreq"] == 5).mean()})
    filas.append({"variable": "marital", "tipo": "código 8 = 'Unknown' (oculto)",
                  "casos": int((df["marital"] == 8).sum()), "pct": 100 * (df["marital"] == 8).mean()})
    return pd.DataFrame(filas).sort_values("casos", ascending=False).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Recodifica códigos ocultos de faltantes y crea variables derivadas.

    - Los códigos 2 = 'MISSING' del bloque de salud se mantienen como categoría explícita
      ('no_informado'): el faltante es por diseño (cuestionario distinto para seqn >= 20751).
    - alcoholfreq = 5 y marital = 8 ('Unknown') pasan a NaN.
    - censurado = 1 si no se midió el peso en 1982 (resultado causal faltante).
    - edad_inicio = edad en la que empezó a fumar (age − smokeyrs), para romper la colinealidad age/smokeyrs.
    """
    d = df.copy()
    d["marital_cat"] = d["marital"].map(MARITAL)
    d["alcoholfreq"] = d["alcoholfreq"].where(d["alcoholfreq"] != 5)
    for v in ["diabetes", "hbp", "hbpmed", "pica", "boweltrouble"]:
        d[f"{v}_cat"] = d[v].map(SI_NO_NI)
    d["bloque_no_informado"] = (d["diabetes"] == 2).astype(int)
    d["censurado"] = d[OUTCOME_CAUSAL].isna().astype(int)
    d["edad_inicio"] = d["age"] - d["smokeyrs"]
    for c in ["education", "exercise", "active"]:
        d[c] = d[c].astype(int)
    return d


def causal_sample(d: pd.DataFrame) -> pd.DataFrame:
    """Muestra para la pregunta causal: participantes con el peso de 1982 observado (no censurados)."""
    return d[d["censurado"] == 0].reset_index(drop=True)


def missing_vs_observed(d: pd.DataFrame, flag: str, cols: list[str]) -> pd.DataFrame:
    """Compara medias de covariables entre registros con y sin faltante (diagnóstico MCAR/MAR)."""
    from scipy import stats

    filas = []
    for c in cols:
        a, b = d.loc[d[flag] == 1, c].dropna(), d.loc[d[flag] == 0, c].dropna()
        t = stats.ttest_ind(a, b, equal_var=False)
        filas.append({"variable": c, "media con faltante": a.mean(), "media sin faltante": b.mean(),
                      "diferencia": a.mean() - b.mean(), "p-valor (Welch)": t.pvalue})
    return pd.DataFrame(filas).set_index("variable")


def summary_numbers(d: pd.DataFrame) -> dict:
    """Números de control rápidos (usados por los tests de integridad)."""
    return {"n": len(d), "n_censurados": int(d["censurado"].sum()),
            "prev_qsmk": float(d["qsmk"].mean()), "prev_death": float(d["death"].mean()),
            "media_wt82_71": float(np.nanmean(d[OUTCOME_CAUSAL]))}

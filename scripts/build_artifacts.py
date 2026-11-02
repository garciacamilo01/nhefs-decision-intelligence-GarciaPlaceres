"""Regenera todos los artefactos de /models que consume la web app.

Uso:  python scripts/build_artifacts.py
Descarga data/raw/nhefs.csv si falta. Tarda ~3–5 min (refutaciones DoWhy + DML).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nhefs.pipeline import build_all  # noqa: E402

if __name__ == "__main__":
    res = build_all()
    m = res["pred"]["metricas_test"]
    c = res["causal"]
    print(f"Predictivo  → AUC {m['auc_roc']:.3f} | Brier {m['brier']:.3f}")
    print(f"Causal      → ATE OLS {c['ols']['ate']:.2f} kg | DML {c['dml']['ate']:.2f} kg")
    print("Refutaciones:", {k: v["semaforo"] for k, v in c["refutaciones"].items()})

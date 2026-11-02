"""Figuras de la presentación ejecutiva (se generan a partir de los mismos artefactos que la app)."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhefs import data, features  # noqa: E402
from nhefs.config import PCA_VARS  # noqa: E402

OUT = ROOT / "slides" / "img"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, ACC, ACC2, GRID = "#1f2933", "#7b8794", "#d9480f", "#1c7ed6", "#e4e7eb"
plt.rcParams.update({"font.family": "Inter", "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "figure.dpi": 200, "savefig.bbox": "tight"})

C = json.loads((ROOT / "models/causal.json").read_text())
d = data.clean(data.load_raw())
dc = data.causal_sample(d)

# 1. distribución del cambio de peso
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for t, col, lab in [(0, MUTED, "Siguió fumando"), (1, ACC, "Dejó de fumar")]:
    y = dc.loc[dc.qsmk == t, "wt82_71"]
    ax.hist(y, bins=np.arange(-42, 50, 2.5), density=True, alpha=0.45, color=col, label=f"{lab} (n={len(y)})")
    ax.axvline(y.mean(), color=col, lw=2.5)
    ax.text(y.mean() + (0.8 if t else -0.8), 0.083, f"media {y.mean():+.2f} kg", color=col, fontsize=12, weight="bold", ha="left" if t else "right")
ax.set_xlim(-25, 35); ax.set_xlabel("Cambio de peso 1971→1982 (kg)"); ax.set_yticks([]); ax.set_ylim(0, 0.092); ax.legend(frameon=False, loc="center right", fontsize=11)
fig.savefig(OUT / "f_peso.png"); plt.close(fig)

# 2. medias por educación (ANOVA de Welch)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
g = dc.groupby("education")["wt82_71"].agg(["mean", "sem", "count"])
lab = ["≤8.º", "Sec. inc.", "Sec. comp.", "Univ. inc.", "Univ. comp."]
ax.errorbar(range(5), g["mean"], yerr=1.96 * g["sem"], fmt="o", color=ACC2, ms=9, capsize=6, lw=2)
ax.axhline(dc.wt82_71.mean(), color=MUTED, ls="--", lw=1)
ax.set_xticks(range(5), [f"{a}\nn={n}" for a, n in zip(lab, g["count"])]); ax.set_ylabel("Cambio de peso medio (kg)")
fig.savefig(OUT / "f_anova.png"); plt.close(fig)

# 3. PCA (train) coloreado por muerte
pca = joblib.load(ROOT / "models/pca.joblib")
Xtr = features.split(d)[0]
P = d.loc[Xtr.index, PCA_VARS]; Z = pca.transform(P.fillna(P.median()))
yd = d.loc[Xtr.index, "death"].values
evr = pca.named_steps["pca"].explained_variance_ratio_
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.scatter(Z[yd == 0, 0], Z[yd == 0, 1], s=9, c=MUTED, alpha=0.35, label="Vivo 1992")
ax.scatter(Z[yd == 1, 0], Z[yd == 1, 1], s=12, c=ACC, alpha=0.7, label="Murió 1983–92")
ax.set_xlabel(f"PC1 · edad y exposición al tabaco ({evr[0]:.0%})"); ax.set_ylabel(f"PC2 · tamaño corporal ({evr[1]:.0%})")
ax.legend(frameon=False, markerscale=2)
fig.savefig(OUT / "f_pca.png"); plt.close(fig)

# 4. torneo de modelos (CV) + calibración en test
cv = pd.DataFrame({"modelo": ["Logística\nregularizada", "Random\nForest", "CART\npodado", "Naive\nBayes"],
                   "auc": [0.838, 0.834, 0.776, 0.756], "brier": [0.1125, 0.118, 0.130, 0.202]})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
cols = [ACC, MUTED, MUTED, MUTED]
axs[0].barh(cv.modelo[::-1], cv.auc[::-1], color=cols[::-1]); axs[0].set_xlim(0.7, 0.86); axs[0].set_title("AUC (CV 5 pliegues) ↑", fontsize=13)
axs[1].barh(cv.modelo[::-1], cv.brier[::-1], color=cols[::-1]); axs[1].axvline(0.158, color=INK, ls="--", lw=1.2)
axs[1].text(0.160, 3.35, "línea base\n0,158", fontsize=10, color=INK); axs[1].set_title("Brier (CV) ↓", fontsize=13); axs[1].set_yticks([])
for ax, k, f in [(axs[0], "auc", "{:.3f}"), (axs[1], "brier", "{:.3f}")]:
    for i, v in enumerate(cv[k][::-1]):
        ax.text(v, i, " " + f.format(v), va="center", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "f_torneo.png"); plt.close(fig)

modelo = joblib.load(ROOT / "models/modelo_predictivo.joblib")
_, Xte, _, yte = features.split(d)
p = modelo.predict_proba(Xte)[:, 1]
fy, fx = calibration_curve(yte, p, n_bins=8, strategy="quantile")
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
ax.plot(fx, fy, "o-", color=ACC, lw=2.5, ms=7)
ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.75); ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
ax.set_title("Diagrama de fiabilidad (test)", fontsize=13)
fig.savefig(OUT / "f_calib.png"); plt.close(fig)

# 5. forest plot de ATE
est = [("Ingenuo (sin ajuste)", C["naive"]["ate"], C["naive"]["ic"], MUTED),
       ("Backdoor · regresión", C["ols"]["ate"], C["ols"]["ic"], ACC),
       ("Backdoor · IPW", C["ipw"]["ate"], C["ipw"]["ic"], ACC),
       ("Double ML (EconML)", C["dml"]["ate"], C["dml"]["ic"], ACC)]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for i, (n, a, (lo, hi), col) in enumerate(est[::-1]):
    ax.plot([lo, hi], [i, i], color=col, lw=3); ax.plot(a, i, "o", color=col, ms=11)
    ax.text(hi + 0.1, i, f"{a:.2f} [{lo:.2f}; {hi:.2f}]", va="center", fontsize=11, color=INK)
ax.set_yticks(range(4), [e[0] for e in est[::-1]]); ax.set_xlim(1.3, 6.2); ax.set_xlabel("kg adicionales por dejar de fumar")
ax.grid(axis="y", visible=False)
fig.savefig(OUT / "f_ate.png"); plt.close(fig)

# 6. CATE
grid = pd.DataFrame(C["cate_grid"])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for age, col in [(30, ACC2), (60, ACC)]:
    a = grid.age.unique()[np.argmin(abs(grid.age.unique() - age))]
    s = grid[grid.age == a]
    ax.fill_between(s.smokeintensity, s.ic_inf, s.ic_sup, color=col, alpha=0.15)
    ax.plot(s.smokeintensity, s.cate, color=col, lw=2.5, label=f"Edad {a:.0f}")
ax.axhline(C["ols"]["ate"], color=INK, ls="--", lw=1); ax.text(grid.smokeintensity.max(), C["ols"]["ate"] + 0.15, "ATE", ha="right", fontsize=10)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("Cigarrillos por día en 1971"); ax.set_ylabel("CATE (kg)"); ax.legend(frameon=False, loc="upper left")
fig.savefig(OUT / "f_cate.png"); plt.close(fig)

# 7. DAG con graphviz
roles, labels = C["dag"]["roles"], C["dag"]["labels"]
COL = {"Confusor": "#c5d4f0", "Tratamiento": "#ffb48a", "Resultado": "#9fe0c7", "Instrumento": "#f0dfb8",
       "Colisionador": "#f3b6da", "Mediador (latente)": "#ffe680", "Confusor no observado": "#eeeeee", "Selección (censura)": "#d5d5d5"}
lat = {"ingesta_metabolismo", "conciencia_salud"}
conf = [n for n, r in roles.items() if r == "Confusor"]
lines = ['digraph { rankdir=LR; splines=true; ranksep=1.1; nodesep=0.12; bgcolor="transparent";',
         'node [shape=box, style="rounded,filled", fontname="Inter", fontsize=13, color="#9aa5b1", penwidth=1];',
         'edge [color="#9aa5b1", arrowsize=0.6];',
         'subgraph cluster_z { label="9 confusores (1971)"; fontname="Inter"; fontsize=12; style="rounded,dashed"; color="#9aa5b1"; '
         + " ".join(f'"{z}";' for z in conf) + "}"]
nodes = {n for e in C["dag"]["edges"] for n in e}
for n in nodes:
    st = "rounded,filled,dashed" if n in lat else "rounded,filled"
    lines.append(f'"{n}" [label="{labels.get(n, n)}", fillcolor="{COL.get(roles.get(n), "#fff")}", style="{st}"];')
for a, b in C["dag"]["edges"]:
    if a in conf and b in conf:
        continue
    st = "dashed" if a in lat or b in lat else "solid"
    w = ', color="#d9480f", penwidth=2.5' if (a, b) == ("qsmk", "wt82_71") else ""
    lines.append(f'"{a}" -> "{b}" [style={st}{w}];')
lines.append("}")
subprocess.run(["dot", "-Tpng", "-Gdpi=200", "-o", str(OUT / "f_dag.png")], input="\n".join(lines).encode(), check=True)
print("ok")

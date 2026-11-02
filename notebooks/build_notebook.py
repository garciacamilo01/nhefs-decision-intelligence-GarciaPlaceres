"""Genera notebooks/TP_Integrador_Final.ipynb a partir de celdas definidas acá (texto en textos.json)."""

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
TXT = json.loads((HERE / "textos.json").read_text(encoding="utf-8")) if (HERE / "textos.json").exists() else {}


def T(k):
    return TXT.get(k, f"*(pendiente: {k})*")


nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))


# =============================================================================== PORTADA
md(r"""
# Trabajo Práctico Integrador — Decision Intelligence Capstone
## Dejar de fumar: ¿cuánto peso se gana y quién está en riesgo? Predicción, inferencia y causalidad con NHEFS

**Asignatura:** Estadística Aplicada y Modelos de Regresión (EAyMR 2026) · UCA — Facultad de Ingeniería y Ciencias Agrarias
**Autor:** Camilo Garcia Placeres
**Roles Scrum:** proyecto individual; el autor asume los cuatro roles (Project Lead, Backend/ML-Causal, Frontend, QA). Ver `BITACORA.md`.
**Semilla global:** `SEED = 42` · **Dataset:** NHEFS — *NHANES I Epidemiologic Follow-up Study* (1971–1992), N = 1.629, 64 variables.

> **Declaración de uso de IA generativa.** Este proyecto se desarrolló con asistencia de un modelo de IA (Claude, Anthropic) para escribir código, redactar y revisar. El detalle de qué se delegó, qué errores de la IA se detectaron y qué pruebas determinísticas se escribieron para controlarlos está en la Entrada 5 de `BITACORA.md`.

### Índice
0. [Marco del problema y preguntas de decisión](#fase0)
1. [FASE 1 — Inferencia estadística clásica, contrastes y ANOVA](#fase1)
2. [FASE 2 — Gauss-Markov, puntos influyentes, VIF y PCA sin *leakage*](#fase2)
3. [FASE 3 — Torneo predictivo, calibración y explicabilidad](#fase3)
4. [FASE 4 — Inferencia causal: DAG, *backdoor*, DoWhy, Double ML y refutaciones](#fase4)
5. [Ensayo crítico: capacidad predictiva vs. palanca causal](#ensayo)
6. [Conclusiones, recomendaciones y limitaciones](#conclusiones)
7. [Referencias](#referencias)
""")

code(r"""
import sys, warnings, logging
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger("dowhy").setLevel(logging.ERROR)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower

from nhefs import data, features, causal, pipeline
from nhefs.config import (SEED, TREATMENT, OUTCOME_CAUSAL, OUTCOME_PRED, CONFOUNDERS, EFFECT_MODIFIERS,
                          INSTRUMENT, COLLIDER, PCA_VARS, NUM_PRED, BIN_PRED)

np.random.seed(SEED)
sns.set_theme(style="whitegrid")
pd.set_option("display.float_format", "{:.4f}".format)
plt.rcParams["figure.dpi"] = 100
ALPHA = 0.05   # nivel de significación fijado a priori para todo el trabajo
print("Raíz del proyecto:", ROOT)
""")

# =============================================================================== FASE 0
md(r"""
<a id="fase0"></a>
---
# 0. Marco del problema y preguntas de decisión
""" + "\n" + T("F0"))

code(r"""
df_raw = data.load_raw(ROOT / "data" / "raw" / "nhefs.csv")
codebook = pd.read_csv(ROOT / "data" / "raw" / "NHEFS_codebook.csv")
print(f"Dimensiones: {df_raw.shape[0]} participantes x {df_raw.shape[1]} variables")

roles = pd.DataFrame([
    ("qsmk", "Tratamiento T", "Binaria", "Dejó de fumar entre 1971 y 1982"),
    ("wt82_71", "Resultado causal Y", "Continua (kg)", "Cambio de peso 1971 → 1982"),
    ("death", "Resultado predictivo", "Binaria", "Muerte entre 1983 y 1992"),
    ("sex, race", "Confusores Z", "Binarias nominales", "Sexo, raza"),
    ("age, smokeyrs, smokeintensity, wt71", "Confusores Z", "Continuas / discretas", "Edad, años fumando, cigarrillos/día, peso 1971"),
    ("education, exercise, active", "Confusores Z", "Ordinales (codificadas)", "Educación (1–5), ejercicio (0–2), actividad diaria (0–2)"),
    ("price82, tax82", "Instrumento candidato", "Continuas (US$ 2008)", "Precio e impuesto al tabaco en el estado (1982)"),
    ("sbp, dbp", "Colisionador C (post-tratamiento)", "Continuas (mmHg)", "Presión arterial medida en 1982"),
    ("ht, cholesterol, income, alcohol…, dx clínicos", "Predictoras basales", "Mixtas", "Altura, colesterol, ingreso, alcohol, 20 antecedentes clínicos"),
], columns=["Variable(s)", "Rol", "Escala", "Descripción"])
roles.style.hide(axis="index")
""")

# =============================================================================== FASE 1
md(r"""
<a id="fase1"></a>
---
# FASE 1 — Inferencia estadística clásica, contrastes y ANOVA

## 1.1 Auditoría forense de datos (DGP)
""")

code(r"""
auditoria = data.audit(df_raw)
display(auditoria.style.format({"pct": "{:.1f}%"}).hide(axis="index"))
d = data.clean(df_raw)

bloque = d["bloque_no_informado"] == 1
print(f"Bloque de 5 variables clínicas con código 2 ('MISSING'): {bloque.sum()} participantes")
print(f"  seqn de esos participantes: {d.loc[bloque, 'seqn'].min()} a {d.loc[bloque, 'seqn'].max()}"
      f"  |  resto: {d.loc[~bloque, 'seqn'].min()} a {d.loc[~bloque, 'seqn'].max()}")
print(f"  ¿se superponen los rangos de seqn? {d.loc[bloque, 'seqn'].min() <= d.loc[~bloque, 'seqn'].max()}")
print(f"pregnancies faltante: hombres {d.loc[d.sex == 0, 'pregnancies'].isna().mean():.0%} | mujeres {d.loc[d.sex == 1, 'pregnancies'].isna().mean():.0%}")
print(f"alcoholhowmuch faltante entre quienes no bebieron (alcoholpy = 0): {d.loc[d.alcoholpy == 0, 'alcoholhowmuch'].isna().mean():.0%}")
""")

md(r"""
**Mecanismo del faltante en el resultado causal.** El peso de 1982 falta en 63 participantes (3,9%): son **censurados** (no volvieron a la medición de 1982). Comparo sus características basales con las de los observados y estimo un modelo logístico de la probabilidad de censura.
""")

code(r"""
cmp = data.missing_vs_observed(d, "censurado", ["age", "wt71", "smokeintensity", "smokeyrs", "qsmk", "sex", "race", "education"])
display(cmp.style.format("{:.3f}"))
m_cens = smf.logit("censurado ~ qsmk + sex + race + age + education + smokeintensity + smokeyrs + exercise + active + wt71", d).fit(disp=0)
print(m_cens.summary2().tables[1][["Coef.", "P>|z|"]].round(4).to_string())
print(f"\nTest LR global del modelo de censura: χ² = {m_cens.llr:.2f}, gl = {int(m_cens.df_model)}, p = {m_cens.llr_pvalue:.4f}")
print(f"Tasa de muerte entre censurados: {d.loc[d.censurado == 1, 'death'].mean():.1%} vs. observados {d.loc[d.censurado == 0, 'death'].mean():.1%}")
""")

md(r"""
### 1.1 · Clasificación del mecanismo de faltantes y estrategia
""" + "\n" + T("F1_missing"))

md(r"""
## 1.2 Exploración: distribuciones y correlaciones
""")

code(r"""
dc = data.causal_sample(d)
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for t, col, lab in [(0, "#66c2a5", "Siguió fumando"), (1, "#fc8d62", "Dejó de fumar")]:
    s = dc[dc[TREATMENT] == t]
    sns.kdeplot(s[OUTCOME_CAUSAL], ax=axes[0, 0], fill=True, color=col, label=f"{lab} (n={len(s)})")
    sns.kdeplot(s["age"], ax=axes[0, 1], fill=True, color=col, label=lab)
    sns.kdeplot(s["smokeintensity"], ax=axes[0, 2], fill=True, color=col, label=lab, clip=(0, 80))
axes[0, 0].set_title("Cambio de peso 1971–1982 (kg)"); axes[0, 0].legend()
axes[0, 1].set_title("Edad en 1971"); axes[0, 1].legend()
axes[0, 2].set_title("Cigarrillos por día en 1971"); axes[0, 2].legend()
sns.boxplot(data=dc, x="education", y=OUTCOME_CAUSAL, ax=axes[1, 0], color="#8da0cb")
axes[1, 0].set_title("Cambio de peso por nivel educativo"); axes[1, 0].set_xlabel("Educación (1 = ≤ 8.º grado … 5 = universitario)")
sns.histplot(d["wt71"], kde=True, ax=axes[1, 1], color="#a6d854"); axes[1, 1].set_title("Peso en 1971 (kg)")
stats.probplot(dc[OUTCOME_CAUSAL], dist="norm", plot=axes[1, 2]); axes[1, 2].set_title("Q-Q del cambio de peso")
plt.tight_layout(); plt.show()
print(f"Cambio de peso: media {dc[OUTCOME_CAUSAL].mean():.2f} kg, mediana {dc[OUTCOME_CAUSAL].median():.2f}, "
      f"asimetría {stats.skew(dc[OUTCOME_CAUSAL]):.2f}, curtosis (exceso) {stats.kurtosis(dc[OUTCOME_CAUSAL]):.2f}")
""")

code(r"""
vars_corr = [OUTCOME_CAUSAL, TREATMENT, "age", "smokeyrs", "smokeintensity", "wt71", "ht", "education", "school", "income", "exercise", "active", "cholesterol", "sbp", "death"]
fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, met in zip(axes, ["pearson", "spearman"]):
    sns.heatmap(dc[vars_corr].corr(method=met), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7}, cbar=False)
    ax.set_title(f"Correlación de {met.capitalize()}")
plt.tight_layout(); plt.show()
dif = (dc[vars_corr].corr("spearman") - dc[vars_corr].corr("pearson")).abs()
i, j = np.unravel_index(np.argmax(dif.values), dif.shape)
print(f"Mayor diferencia Spearman − Pearson: {vars_corr[i]} vs {vars_corr[j]} ({dif.values[i, j]:.2f})")
""")

md(r"""
## 1.3 Contrastes de hipótesis (α = 0,05 fijado a priori)

**H1 (pregunta de negocio):** ¿quienes dejaron de fumar ganaron más peso que quienes siguieron fumando?
- $H_0: \mu_{\Delta peso}^{dejó} = \mu_{\Delta peso}^{siguió}$ vs. $H_1: \mu_{\Delta peso}^{dejó} \neq \mu_{\Delta peso}^{siguió}$ (bilateral).
- Estadístico: **t de Welch** (no supone varianzas iguales; con n > 400 por grupo el TCL hace robusta la inferencia sobre medias) y, como control no paramétrico, **Mann-Whitney U**.

**H2 (autoselección):** ¿quienes dejaron de fumar eran distintos *antes* del tratamiento? Se contrastan la **edad** (t de Welch) y la **intensidad de consumo** (Mann-Whitney, por su fuerte asimetría). Si difieren, la comparación de H1 está confundida.
""")

code(r"""
def contraste(x1, x0, nombre):
    w = stats.ttest_ind(x1, x0, equal_var=False)
    mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
    dif = x1.mean() - x0.mean()
    se = np.sqrt(x1.var(ddof=1) / len(x1) + x0.var(ddof=1) / len(x0))
    gl = w.df
    ic = dif + np.array([-1, 1]) * stats.t.ppf(0.975, gl) * se
    sp = np.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x0) - 1) * x0.var(ddof=1)) / (len(x1) + len(x0) - 2))
    rbc = 1 - 2 * mw.statistic / (len(x1) * len(x0))
    return {"Contraste": nombre, "media T=1": x1.mean(), "media T=0": x0.mean(), "diferencia": dif,
            "IC95 inf": ic[0], "IC95 sup": ic[1], "t Welch": w.statistic, "gl": gl, "p Welch": w.pvalue,
            "p Mann-Whitney": mw.pvalue, "d de Cohen": dif / sp, "r biserial (rangos)": -rbc}

g1, g0 = dc[dc[TREATMENT] == 1], dc[dc[TREATMENT] == 0]
tabla_h = pd.DataFrame([
    contraste(g1[OUTCOME_CAUSAL], g0[OUTCOME_CAUSAL], "H1: cambio de peso (kg)"),
    contraste(g1["age"], g0["age"], "H2a: edad en 1971"),
    contraste(g1["smokeintensity"], g0["smokeintensity"], "H2b: cigarrillos/día 1971"),
]).set_index("Contraste")
display(tabla_h.style.format("{:.4f}").format({"p Welch": "{:.2e}", "p Mann-Whitney": "{:.2e}", "gl": "{:.0f}"}))

pot = TTestIndPower()
d_obs = tabla_h.loc["H1: cambio de peso (kg)", "d de Cohen"]
print(f"Potencia alcanzada para detectar d = {d_obs:.2f} con n1 = {len(g1)}, n0 = {len(g0)}: "
      f"{pot.power(effect_size=d_obs, nobs1=len(g1), ratio=len(g0)/len(g1), alpha=ALPHA):.3f}")
print(f"n por grupo necesario para detectar 1 kg (d = {1/dc[OUTCOME_CAUSAL].std():.3f}) con potencia 0,80: "
      f"{pot.solve_power(effect_size=1/dc[OUTCOME_CAUSAL].std(), alpha=ALPHA, power=0.8):.0f}")
""")

md(r"""
### 1.3 · Interpretación
""" + "\n" + T("F1_tests"))

md(r"""
## 1.4 ANOVA de un factor: cambio de peso según nivel educativo (5 grupos)

$H_0: \mu_1 = \dots = \mu_5$ vs. $H_1$: al menos una media difiere. Primero se verifican los supuestos, con el criterio que pide la consigna para muestras grandes.
""")

code(r"""
grupos = {k: g[OUTCOME_CAUSAL].values for k, g in dc.groupby("education")}
diag = pd.DataFrame({k: {"n": len(v), "media": v.mean(), "desvío": v.std(ddof=1), "asimetría": stats.skew(v),
                         "curtosis (exceso)": stats.kurtosis(v), "Shapiro p": stats.shapiro(v).pvalue}
                     for k, v in grupos.items()}).T
diag.index.name = "educación"
display(diag.style.format({"n": "{:.0f}", "Shapiro p": "{:.2e}"}, precision=3))
lev = stats.levene(*grupos.values(), center="median")
print(f"Levene (Brown-Forsythe): W = {lev.statistic:.3f}, p = {lev.pvalue:.4f}  |  cociente var. máx/mín = {diag['desvío'].max()**2 / diag['desvío'].min()**2:.2f}")

fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
for ax, (k, v) in zip(axes, grupos.items()):
    stats.probplot(v, dist="norm", plot=ax); ax.set_title(f"Educación {k} (n = {len(v)})"); ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("Q-Q Plots del cambio de peso por grupo", y=1.03); plt.tight_layout(); plt.show()
""")

code(r"""
f_clas = stats.f_oneway(*grupos.values())
welch = anova_oneway(dc[OUTCOME_CAUSAL], dc["education"], use_var="unequal", welch_correction=True)
kw = stats.kruskal(*grupos.values())
ss_between = sum(len(v) * (v.mean() - dc[OUTCOME_CAUSAL].mean()) ** 2 for v in grupos.values())
eta2 = ss_between / ((dc[OUTCOME_CAUSAL] - dc[OUTCOME_CAUSAL].mean()) ** 2).sum()
anova_tab = pd.DataFrame({
    "Estadístico": [f_clas.statistic, welch.statistic, kw.statistic],
    "gl": [f"{len(grupos)-1}, {len(dc)-len(grupos)}", f"{welch.df[0]:.0f}, {welch.df[1]:.1f}", f"{len(grupos)-1}"],
    "p-valor": [f_clas.pvalue, welch.pvalue, kw.pvalue],
    "Supuesto de varianzas": ["Iguales", "Distintas (robusto)", "No paramétrico"],
}, index=["ANOVA clásico (F)", "Welch ANOVA", "Kruskal-Wallis (H)"])
display(anova_tab.style.format({"Estadístico": "{:.3f}", "p-valor": "{:.4f}"}))
print(f"η² (proporción de la varianza explicada por la educación) = {eta2:.4f}")
""")

code(r"""
tukey = pairwise_tukeyhsd(dc[OUTCOME_CAUSAL], dc["education"], alpha=ALPHA)
tk = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0]).rename(columns={"p-adj": "p Tukey"})
pares = []
for (a, b) in [(int(r.group1), int(r.group2)) for r in tk.itertuples()]:
    t_ = stats.ttest_ind(grupos[a], grupos[b], equal_var=False)
    pares.append(min(t_.pvalue * len(tk), 1.0))
tk["p Bonferroni (Welch)"] = pares
tk["Difieren (Tukey)"] = tk["reject"].map({True: "Sí", False: "No"})
tk["Difieren (Bonferroni)"] = np.where(tk["p Bonferroni (Welch)"] < ALPHA, "Sí", "No")
tk.drop(columns="reject").style.format({"meandiff": "{:.3f}", "p Tukey": "{:.4f}", "lower": "{:.3f}", "upper": "{:.3f}", "p Bonferroni (Welch)": "{:.4f}"}).hide(axis="index")
""")

md(r"""
### 1.4 · Interpretación
""" + "\n" + T("F1_anova"))

# =============================================================================== FASE 2
md(r"""
<a id="fase2"></a>
---
# FASE 2 — Diagnóstico de Gauss-Markov, puntos influyentes, VIF y PCA

## 2.1 Modelo lineal de referencia (MCO)

$$\Delta peso_i = \beta_0 + \beta_1\,qsmk_i + \sum_k \beta_k\, X_{ik} + \varepsilon_i$$

con las covariables basales de 1971. Es un modelo **descriptivo/predictivo** de referencia; el efecto causal se trata en la Fase 4.
""")

code(r"""
FORM_MCO = (f"{OUTCOME_CAUSAL} ~ qsmk + sex + race + age + smokeyrs + smokeintensity + wt71 + ht + school + C(education)"
            " + income + C(exercise) + C(active) + cholesterol + alcoholfreq")
d2 = dc.dropna(subset=["income", "cholesterol", "alcoholfreq"]).reset_index(drop=True)
mco = smf.ols(FORM_MCO, data=d2).fit()
print(f"n = {int(mco.nobs)} | R² = {mco.rsquared:.4f} | R² ajustado = {mco.rsquared_adj:.4f} | "
      f"F = {mco.fvalue:.2f} (p = {mco.f_pvalue:.2e}) | RMSE = {np.sqrt(mco.mse_resid):.3f} kg")
coefs = pd.DataFrame({"β̂": mco.params, "EE": mco.bse, "t": mco.tvalues, "p-valor": mco.pvalues})
coefs.style.format({"β̂": "{:.4f}", "EE": "{:.4f}", "t": "{:.2f}", "p-valor": "{:.4f}"})
""")

md(r"""
## 2.2 Auditoría de los supuestos de Gauss-Markov
""")

code(r"""
res = mco.resid
bp = het_breuschpagan(res, mco.model.exog)
jb = jarque_bera(res)
dw = durbin_watson(res)
sup = pd.DataFrame({
    "Estadístico": [bp[0], jb[0], dw],
    "p-valor": [bp[1], jb[1], np.nan],
    "Lectura": ["H0: homocedasticidad", "H0: residuos normales", "≈ 2 → sin autocorrelación (orden por seqn)"],
}, index=["Breusch-Pagan", "Jarque-Bera", "Durbin-Watson"])
display(sup.style.format({"Estadístico": "{:.3f}", "p-valor": "{:.4f}"}, na_rep="—"))
print(f"Residuos: asimetría {stats.skew(res):.3f}, curtosis (exceso) {stats.kurtosis(res):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].scatter(mco.fittedvalues, res, s=8, alpha=0.4); axes[0].axhline(0, color="red", ls="--")
lw = sm.nonparametric.lowess(res, mco.fittedvalues, frac=0.3); axes[0].plot(lw[:, 0], lw[:, 1], color="orange", lw=2)
axes[0].set_xlabel("ŷ (kg)"); axes[0].set_ylabel("Residuo (kg)"); axes[0].set_title("Residuos vs. ajustados")
stats.probplot(res, dist="norm", plot=axes[1]); axes[1].set_title("Q-Q de los residuos")
axes[2].scatter(mco.fittedvalues, np.sqrt(np.abs(mco.get_influence().resid_studentized_internal)), s=8, alpha=0.4)
axes[2].set_xlabel("ŷ (kg)"); axes[2].set_ylabel("√|residuo estandarizado|"); axes[2].set_title("Scale-Location")
plt.tight_layout(); plt.show()

hc3 = mco.get_robustcov_results(cov_type="HC3")
print(f"EE de qsmk: clásico {mco.bse['qsmk']:.4f} vs. robusto HC3 {hc3.bse[list(mco.params.index).index('qsmk')]:.4f}")
""")

md(r"""
## 2.3 Observaciones extremas: *outliers*, *leverage* y distancia de Cook

$$D_i = \frac{e_i^2}{(p+1)\,MSE}\left[\frac{h_{ii}}{(1-h_{ii})^2}\right]$$
""")

code(r"""
inf = OLSInfluence(mco)
n, p1 = int(mco.nobs), int(mco.df_model) + 1
diag_inf = pd.DataFrame({"seqn": d2["seqn"], "Δpeso": d2[OUTCOME_CAUSAL], "qsmk": d2["qsmk"], "wt71": d2["wt71"],
                         "r_estud": inf.resid_studentized_external, "h": inf.hat_matrix_diag, "Cook": inf.cooks_distance[0]})
umb = {"|r| > 3": (diag_inf["r_estud"].abs() > 3).sum(), f"h > 2(p+1)/n = {2*p1/n:.4f}": (diag_inf["h"] > 2 * p1 / n).sum(),
       f"Cook > 4/n = {4/n:.4f}": (diag_inf["Cook"] > 4 / n).sum(), "Cook > 0,5": (diag_inf["Cook"] > 0.5).sum()}
print(pd.Series(umb).to_string())
display(diag_inf.sort_values("Cook", ascending=False).head(8).style.format({"Δpeso": "{:.2f}", "wt71": "{:.1f}", "r_estud": "{:.2f}", "h": "{:.4f}", "Cook": "{:.4f}"}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(9, 4.5))
sc = ax.scatter(diag_inf["h"], diag_inf["r_estud"], c=diag_inf["Cook"], cmap="viridis", s=12 + 2000 * diag_inf["Cook"])
ax.axhline(3, color="gray", ls="--"); ax.axhline(-3, color="gray", ls="--"); ax.axvline(2 * p1 / n, color="orange", ls=":")
plt.colorbar(sc, label="Distancia de Cook"); ax.set_xlabel("Leverage h"); ax.set_ylabel("Residuo estudentizado")
ax.set_title("Mapa de influencia"); plt.tight_layout(); plt.show()

huber = smf.rlm(FORM_MCO, data=d2, M=sm.robust.norms.HuberT()).fit()
comp_rob = pd.DataFrame({"MCO": mco.params, "Huber (RLM)": huber.params}).loc[["qsmk", "age", "smokeintensity", "wt71", "ht"]]
comp_rob["diferencia %"] = 100 * (comp_rob["Huber (RLM)"] - comp_rob["MCO"]) / comp_rob["MCO"].abs()
display(comp_rob.style.format("{:.4f}").format({"diferencia %": "{:+.1f}%"}))
""")

md(r"""
### 2.2–2.3 · Interpretación
""" + "\n" + T("F2_diag"))

md(r"""
## 2.4 Multicolinealidad (VIF)
""")

code(r"""
def vif_tabla(datos, cols):
    X = sm.add_constant(datos[cols].astype(float)).values
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(len(cols))], index=cols)

cols_vif = ["qsmk", "sex", "race", "age", "smokeyrs", "smokeintensity", "wt71", "ht", "school", "education", "income", "exercise", "active", "cholesterol", "alcoholfreq"]
cols_vif2 = [c for c in cols_vif if c not in ("school", "smokeyrs")] + ["edad_inicio"]
vif = pd.concat([vif_tabla(d2, cols_vif).rename("VIF original"), vif_tabla(d2, cols_vif2).rename("VIF tras la resolución")], axis=1)
display(vif.sort_values("VIF original", ascending=False).style.format("{:.2f}", na_rep="—"))
print(f"corr(age, smokeyrs) = {d2[['age','smokeyrs']].corr().iloc[0,1]:.3f}  |  corr(age, edad_inicio) = {d2[['age','edad_inicio']].corr().iloc[0,1]:.3f}"
      f"  |  corr(school, education) = {d2[['school','education']].corr(method='spearman').iloc[0,1]:.3f}")
""")

md(r"""
### 2.4 · Resolución
""" + "\n" + T("F2_vif"))

md(r"""
## 2.5 PCA sin *data leakage*

Regla estricta: la estandarización y la descomposición se ajustan **solo con el conjunto de entrenamiento** (el mismo de la Fase 3) y la matriz de rotación se aplica "a ciegas" sobre test.
""")

code(r"""
Xtr, Xte, ytr, yte = features.split(d)
med_tr = d.loc[Xtr.index, PCA_VARS].median()
P_tr = d.loc[Xtr.index, PCA_VARS].fillna(med_tr)
P_te = d.loc[Xte.index, PCA_VARS].fillna(med_tr)          # imputación con la mediana de TRAIN
pca_pipe = pipeline.fit_pca(P_tr)
pca = pca_pipe.named_steps["pca"]
tabla_pca = pd.DataFrame({"autovalor λ": pca.explained_variance_, "% varianza": 100 * pca.explained_variance_ratio_,
                          "% acumulado": 100 * np.cumsum(pca.explained_variance_ratio_)}, index=[f"PC{i+1}" for i in range(len(PCA_VARS))])
display(tabla_pca.style.format("{:.3f}"))
k_kaiser = int((pca.explained_variance_ > 1).sum())
print(f"Criterio de Kaiser (λ > 1): {k_kaiser} componentes ({tabla_pca['% acumulado'].iloc[k_kaiser-1]:.1f}% de la varianza)")
print(f"Control anti-leakage: medias del escalador = medias de TRAIN → {np.allclose(pca_pipe.named_steps['escalar'].mean_, P_tr.mean())}")

Z_te = pca_pipe.transform(P_te)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
pcs = np.arange(1, len(PCA_VARS) + 1)
axes[0].bar(pcs, pca.explained_variance_, color="#8da0cb"); axes[0].axhline(1, color="red", ls="--", label="Kaiser λ = 1")
ax2 = axes[0].twinx(); ax2.plot(pcs, 100 * np.cumsum(pca.explained_variance_ratio_), "o-", color="#d95f02"); ax2.set_ylabel("% acumulado")
axes[0].set_xlabel("Componente"); axes[0].set_ylabel("Autovalor"); axes[0].set_title("Scree plot"); axes[0].legend(loc="center right")
cargas = pd.DataFrame(pca.components_[:3].T, index=PCA_VARS, columns=["PC1", "PC2", "PC3"])
sns.heatmap(cargas, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[1]); axes[1].set_title("Cargas (loadings)")
for k, col in [(0, "#66c2a5"), (1, "#fc8d62")]:
    m = yte.values == k
    axes[2].scatter(Z_te[m, 0], Z_te[m, 1], s=10, alpha=0.6, color=col, label="Murió 1983-92" if k else "Vivo en 1992")
axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2"); axes[2].set_title("Test proyectado con la rotación de train"); axes[2].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
### 2.5 · Interpretación de las componentes
""" + "\n" + T("F2_pca"))

# =============================================================================== FASE 3
md(r"""
<a id="fase3"></a>
---
# FASE 3 — Torneo de modelos predictivos, calibración y explicabilidad

**Pregunta predictiva:** con la información disponible en 1971 (y si la persona dejó de fumar), ¿cuál es la probabilidad de morir entre 1983 y 1992? Es un problema de **clasificación binaria** con prevalencia ≈ 19,5%. Partición 80/20 **estratificada** y `StratifiedKFold(5)` en train; todas las transformaciones viven dentro de `Pipeline` (sin *leakage*).
""")

code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, precision_recall_fscore_support,
                             roc_curve, precision_recall_curve, f1_score, accuracy_score)

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
print(f"Train: {len(Xtr)} ({ytr.mean():.2%} muertes) | Test: {len(Xte)} ({yte.mean():.2%} muertes) | predictoras: {Xtr.shape[1]}")
modelos = {}
""")

md(r"""
### Familia 1 — Regresión logística: Odds Ratios (statsmodels) y versión regularizada (scikit-learn)
""")

code(r"""
FORM_LOGIT = "death ~ age + sex + race + C(education) + smokeintensity + smokeyrs + wt71 + C(exercise) + C(active) + qsmk + hf + tumor + bronch + C(hbp_cat)"
logit_sm = smf.logit(FORM_LOGIT, data=d.loc[Xtr.index]).fit(disp=0)
or_tab = pd.DataFrame({"β̂": logit_sm.params, "OR": np.exp(logit_sm.params), "IC95 inf": np.exp(logit_sm.conf_int()[0]),
                       "IC95 sup": np.exp(logit_sm.conf_int()[1]), "p-valor": logit_sm.pvalues}).drop(index="Intercept")
print(f"Pseudo R² McFadden = {logit_sm.prsquared:.4f} | LLR p = {logit_sm.llr_pvalue:.2e}")
display(or_tab.style.format({"β̂": "{:.4f}", "OR": "{:.3f}", "IC95 inf": "{:.3f}", "IC95 sup": "{:.3f}", "p-valor": "{:.4f}"}))

gs_lr = GridSearchCV(features.make_pipeline(pipeline.logit_base()), pipeline.LOGIT_GRID, cv=skf, scoring="neg_brier_score").fit(Xtr, ytr)
modelos["Logística regularizada"] = gs_lr.best_estimator_
print("Mejores hiperparámetros (CV, Brier):", gs_lr.best_params_)
""")

md(r"""
### Familia 2 — Gaussian Naive Bayes
""")

code(r"""
modelos["Gaussian Naive Bayes"] = features.make_pipeline(GaussianNB()).fit(Xtr, ytr)
corr_cond = Xtr[ytr == 0][["age", "smokeyrs", "wt71", "ht"]].corr()
print("Correlaciones DENTRO de la clase 'vivo' (el supuesto de NB exige ≈ 0):")
print(corr_cond.round(2).to_string())
""")

md(r"""
### Familia 3 — Árbol CART: sobreajuste y poda por costo-complejidad
""")

code(r"""
arbol_full = features.make_pipeline(DecisionTreeClassifier(random_state=SEED), scale=False).fit(Xtr, ytr)
print(f"Árbol sin restricciones: {arbol_full[-1].get_n_leaves()} hojas | AUC train = {roc_auc_score(ytr, arbol_full.predict_proba(Xtr)[:, 1]):.3f} | "
      f"AUC test = {roc_auc_score(yte, arbol_full.predict_proba(Xte)[:, 1]):.3f} | accuracy train = {accuracy_score(ytr, arbol_full.predict(Xtr)):.3f}, test = {accuracy_score(yte, arbol_full.predict(Xte)):.3f}")

prep_tree = features.make_preprocessor(scale=False).fit(Xtr)
ruta = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(prep_tree.transform(Xtr), ytr)
alphas = np.unique(np.quantile(ruta.ccp_alphas[:-1], np.linspace(0, 1, 40)))
gs_tree = GridSearchCV(features.make_pipeline(DecisionTreeClassifier(random_state=SEED), scale=False),
                       {"modelo__ccp_alpha": alphas}, cv=skf, scoring="roc_auc").fit(Xtr, ytr)
modelos["Árbol CART podado"] = gs_tree.best_estimator_
arbol = gs_tree.best_estimator_[-1]
print(f"α óptimo (CV) = {gs_tree.best_params_['modelo__ccp_alpha']:.5f} → {arbol.get_n_leaves()} hojas, profundidad {arbol.get_depth()}")

fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 2]})
cvr = pd.DataFrame(gs_tree.cv_results_)
axes[0].errorbar(cvr["param_modelo__ccp_alpha"].astype(float), cvr["mean_test_score"], yerr=cvr["std_test_score"], fmt="o-", ms=3)
axes[0].set_xscale("symlog", linthresh=1e-4); axes[0].set_xlabel("ccp_alpha"); axes[0].set_ylabel("AUC (CV)"); axes[0].set_title("Poda: AUC de validación")
plot_tree(arbol, max_depth=2, feature_names=list(gs_tree.best_estimator_[0].get_feature_names_out()), filled=True, rounded=True,
          class_names=["vivo", "murió"], proportion=True, precision=2, fontsize=9, ax=axes[1])
axes[1].set_title("Árbol podado: primeros niveles"); plt.tight_layout(); plt.show()
""")

md(r"""
### Familia 4 — Random Forest: búsqueda de hiperparámetros y error OOB
""")

code(r"""
param_rf = {"modelo__n_estimators": [200, 400, 600], "modelo__max_depth": [3, 5, 8, 12, None],
            "modelo__min_samples_split": [2, 10, 30], "modelo__min_samples_leaf": [1, 5, 10, 20],
            "modelo__max_features": ["sqrt", "log2", 0.3, 0.5]}
rs_rf = RandomizedSearchCV(features.make_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1, oob_score=True), scale=False),
                           param_rf, n_iter=30, cv=skf, scoring="roc_auc", random_state=SEED).fit(Xtr, ytr)
modelos["Random Forest"] = rs_rf.best_estimator_
print("Mejores hiperparámetros:", rs_rf.best_params_)
print(f"AUC (CV 5-fold) = {rs_rf.best_score_:.4f}  |  accuracy OOB = {rs_rf.best_estimator_[-1].oob_score_:.4f}")

best_rf = {k.replace("modelo__", ""): v for k, v in rs_rf.best_params_.items() if k != "modelo__n_estimators"}
Xtr_t = rs_rf.best_estimator_[0].transform(Xtr)
oob = []
rf_ws = RandomForestClassifier(random_state=SEED, n_jobs=-1, oob_score=True, warm_start=True, **best_rf)
for B in [25, 50, 100, 200, 300, 400, 600]:
    rf_ws.set_params(n_estimators=B).fit(Xtr_t, ytr)
    oob.append({"B": B, "AUC OOB": roc_auc_score(ytr, rf_ws.oob_decision_function_[:, 1])})
oob = pd.DataFrame(oob)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(oob["B"], oob["AUC OOB"], "o-", label="AUC OOB"); ax.axhline(rs_rf.best_score_, color="red", ls="--", label="AUC CV 5-fold (mejor)")
ax.set_xlabel("Número de árboles B"); ax.set_ylabel("AUC"); ax.set_title("Convergencia OOB vs. validación cruzada"); ax.legend()
plt.tight_layout(); plt.show()
print(oob.set_index("B").T.round(4).to_string())
""")

md(r"""
### Selección del modelo final por validación cruzada (no por test) y calibración
""")

code(r"""
cv_sel = []
for nombre, m in modelos.items():
    p_cv = cross_val_predict(m, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    cv_sel.append({"Modelo": nombre, "AUC CV": roc_auc_score(ytr, p_cv), "Brier CV": brier_score_loss(ytr, p_cv)})
cv_sel = pd.DataFrame(cv_sel).set_index("Modelo").sort_values("Brier CV")
display(cv_sel.style.format("{:.4f}"))

C_final, L1_final = gs_lr.best_params_["modelo__C"], gs_lr.best_params_["modelo__l1_ratio"]
modelos["Logística calibrada (final)"] = pipeline.final_predictive_model(Xtr, ytr, C=C_final, l1_ratio=L1_final)
modelos["RF calibrado (isotónica)"] = CalibratedClassifierCV(rs_rf.best_estimator_, method="isotonic", cv=5).fit(Xtr, ytr)
modelos["NB calibrado (isotónica)"] = CalibratedClassifierCV(features.make_pipeline(GaussianNB()), method="isotonic", cv=5).fit(Xtr, ytr)
""")

md(r"""
### Evaluación en test
""")

code(r"""
def eval_test(m, thr=0.5):
    p = m.predict_proba(Xte)[:, 1]; yhat = (p >= thr).astype(int)
    pm, rm, fm, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yte, yhat, average="weighted", zero_division=0)
    return {"AUC-ROC": roc_auc_score(yte, p), "PR-AUC": average_precision_score(yte, p), "Brier": brier_score_loss(yte, p),
            "Precision macro": pm, "Recall macro": rm, "F1 macro": fm, "F1 ponderado": fw}

tabla_test = pd.DataFrame({k: eval_test(m) for k, m in modelos.items()}).T
tabla_test.loc["Referencia: predecir la prevalencia"] = {"AUC-ROC": 0.5, "PR-AUC": yte.mean(), "Brier": brier_score_loss(yte, np.full(len(yte), ytr.mean())),
                                                        "Precision macro": np.nan, "Recall macro": np.nan, "F1 macro": np.nan, "F1 ponderado": np.nan}
display(tabla_test.style.format("{:.4f}", na_rep="—"))

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
mostrar = ["Logística calibrada (final)", "Gaussian Naive Bayes", "Árbol CART podado", "Random Forest", "NB calibrado (isotónica)"]
for nombre in mostrar:
    p = modelos[nombre].predict_proba(Xte)[:, 1]
    f, t_, _ = roc_curve(yte, p); axes[0].plot(f, t_, label=f"{nombre} ({roc_auc_score(yte, p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p); axes[1].plot(rc, pr, label=f"{nombre} ({average_precision_score(yte, p):.3f})")
    fr, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile"); axes[2].plot(mp, fr, "o-", label=nombre)
axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set_title("Curvas ROC (AUC)"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[1].axhline(yte.mean(), color="k", ls="--"); axes[1].set_title("Curvas Precision-Recall (PR-AUC)"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[2].plot([0, 1], [0, 1], "k--"); axes[2].set_title("Reliability diagram (test)"); axes[2].set_xlabel("Probabilidad predicha"); axes[2].set_ylabel("Frecuencia observada")
for ax in axes: ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

md(r"""
### Explicabilidad: MDI vs. importancia por permutación
""")

code(r"""
rf_final = rs_rf.best_estimator_
nombres = list(rf_final[0].get_feature_names_out())
mdi = pd.Series(rf_final[-1].feature_importances_, index=nombres)
perm = permutation_importance(rf_final, Xte, yte, n_repeats=20, random_state=SEED, scoring="roc_auc", n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=Xte.columns)
perm_sd = pd.Series(perm.importances_std, index=Xte.columns)
mdi_orig = pd.Series({c: mdi[[n for n in nombres if n == c or n.startswith(c + "_")]].sum() for c in Xte.columns})
imp = pd.DataFrame({"MDI (agregado por variable)": mdi_orig, "Permutación ΔAUC (test)": perm_s, "desvío": perm_sd,
                    "valores únicos": Xtr.nunique()}).sort_values("MDI (agregado por variable)", ascending=False)
display(imp.head(12).style.format({"MDI (agregado por variable)": "{:.4f}", "Permutación ΔAUC (test)": "{:.4f}", "desvío": "{:.4f}", "valores únicos": "{:.0f}"}))

top = imp.head(10).index
fig, ax = plt.subplots(figsize=(11, 4.6)); x = np.arange(len(top)); w = 0.4
ax.bar(x - w/2, imp.loc[top, "MDI (agregado por variable)"] / imp["MDI (agregado por variable)"].sum(), w, label="MDI (normalizada)")
ax.bar(x + w/2, imp.loc[top, "Permutación ΔAUC (test)"].clip(lower=0) / perm_s.clip(lower=0).sum(), w, label="Permutación en test (normalizada)")
ax.set_xticks(x); ax.set_xticklabels(top, rotation=30); ax.legend(); ax.set_title("MDI vs. permutación (top 10 por MDI)")
plt.tight_layout(); plt.show()

rng = np.random.default_rng(SEED)
Xr_tr = Xtr.assign(ruido_continuo=rng.normal(size=len(Xtr)), ruido_binario=rng.integers(0, 2, len(Xtr)))
Xr_te = Xte.assign(ruido_continuo=rng.normal(size=len(Xte)), ruido_binario=rng.integers(0, 2, len(Xte)))
prep_r = features.make_preprocessor(scale=False)
prep_r.transformers.append(("ruido", "passthrough", ["ruido_continuo", "ruido_binario"]))
rf_r = RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=400, **best_rf)
from sklearn.pipeline import Pipeline as _P
pipe_r = _P([("prep", prep_r), ("modelo", rf_r)]).fit(Xr_tr, ytr)
mdi_r = pd.Series(pipe_r[-1].feature_importances_, index=pipe_r[0].get_feature_names_out())
perm_r = permutation_importance(pipe_r, Xr_te, yte, n_repeats=20, random_state=SEED, scoring="roc_auc", n_jobs=-1)
perm_r = pd.Series(perm_r.importances_mean, index=Xr_te.columns)
print("Control de sesgo de cardinalidad (variables de ruido puro):")
print(pd.DataFrame({"MDI": mdi_r[["ruido_continuo", "ruido_binario"]], "Permutación ΔAUC": perm_r[["ruido_continuo", "ruido_binario"]]}).round(4).to_string())
""")

md(r"""
### Fase 3 · Interpretación del torneo
""" + "\n" + T("F3"))

# =============================================================================== FASE 4
md(r"""
<a id="fase4"></a>
---
# FASE 4 — Inferencia causal: DAG, criterio *backdoor*, DoWhy, Double ML y refutaciones

**Pregunta causal:** si un programa de salud pública lograra que una persona fumadora deje de fumar — $do(qsmk = 1)$ —, ¿cuántos kilos ganaría en promedio en 11 años, comparado con seguir fumando? El aumento de peso es el principal efecto adverso que desalienta a dejar de fumar, y dimensionarlo permite planificar apoyo nutricional.

## 4.1 DAG estructural
""" + "\n" + T("F4_dag"))

code(r"""
import networkx as nx
G_full = causal.graph(include_latent=True)
pos = {"sex": (0, 4), "race": (1.2, 4.6), "age": (2.6, 4.9), "education": (4.0, 4.6), "smokeintensity": (5.4, 4.6), "smokeyrs": (3.4, 3.7),
       "exercise": (7.8, 3.3), "active": (8.6, 2.2), "wt71": (-0.6, 2.6), "conciencia_salud": (8.8, 4.8),
       "price82": (-1.0, 0.6), TREATMENT: (1.2, 0.0), OUTCOME_CAUSAL: (7.0, 0.0), "ingesta_metabolismo": (4.1, -1.1),
       COLLIDER: (4.1, -2.4), "peso82_observado": (1.2, -2.4)}
COL = {"Confusor": "#8da0cb", "Tratamiento": "#fc8d62", "Resultado": "#66c2a5", "Instrumento": "#e5c494", "Colisionador": "#e78ac3",
       "Mediador (latente)": "#ffd92f", "Confusor no observado": "#dddddd", "Selección (censura)": "#b3b3b3"}
latentes = {"ingesta_metabolismo", "conciencia_salud"}
fig, ax = plt.subplots(figsize=(15, 8.5))
nx.draw_networkx_nodes(G_full, pos, node_color=[COL[causal.ROLES[n]] for n in G_full], node_size=3300, edgecolors="black", ax=ax)
for n in latentes:
    ax.scatter(*pos[n], s=3300, facecolors="none", edgecolors="black", linestyles="dashed", linewidths=1.5, zorder=3)
nx.draw_networkx_labels(G_full, pos, labels={n: causal.LABELS[n].replace(" (", "\n(").replace(" ", "\n", 1) if len(causal.LABELS[n]) > 12 else causal.LABELS[n] for n in G_full}, font_size=7.5, ax=ax)
lat_e = [e for e in G_full.edges if e[0] in latentes or e[1] in latentes]
nx.draw_networkx_edges(G_full, pos, edgelist=[e for e in G_full.edges if e not in lat_e], arrowsize=14, node_size=3300, width=1.0, alpha=0.75, ax=ax)
nx.draw_networkx_edges(G_full, pos, edgelist=lat_e, arrowsize=14, node_size=3300, width=1.4, style="dashed", ax=ax)
for rol, c in COL.items(): ax.scatter([], [], s=150, color=c, edgecolor="black", label=rol)
ax.legend(loc="lower right", fontsize=8); ax.set_axis_off(); ax.set_title("DAG: dejar de fumar → cambio de peso (líneas punteadas = no observado)", fontsize=13)
plt.tight_layout(); plt.show()
print("DOT (nodos observados, usado por DoWhy):\n", causal.dot())
""")

md(r"""
## 4.2 Identificación: caminos y criterio *backdoor*
""")

code(r"""
G = causal.graph()
caminos = causal.paths_table(G)
print(caminos["tipo"].value_counts().to_string(), "\n")
display(caminos.groupby("tipo").head(4).style.hide(axis="index"))
cands = CONFOUNDERS + [INSTRUMENT, COLLIDER]
minimos = causal.minimal_adjustment_sets(G, cands)
print("Conjuntos de ajuste mínimos válidos:", [sorted(s) for s in minimos])
pruebas = {"Z (9 confusores)": set(CONFOUNDERS), "Z sin wt71": set(CONFOUNDERS) - {"wt71"}, "Z sin smokeintensity": set(CONFOUNDERS) - {"smokeintensity"},
           "Z + colisionador sbp": set(CONFOUNDERS) | {COLLIDER}, "Z + instrumento price82": set(CONFOUNDERS) | {INSTRUMENT}, "∅ (ninguno)": set()}
pd.Series({k: causal.satisfies_backdoor(G, v) for k, v in pruebas.items()}, name="¿Cumple el criterio backdoor?").to_frame()
""")

md(r"""
### 4.2 · Justificación formal y preguntas trampa
""" + "\n" + T("F4_backdoor"))

md(r"""
## 4.3 Flujo DoWhy en 4 pasos + Double ML
""")

code(r"""
res_c = pipeline.run_causal(dc, d, n_sim=100)
print("PASO 2 — Estimando identificado:\n", res_c["estimando"])
""")

code(r"""
tab_ate = pd.DataFrame({
    "ATE (kg)": [res_c["naive"]["ate"], res_c["ols"]["ate"], res_c["ipw"]["ate"], res_c["dml"]["ate"]],
    "IC95 inf": [res_c["naive"]["ic"][0], res_c["ols"]["ic"][0], res_c["ipw"]["ic"][0], res_c["dml"]["ic"][0]],
    "IC95 sup": [res_c["naive"]["ic"][1], res_c["ols"]["ic"][1], res_c["ipw"]["ic"][1], res_c["dml"]["ic"][1]],
}, index=["Ingenuo: Y ~ T", "Backdoor: regresión lineal (DoWhy)", "Backdoor: IPW (bootstrap 300)", "Double ML (LinearDML, RF)"])
tab_ate["Sesgo vs. backdoor"] = tab_ate["ATE (kg)"] - res_c["ols"]["ate"]
display(tab_ate.style.format("{:.3f}").format({"Sesgo vs. backdoor": "{:+.3f}"}))
flex = smf.ols(f"{OUTCOME_CAUSAL} ~ qsmk + sex + race + age + I(age**2) + C(education) + smokeintensity + I(smokeintensity**2)"
               " + smokeyrs + I(smokeyrs**2) + C(exercise) + C(active) + wt71 + I(wt71**2)", data=dc).fit()
con_sbp = causal.ols_adjusted(dc.dropna(subset=[COLLIDER]), f" + {COLLIDER}")
sin_sbp = causal.ols_adjusted(dc.dropna(subset=[COLLIDER]))
sens = pd.Series({
    "MCO flexible (términos cuadráticos + categóricas)": flex.params["qsmk"],
    "IPW versión DoWhy (ips_weight)": res_c["ipw"]["ate_dowhy"],
    "MCO ponderado por censura (IPCW, supuesto MAR)": res_c["ipcw"]["ate"],
    "Misma muestra con sbp: sin ajustar por sbp": sin_sbp.params["qsmk"],
    "Misma muestra con sbp: AJUSTANDO por el colisionador sbp": con_sbp.params["qsmk"],
}, name="ATE (kg)")
display(sens.to_frame().style.format("{:.3f}"))
""")

code(r"""
# Variable instrumental (sensibilidad a confusión no observada): 2SLS manual con price82
div = dc.dropna(subset=[INSTRUMENT])
fs = smf.ols(f"qsmk ~ {INSTRUMENT}", data=div).fit()
rf_iv = smf.ols(f"{OUTCOME_CAUSAL} ~ {INSTRUMENT}", data=div).fit()
print(f"Primera etapa: qsmk ~ price82 -> coef = {fs.params[INSTRUMENT]:.3f}, F = {fs.fvalue:.2f} (regla práctica: F > 10)")
print(f"Estimador de Wald (IV) = {rf_iv.params[INSTRUMENT] / fs.params[INSTRUMENT]:.2f} kg  ← instrumento débil: no informativo")
""")

code(r"""
refs = pd.DataFrame(res_c["refutaciones"]).T
refs["efecto original"] = res_c["ols"]["ate"]
display(refs[["efecto original", "efecto_nuevo", "p_valor", "condicion_ok", "semaforo"]].style.format({"efecto original": "{:.3f}", "efecto_nuevo": "{:.3f}", "p_valor": "{:.3f}"}))
""")

code(r"""
grid = pd.DataFrame(res_c["cate_grid"])
dml_c = res_c["dml"]
print(f"CATE lineal: τ(x) = {dml_c['intercepto']:.3f} + {dml_c['coef']['smokeintensity']:.4f}·cigarrillos/día + {dml_c['coef']['age']:.4f}·edad")
for k, v in dml_c["coef_ic"].items():
    print(f"  IC95 del coeficiente de {k}: [{v[0]:.4f}; {v[1]:.4f}]")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for edad, col in zip([25, 45, 65], ["#1b9e77", "#7570b3", "#d95f02"]):
    g = grid[np.isclose(grid["age"], grid["age"].unique()[np.argmin(np.abs(grid["age"].unique() - edad))])]
    axes[0].plot(g["smokeintensity"], g["cate"], color=col, lw=2, label=f"edad ≈ {edad}")
    axes[0].fill_between(g["smokeintensity"], g["ic_inf"], g["ic_sup"], color=col, alpha=0.12)
axes[0].axhline(res_c["ols"]["ate"], color="red", ls="--", label=f"ATE backdoor = {res_c['ols']['ate']:.2f}"); axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xlabel("Cigarrillos por día en 1971"); axes[0].set_ylabel("CATE: kg adicionales por dejar de fumar"); axes[0].set_title("Heterogeneidad del efecto (LinearDML)"); axes[0].legend()
piv = grid.pivot_table(index="age", columns="smokeintensity", values="cate")
sns.heatmap(piv.iloc[::-1], cmap="viridis", ax=axes[1], cbar_kws={"label": "CATE (kg)"}, xticklabels=8, yticklabels=5)
axes[1].set_title("Mapa de CATE: edad × cigarrillos/día"); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.3 · Interpretación de la estimación, DML y refutaciones
""" + "\n" + T("F4_estim"))

# =============================================================================== ENSAYO
md(r"""
<a id="ensayo"></a>
---
# 5. Ensayo crítico: el choque entre capacidad predictiva y palanca causal
""")

code(r"""
d_death = d.copy()
naive_death = d_death.groupby("qsmk")["death"].mean()
m_death_naive = smf.ols("death ~ qsmk", d_death).fit()
m_death_adj = smf.ols("death ~ qsmk + " + causal.FORMULA_Z, d_death).fit()
print(f"Mortalidad observada: dejó de fumar {naive_death[1]:.1%} vs. siguió fumando {naive_death[0]:.1%}  → asociación cruda {100*m_death_naive.params['qsmk']:+.1f} p.p.")
print(f"Efecto ajustado por backdoor (diferencia de riesgos): {100*m_death_adj.params['qsmk']:+.1f} p.p. "
      f"(IC95 {100*m_death_adj.conf_int().loc['qsmk',0]:+.1f}; {100*m_death_adj.conf_int().loc['qsmk',1]:+.1f})")
print(f"Variable con mayor importancia por permutación en el RF de mortalidad: {perm_s.idxmax()} (ΔAUC = {perm_s.max():.3f})")
""")

md(T("ENSAYO"))

# =============================================================================== CONCLUSIONES
md(r"""
<a id="conclusiones"></a>
---
# 6. Conclusiones, recomendaciones y limitaciones
""" + "\n" + T("CONC"))

code(r"""
# Artefactos para la web app (mismos objetos y semillas que este notebook)
info = {"modelo": "Regresión logística regularizada y calibrada (sigmoide)", "C": C_final, "l1_ratio": L1_final,
        "metricas_test": pipeline.predictive_metrics(modelos["Logística calibrada (final)"], Xte, yte),
        "features": list(Xtr.columns), "medianas_train": Xtr.median(numeric_only=True).to_dict(),
        "modas_train": Xtr.mode().iloc[0].astype(str).to_dict()}
pipeline.save_artifacts(modelos["Logística calibrada (final)"], info, pca_pipe, res_c, d)
print("Artefactos guardados en", ROOT / "models")
""")

md(r"""
<a id="referencias"></a>
---
# 7. Referencias
""" + "\n" + T("REFS"))

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "TP_Integrador_Final.ipynb")
print("ok")

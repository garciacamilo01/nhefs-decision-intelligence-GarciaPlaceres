"""Web app de decisión: Diagnóstico (PCA), Predictivo (What-Is) y Simulación causal de políticas (What-If).

Ejecutar localmente:  streamlit run app/streamlit_app.py
La app solo lee artefactos de /models (generados por el notebook o por scripts/build_artifacts.py).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

st.set_page_config(page_title="NHEFS · Decision Intelligence", page_icon="🚭", layout="wide")


# ----------------------------------------------------------------------------- carga
@st.cache_resource
def cargar():
    datos = pd.read_csv(MODELS / "datos_app.csv")
    pca = joblib.load(MODELS / "pca.joblib")
    modelo = joblib.load(MODELS / "modelo_predictivo.joblib")
    pred = json.loads((MODELS / "predictivo.json").read_text(encoding="utf-8"))
    causal = json.loads((MODELS / "causal.json").read_text(encoding="utf-8"))
    return datos, pca, modelo, pred, causal


datos, pca, modelo, PRED, CAUSAL = cargar()
PCA_VARS = list(pca.feature_names_in_)
FEATURES = PRED["features"]

EDU = {1: "≤ 8.º grado", 2: "Secundario incompleto", 3: "Secundario completo", 4: "Universitario incompleto", 5: "Universitario completo"}
EXE = {0: "Mucho", 1: "Moderado", 2: "Poco o nada"}
ACT = {0: "Muy activo", 1: "Moderadamente activo", 2: "Inactivo"}
COLOR_ROL = {"Confusor": "#8da0cb", "Tratamiento": "#fc8d62", "Resultado": "#66c2a5", "Instrumento": "#e5c494",
             "Colisionador": "#e78ac3", "Mediador (latente)": "#ffd92f", "Confusor no observado": "#dddddd",
             "Selección (censura)": "#b3b3b3"}
SEMAFORO = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}

st.sidebar.title("🚭 NHEFS · Decision Intelligence")
modulo = st.sidebar.radio("Módulo", ["1 · Diagnóstico y exploración", "2 · Predictivo (What-Is)",
                                     "3 · Simulación causal (What-If)", "Acerca de"])
st.sidebar.caption("Datos: NHANES I Epidemiologic Follow-up Study (1971–1992), N = 1.629. "
                   "Proyecto integrador EAyMR 2026 · UCA · Camilo Garcia Placeres.")


# ============================================================================ MÓDULO 1
if modulo.startswith("1"):
    st.title("Diagnóstico y exploración")
    c1, c2, c3, c4 = st.columns(4)
    sexo = c1.multiselect("Sexo", ["Hombre", "Mujer"], default=["Hombre", "Mujer"])
    trat = c2.multiselect("Dejó de fumar (qsmk)", ["No", "Sí"], default=["No", "Sí"])
    edad = c3.slider("Edad en 1971", 25, 74, (25, 74))
    edu = c4.multiselect("Educación", list(EDU.values()), default=list(EDU.values()))

    f = datos[datos["sex"].map({0: "Hombre", 1: "Mujer"}).isin(sexo) & datos["qsmk"].map({0: "No", 1: "Sí"}).isin(trat)
              & datos["age"].between(*edad) & datos["education"].map(EDU).isin(edu)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Participantes", f"{len(f):,}".replace(",", "."))
    m2.metric("Dejaron de fumar", f"{f['qsmk'].mean():.1%}" if len(f) else "—")
    m3.metric("Murieron 1983–92", f"{f['death'].mean():.1%}" if len(f) else "—")
    m4.metric("Cambio de peso medio", f"{f['wt82_71'].mean():+.2f} kg" if f["wt82_71"].notna().any() else "—")

    if len(f) < 5:
        st.warning("Muy pocos participantes con estos filtros.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        var = st.selectbox("Variable a inspeccionar", ["wt82_71", "age", "smokeintensity", "smokeyrs", "wt71", "ht", "cholesterol", "income"])
        fig = px.histogram(f, x=var, color=f["qsmk"].map({0: "Siguió fumando", 1: "Dejó de fumar"}), marginal="box",
                           barmode="overlay", opacity=0.6, labels={"color": ""}, title=f"Distribución de {var}")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        color_por = st.selectbox("Colorear el plano PCA por", ["death", "qsmk", "sex"])
        P = f[PCA_VARS].fillna(datos[PCA_VARS].median())
        Z = pca.transform(P)
        evr = pca.named_steps["pca"].explained_variance_ratio_
        g = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "grupo": f[color_por].astype(str).values, "seqn": f["seqn"].values})
        fig = px.scatter(g, x="PC1", y="PC2", color="grupo", hover_data=["seqn"], opacity=0.6,
                         title=f"Plano PC1 ({evr[0]:.0%}) × PC2 ({evr[1]:.0%})")
        caso = st.number_input("Resaltar participante (seqn)", value=int(f["seqn"].iloc[0]), step=1)
        if caso in set(g["seqn"]):
            r = g[g["seqn"] == caso].iloc[0]
            fig.add_trace(go.Scatter(x=[r.PC1], y=[r.PC2], mode="markers", marker=dict(size=18, symbol="star", color="black"), name=f"seqn {caso}"))
        st.plotly_chart(fig, width="stretch")
    cargas = pd.DataFrame(pca.named_steps["pca"].components_[:2].T, index=PCA_VARS, columns=["PC1", "PC2"])
    st.caption("PC1 ≈ envejecimiento y exposición acumulada al tabaco (edad, años fumando, menor escolaridad); "
               "PC2 ≈ tamaño corporal (altura, peso). Cargas:")
    st.dataframe(cargas.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")


# ============================================================================ MÓDULO 2
elif modulo.startswith("2"):
    st.title("Predictivo (What-Is): riesgo de muerte 1983–1992")
    met = PRED["metricas_test"]
    st.caption(f"Modelo: {PRED['modelo']} · Test: AUC {met['auc_roc']:.3f} · PR-AUC {met['pr_auc']:.3f} · "
               f"Brier {met['brier']:.3f} (prevalencia {met['prevalencia_test']:.1%}). "
               "Es un modelo **predictivo**: no indica qué pasaría al cambiar una variable (ver Módulo 3).")

    with st.form("caso"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Edad", 25, 74, 45)
        sex = c1.radio("Sexo", [0, 1], format_func=lambda x: "Hombre" if x == 0 else "Mujer", horizontal=True)
        race = c1.radio("Raza", [0, 1], format_func=lambda x: "Blanca" if x == 0 else "Negra u otra", horizontal=True)
        education = c1.selectbox("Educación", list(EDU), index=2, format_func=EDU.get)
        smokeintensity = c2.slider("Cigarrillos por día", 1, 80, 20)
        smokeyrs = c2.slider("Años fumando", 1, 64, min(25, age - 10))
        qsmk = c2.radio("¿Dejó de fumar?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", horizontal=True)
        exercise = c2.selectbox("Ejercicio recreativo", list(EXE), index=1, format_func=EXE.get)
        active = c2.selectbox("Actividad diaria", list(ACT), index=1, format_func=ACT.get)
        wt71 = c3.number_input("Peso (kg)", 35.0, 170.0, 70.0)
        ht = c3.number_input("Altura (cm)", 140.0, 200.0, 168.0)
        cholesterol = c3.number_input("Colesterol (mg/100 ml)", 80, 420, 220)
        income = c3.slider("Ingreso familiar (código 11–22)", 11, 22, 19)
        hbp = c3.selectbox("Hipertensión diagnosticada", ["no", "si", "no_informado"])
        enviar = st.form_submit_button("Calcular riesgo")

    base = {c: PRED["medianas_train"].get(c, np.nan) for c in FEATURES}
    for c, v in PRED["modas_train"].items():
        if c in base and (pd.isna(base[c]) or c not in PRED["medianas_train"]):
            base[c] = v
    caso = {**base, "age": age, "sex": sex, "race": race, "education": education, "smokeintensity": smokeintensity,
            "smokeyrs": smokeyrs, "qsmk": qsmk, "exercise": exercise, "active": active, "wt71": wt71, "ht": ht,
            "cholesterol": cholesterol, "income": income, "hbp_cat": hbp, "diabetes_cat": PRED["modas_train"].get("diabetes_cat", "no"),
            "marital_cat": PRED["modas_train"].get("marital_cat", "casado"), "alcoholfreq": PRED["medianas_train"].get("alcoholfreq", 2)}
    X = pd.DataFrame([caso])[FEATURES]
    for c in ["sex", "race", "education", "exercise", "active", "qsmk"]:
        X[c] = X[c].astype(int)
    p = float(modelo.predict_proba(X)[0, 1])
    p_folds = np.array([cc.predict_proba(X)[0, 1] for cc in modelo.calibrated_classifiers_])

    k1, k2 = st.columns([1, 2])
    with k1:
        st.metric("Probabilidad calibrada de muerte (1983–92)", f"{p:.1%}")
        st.caption(f"Rango entre los {len(p_folds)} modelos calibrados de la validación cruzada: "
                   f"{p_folds.min():.1%} – {p_folds.max():.1%}. Prevalencia de la cohorte: 19,5%.")
        nivel = "alto" if p >= 0.35 else "moderado" if p >= 0.15 else "bajo"
        st.info(f"Riesgo **{nivel}** respecto de la cohorte.")
    with k2:
        filas = []
        for c, etiqueta in [("age", "Edad"), ("smokeyrs", "Años fumando"), ("smokeintensity", "Cigarrillos/día"), ("sex", "Sexo"),
                            ("education", "Educación"), ("income", "Ingreso"), ("hbp_cat", "Hipertensión"), ("qsmk", "Dejó de fumar"),
                            ("wt71", "Peso"), ("cholesterol", "Colesterol"), ("active", "Actividad diaria"), ("exercise", "Ejercicio")]:
            Xr = X.copy()
            if c == "hbp_cat":
                Xr[c] = PRED["modas_train"][c]
            elif c in ["sex", "race", "education", "exercise", "active", "qsmk"]:
                Xr[c] = int(float(PRED["modas_train"][c]))
            else:
                Xr[c] = float(PRED["medianas_train"][c])
            filas.append({"variable": etiqueta, "contribución (p.p.)": 100 * (p - modelo.predict_proba(Xr)[0, 1])})
        expl = pd.DataFrame(filas).sort_values("contribución (p.p.)")
        fig = px.bar(expl, x="contribución (p.p.)", y="variable", orientation="h", color="contribución (p.p.)",
                     color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                     title="Explicación local: cuánto cambia el riesgo si la variable tomara el valor típico de la cohorte")
        st.plotly_chart(fig, width="stretch")
    st.caption("La explicación es asociativa: muestra cuánto pesa cada dato en esta predicción, no el efecto de intervenirlo.")


# ============================================================================ MÓDULO 3
elif modulo.startswith("3"):
    st.title("Simulación causal de políticas (What-If): do(dejar de fumar)")
    st.markdown("**Pregunta:** si un programa logra que una persona fumadora deje de fumar, ¿cuántos kg gana en 11 años "
                "respecto de seguir fumando? Identificación por criterio *backdoor* con 9 confusores medidos en 1971.")

    refs = CAUSAL["refutaciones"]
    st.subheader("Semáforo de confianza causal (refutaciones DoWhy)")
    cols = st.columns(len(refs))
    for col, (nombre, r) in zip(cols, refs.items()):
        col.markdown(f"### {SEMAFORO[r['semaforo']]} {nombre}")
        col.caption(f"Efecto re-estimado: {r['efecto_nuevo']:.3f} kg · p = {r['p_valor']:.2f}")
    if all(r["semaforo"] == "verde" for r in refs.values()):
        st.success("Las tres refutaciones pasan. Advertencia: no descartan confusores **no observados** (p. ej., conciencia de salud).")
    else:
        st.error("Al menos una refutación falla: no use este efecto para decidir sin revisar el modelo.")

    izq, der = st.columns([1.1, 1])
    with izq:
        st.subheader("DAG causal")
        roles, labels = CAUSAL["dag"]["roles"], CAUSAL["dag"]["labels"]
        latentes = {"ingesta_metabolismo", "conciencia_salud"}
        nodos = {n for e in CAUSAL["dag"]["edges"] for n in e}
        dot = ["digraph { rankdir=LR; ranksep=0.35; nodesep=0.12; node [style=filled, fontname=Helvetica, fontsize=16];"]
        for n in nodos:
            estilo = "dashed,filled" if n in latentes else "filled"
            dot.append(f'"{n}" [label="{labels.get(n, n)}", fillcolor="{COLOR_ROL.get(roles.get(n), "#ffffff")}", style="{estilo}"];')
        for a, b in CAUSAL["dag"]["edges"]:
            estilo = "dashed" if a in latentes or b in latentes else "solid"
            dot.append(f'"{a}" -> "{b}" [style={estilo}];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), width="stretch")
        st.caption("Líneas punteadas: variables no observadas. Ajustar por el mediador o por el colisionador (presión 1982) sesgaría el efecto.")
    with der:
        st.subheader("Efecto causal promedio (ATE)")
        tab = pd.DataFrame({
            "Estimador": ["Ingenuo (sin ajuste)", "Backdoor: regresión", "Backdoor: IPW", "Double ML"],
            "ATE (kg)": [CAUSAL["naive"]["ate"], CAUSAL["ols"]["ate"], CAUSAL["ipw"]["ate"], CAUSAL["dml"]["ate"]],
            "IC 95%": [f"[{a:.2f}; {b:.2f}]" for a, b in [CAUSAL["naive"]["ic"], CAUSAL["ols"]["ic"], CAUSAL["ipw"]["ic"], CAUSAL["dml"]["ic"]]],
        })
        st.dataframe(tab.style.format({"ATE (kg)": "{:.2f}"}), hide_index=True, width="stretch")
        sesgo = CAUSAL["naive"]["ate"] - CAUSAL["ols"]["ate"]
        st.caption(f"El estimador ingenuo subestima el efecto en {abs(sesgo):.2f} kg: quienes dejaron de fumar eran mayores, y la edad reduce el aumento de peso.")

    st.subheader("Efecto según el perfil (CATE, LinearDML)")
    grid = pd.DataFrame(CAUSAL["cate_grid"])
    p1, p2 = st.columns([1, 2])
    with p1:
        cig = st.slider("Cigarrillos por día del perfil", float(grid["smokeintensity"].min()), float(grid["smokeintensity"].max()), 20.0)
        edad_p = st.slider("Edad del perfil", float(grid["age"].min()), float(grid["age"].max()), 45.0)
        n_prog = st.number_input("Personas que el programa lograría que dejen de fumar", 10, 1_000_000, 1000, step=100)
        ages = grid["age"].unique()
        a_sel = ages[np.argmin(np.abs(ages - edad_p))]
        g = grid[grid["age"] == a_sel]
        fila = g.iloc[np.argmin(np.abs(g["smokeintensity"].values - cig))]
        st.metric("Aumento de peso causado por dejar de fumar (CATE)", f"{fila['cate']:+.2f} kg",
                  help="Diferencia esperada en el cambio de peso a 11 años entre dejar y seguir fumando, para este perfil.")
        st.caption(f"IC 95%: [{fila['ic_inf']:.2f}; {fila['ic_sup']:.2f}] kg")
        st.metric("Kg totales esperados en el grupo intervenido", f"{fila['cate'] * n_prog:,.0f} kg".replace(",", "."))
        st.caption(f"≈ {fila['cate']:.1f} kg por persona → planificar apoyo nutricional para {n_prog:,} personas.".replace(",", "."))
    with p2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["smokeintensity"], y=g["ic_sup"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=g["smokeintensity"], y=g["ic_inf"], fill="tonexty", line=dict(width=0), name="IC 95%",
                                 fillcolor="rgba(31,119,180,0.2)"))
        fig.add_trace(go.Scatter(x=g["smokeintensity"], y=g["cate"], name=f"CATE (edad {a_sel:.0f})", line=dict(color="#1f77b4", width=3)))
        fig.add_hline(y=CAUSAL["ols"]["ate"], line_dash="dash", line_color="red", annotation_text="ATE")
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.add_vline(x=cig, line_dash="dot")
        fig.update_layout(title="CATE vs. cigarrillos por día", xaxis_title="Cigarrillos por día (1971)", yaxis_title="kg adicionales")
        st.plotly_chart(fig, width="stretch")
    coef = CAUSAL["dml"]["coef_ic"]
    st.info("Heterogeneidad: los IC de los coeficientes del CATE incluyen al 0 "
            f"(cigarrillos/día: [{coef['smokeintensity'][0]:.3f}; {coef['smokeintensity'][1]:.3f}], "
            f"edad: [{coef['age'][0]:.3f}; {coef['age'][1]:.3f}]). No hay evidencia suficiente para diferenciar la política por perfil.")


# ============================================================================ ACERCA DE
else:
    st.title("Acerca de este proyecto")
    st.markdown("""
**Trabajo Práctico Integrador — EAyMR 2026 (UCA).** Autor: Camilo Garcia Placeres (asume los 4 roles Scrum).

- **Módulo 1:** exploración filtrable y proyección de casos sobre las dos primeras componentes principales (PCA ajustado solo en train).
- **Módulo 2:** probabilidad calibrada de muerte 1983–1992 con regresión logística regularizada y calibrada, y explicación local por perturbación.
- **Módulo 3:** efecto causal de dejar de fumar sobre el peso (DoWhy + EconML), con DAG, ATE, CATE y semáforo de refutaciones.

**Uso de IA generativa:** el código y los textos se desarrollaron con asistencia de Claude (Anthropic). Ver la Entrada 5 de `BITACORA.md`.

**Datos:** NHEFS (CDC/NCHS), versión de Hernán & Robins, *Causal Inference: What If* (2020).
""")

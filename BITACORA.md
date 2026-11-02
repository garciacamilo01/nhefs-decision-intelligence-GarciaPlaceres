# Bitácora de investigación

**Proyecto:** NHEFS · Decision Intelligence Capstone (TP Integrador EAyMR 2026, UCA)
**Equipo:** proyecto individual. Camilo Garcia Placeres asume los cuatro roles: Project Lead (PL), Backend / ML-Causal (BE), Frontend (FE) y QA.
**Semilla:** 42 · **Notebook:** `notebooks/TP_Integrador_Final.ipynb`

> Cada entrada indica qué rol toma la decisión y qué rol la revisa. Como hay un solo integrante, la revisión cruzada se implementa con controles automáticos: las pruebas de `tests/` y el quality gate de CI deben pasar antes de dar por cerrada una decisión.

---

## Entrada 1 · Encuadre estratégico

**Fecha:** semana 1 · **Decide:** PL · **Revisa:** BE

**Contexto.** Una agencia de salud pública diseña un programa de cesación tabáquica y enfrenta dos obstáculos:

1. El miedo a engordar frena a muchos fumadores. Hace falta saber cuánto peso **causa** dejar de fumar.
2. El presupuesto obliga a priorizar a quienes tienen mayor riesgo de muerte.

**Decisiones.**

- **Dataset: NHEFS.** Se eligió porque cumple los requisitos de la consigna:
  - tiene más de 1.000 filas;
  - mezcla variables numéricas y categóricas;
  - tiene faltantes reales: 63 participantes censurados y un bloque de 791 respuestas codificadas como 2 = MISSING;
  - permite plantear una pregunta causal con DAG defendible.

  Además es el caso de referencia de Hernán y Robins, lo que da un patrón externo con el cual contrastar los resultados.
- **Dos variables objetivo distintas.**
  - Predictiva: `death` (1983–92), para priorizar.
  - Causal: `wt82_71`, para dimensionar el apoyo nutricional.

  No se usa el modelo predictivo para responder la pregunta causal; el Ensayo del notebook explica por qué.
- **Hipótesis propias.**
  - H-a: autoselección.
  - H-b: el efecto metabólico crece con la intensidad.
  - H-c: *sick quitter*.
- **Restricciones asumidas.**
  - Los datos son observacionales.
  - El mediador principal (la dieta) no está medido.
  - La validez externa se limita a EE. UU. en los años 70.

**Firma:** Camilo Garcia Placeres, como PL. Revisado como BE.

---

## Entrada 2 · Matriz de decisiones estadísticas

**Decide:** BE · **Revisa:** QA

| Situación | Supuesto verificado | Evidencia | Decisión |
|---|---|---|---|
| Comparar el peso entre dejó y siguió fumando (H1) | Igualdad de varianzas | Grupos de tamaño muy distinto (403 vs. 1.163) | **Welch t**, con Cohen d y potencia. Resultado: +2,54 kg, p = 2,6·10⁻⁷ |
| Comparar cigarrillos/día (H2b) | Normalidad | Asimetría fuerte (conteos) | **Mann-Whitney** como prueba principal y Welch como complemento |
| Comparar el peso entre 5 niveles educativos | Homocedasticidad | Levene p = 0,014 | **ANOVA de Welch** (F = 4,21; p = 0,002) y Kruskal-Wallis como robustez |
| Comparaciones *post hoc* | Varianzas iguales (lo asume Tukey) | No se cumple | Se reportan Tukey y **Bonferroni sobre pares Welch**. Solo 1 vs. 3 y 1 vs. 4 sobreviven a Bonferroni |
| Regresión OLS del cambio de peso | Homocedasticidad | Breusch-Pagan = 81,5 | **Errores HC3**. El EE de qsmk pasa de 0,45 a 0,49 |
| | Normalidad de residuos | Jarque-Bera = 1.104 | Con n = 1.566 la inferencia es asintótica; se agrega **Huber RLM** como control (qsmk 2,81) |
| | Independencia | Durbin-Watson = 2,01 | Se cumple |
| Multicolinealidad | VIF < 5 | school 7,7; education 7,5; age y smokeyrs 4,7 | Se quita school y se reemplaza el par age/smokeyrs por `edad_inicio`. Todos los VIF quedan < 2,4 |
| Reducción de dimensión | Sin fuga de datos | — | PCA ajustado **solo con train**. Kaiser retiene 3 componentes (63,7%) |
| Selección del modelo predictivo | Probabilidades confiables | Prevalencia 19,5% | Selección por **Brier en CV estratificada de 5 pliegues**, con AUC y PR-AUC como métricas secundarias |
| Efecto causal | Intercambiabilidad condicional | DAG + criterio backdoor (d-separación) | Ajuste por los 9 confusores, que forman el único conjunto mínimo válido. Tres estimadores: OLS, IPW y DML |
| Censura del resultado (3,9%) | MCAR | Test LR p = 0,0002 | **No es MCAR.** Se agrega IPCW como análisis de sensibilidad (3,35 kg, sin cambios) |

**Firma:** Camilo Garcia Placeres, como BE. Revisado como QA: las pruebas en `tests/test_causal.py` y `tests/test_leakage.py` cubren las decisiones de esta matriz.

---

## Entrada 3 · Autopsia de datos y experimentos fallidos

**Decide:** BE · **Revisa:** QA

### Hallazgo de datos: el faltante "oculto" es por diseño

En 7 variables de salud aparece el código `2 = MISSING`. El caso más grande son 791 filas que coinciden **exactamente** con `seqn ≥ 20751`, es decir, con un cambio de cuestionario a mitad del estudio.

- Imputar esos valores habría inventado diagnósticos.
- Decisión: se mantienen como categoría `no_informado`.

### Experimento fallido 1: árbol CART sin podar

**Qué se probó.** Un árbol de decisión sin restricciones.

**Qué pasó.** Creció hasta 180 hojas, con AUC 1,00 en train y 0,62 en test: memorizó la muestra.

**Qué se aprendió.** Se podó por costo-complejidad con α = 0,0031 elegido por CV. El árbol podado tiene 7 hojas y su primer corte es edad ≤ 54,5. Aun así queda por debajo de la logística (AUC CV 0,776 frente a 0,838).

### Experimento fallido 2: Naive Bayes como clasificador de riesgo

**Qué se probó.** Naive Bayes como candidato a modelo de riesgo.

**Qué pasó.** El AUC era aceptable (0,756), pero el Brier en test fue 0,230, **peor que predecir siempre la prevalencia** (0,158). La independencia condicional no se cumple: edad, años fumando e hipertensión están correlacionadas. El modelo cuenta la misma evidencia varias veces y produce probabilidades extremas.

**Qué se aprendió.** La calibración isotónica bajó el Brier a 0,139, pero no alcanza a la logística. Un buen AUC no garantiza probabilidades utilizables para decidir.

### Experimento fallido 3: logit de statsmodels con IC = NaN

**Qué se probó.** Incluir `diabetes_cat` y `hbp_cat` juntos en la logística de statsmodels.

**Qué pasó.** Los intervalos de confianza salieron NaN. Las dos dummies `no_informado` son idénticas, porque cubren las mismas 791 filas, y la matriz X'X es singular.

**Qué se aprendió.** Se quitó `diabetes_cat` de la fórmula. En el pipeline de sklearn no aparece el problema porque la regularización lo absorbe, y eso lo hace más peligroso.

### Experimento fallido 4: precio del tabaco como instrumento

**Qué se probó.** Usar `price82` como variable instrumental, como alternativa al ajuste backdoor.

**Qué pasó.** La primera etapa da F = 0,38, muy lejos del umbral habitual de 10. El estimador de Wald resulta en −26,9 kg, un valor absurdo.

**Qué se aprendió.** El instrumento es débil y se descarta. Queda en el DAG solo para mostrar que no hace falta ajustar por él.

**Firma:** Camilo Garcia Placeres, como BE. Revisado como QA: `test_backdoor_*`, `test_categoria_desconocida_no_rompe_el_modelo` y `test_metricas_predictivas_minimas`.

---

## Entrada 4 · Debate causal

**Decide:** BE · **Revisa:** PL

**¿Por qué el efecto ingenuo (2,54 kg) es menor que el ajustado (3,35 kg)?**

Quienes dejaron de fumar eran 3,4 años mayores y fumaban 2,6 cigarrillos menos por día. Las dos características se asocian con un menor aumento de peso, así que la comparación cruda está sesgada hacia abajo. El sesgo de confusión es de unos −0,8 kg.

**Decisiones sobre el DAG.**

- **No ajustar por `sbp` (presión 1982).** Es un colisionador, porque la causan T e Y, y además es un descendiente del tratamiento. En la submuestra con presión medida, ajustar por `sbp` desplaza el efecto de 3,41 a 3,33 kg. El cambio es chico en este caso, pero invalida el criterio backdoor (lo verifica `test_colisionador_y_descendientes_invalidan_el_ajuste`).
- **El mediador (dieta/metabolismo) no está medido.** No es un problema para el ATE total, que es lo que necesita el programa. Sí impide descomponer el efecto en directo e indirecto.
- **Confusor no observado U (conciencia de salud).** Es el supuesto más débil. Las refutaciones de DoWhy no pueden detectarlo: el test de causa común aleatoria agrega una variable *independiente*, no una correlacionada con T e Y. Se declara como limitación.

**Robustez.** Seis estimaciones válidas caen entre 3,05 y 3,62 kg:

| Estimador | ATE (kg) |
|---|---|
| OLS | 3,35 |
| IPW | 3,28 |
| IPW (DoWhy) | 3,05 |
| OLS flexible | 3,46 |
| IPCW | 3,35 |
| DML | 3,62 |

Las tres refutaciones pasan en verde:

| Refutación | Efecto re-estimado | p |
|---|---|---|
| Causa común aleatoria | 3,350 | 0,90 |
| Placebo | −0,03 | 0,88 |
| Subconjunto 80% | 3,334 | 0,98 |

**Heterogeneidad (CATE).** LinearDML da τ(x) = 3,21 + 0,044·cig − 0,011·edad. Los IC de ambos coeficientes incluyen al 0, así que **no hay evidencia para diferenciar la política por perfil**. La hipótesis H-b no se confirma con estos datos.

**Ensayo predictivo vs. causal.** En el modelo de mortalidad, `qsmk` aparece asociado a mayor muerte: +5,8 p.p. crudo. Al ajustar, la diferencia baja a +0,5 p.p. con un IC que incluye al 0. Usar el modelo predictivo como palanca ("dejar de fumar mata") sería un error de *sick quitter*. Esto confirma la hipótesis H-c.

**Firma:** Camilo Garcia Placeres, como BE. Revisado como PL: la recomendación final usa solo el ATE y no el CATE.

---

## Entrada 5 · Trazabilidad del uso de IA

**Decide:** QA · **Revisa:** PL

### Declaración

Este proyecto se desarrolló con asistencia de **Claude (Anthropic)**, un modelo de lenguaje, usado de forma conversacional.

**Qué se delegó en la IA:**

- la propuesta de estructura del repositorio;
- la escritura del código de `src/`, del notebook (mediante un script generador), de la app y de los tests;
- la redacción de los textos interpretativos;
- la preparación de las diapositivas.

**Qué decidió el autor:**

- trabajar solo, asumiendo los cuatro roles;
- usar el dataset NHEFS;
- desplegar en GitHub y Streamlit Community Cloud.

### Prompts principales (resumen)

1. Pedido de resolver el Trabajo Integrador a partir de la consigna en PDF y la rúbrica, en forma individual.
2. Elección del dataset entre las opciones propuestas por la IA. Se eligió NHEFS.
3. Pedidos sucesivos de notebook, repo, app y diapositivas, construidos en ese orden.

### Errores de la IA detectados durante el desarrollo

| # | Error | Cómo se detectó | Corrección |
|---|---|---|---|
| 1 | El texto afirmaba que la hipertensión era un predictor significativo de muerte | Al contrastarlo con la tabla del logit: OR con p = 0,13 | Se corrigió el texto |
| 2 | El texto llamaba "hoja" a un nodo interno del CART (68% de muerte, 12% de la muestra) | Inspección del árbol graficado | Se corrigió la descripción |
| 3 | Se incluyeron dos dummies colineales en el logit, lo que dio IC = NaN | Salida del modelo con NaN | Se quitó `diabetes_cat` (Entrada 3) |
| 4 | El diagnóstico de White era demasiado costoso con tantas variables | El notebook no terminaba en tiempo razonable | Se usó solo Breusch-Pagan |
| 5 | Un parámetro deprecado de Streamlit (`use_container_width`) | Advertencias en la prueba de la app | Se reemplazó por `width="stretch"` |

### Pruebas deterministas que controlan lo generado por la IA

- `test_ate_ingenuo_y_ajustado_valores_de_referencia`: el ATE debe coincidir con el valor de referencia (2,54 y 3,35).
- `test_ipw_recupera_efecto_simulado`: con datos sintéticos de efecto conocido igual a 5, el IPW debe recuperar 5 ± 0,6. Controla que el estimador esté bien implementado y no solo que "dé un número".
- `test_ipw_determinista` y `test_dml_reproducible`: misma semilla, mismo resultado.
- `test_backdoor_*` y `test_conjunto_minimo_unico`: el DAG implica exactamente el conjunto de ajuste declarado.
- `tests/test_leakage.py`: el split, la imputación, el escalado y el PCA se aprenden solo con train.
- `test_modelo_reproduce_metricas`: el artefacto de la app reproduce exactamente el AUC del notebook.

### Revisión del autor

*(Completar antes de entregar.)*

- [ ] Leí y entiendo cada sección del notebook y puedo defenderla en el oral.
- [ ] Verifiqué al menos estos resultados contra la salida del notebook: ______________________
- [ ] Errores o ajustes adicionales que encontré yo: ______________________
- [ ] Lo que decidiría distinto en un proyecto real: ______________________

**Firma:** Camilo Garcia Placeres, como QA. Revisado como PL.

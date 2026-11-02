import json
from pathlib import Path

T = {}

T["F0"] = r"""
**Contexto operativo.** Una agencia de salud pública diseña un programa de cesación tabáquica para adultos fumadores. Hay dos obstáculos concretos:
1. **El miedo a engordar** es una de las razones más citadas para no dejar de fumar o para recaer. Para diseñar el componente de apoyo nutricional y comunicar expectativas realistas, el programa necesita saber **cuánto peso causa dejar de fumar**, no cuánto peso ganan quienes dejan (son preguntas distintas).
2. **El presupuesto es limitado.** Conviene priorizar a los fumadores con mayor **riesgo de muerte** en la próxima década, usando información que se pueda relevar en una entrevista inicial.

**Preguntas de decisión**

| | Pregunta | Tipo | Cantidad objetivo |
|---|---|---|---|
| P1 | ¿Qué fumadores tienen mayor probabilidad de morir en los próximos 10 años? | Predictiva | $P(death = 1 \mid X)$ |
| P2 | ¿Cuántos kg más gana una persona si deja de fumar, frente a si sigue fumando? | Causal | $E[Y \mid do(T=1)] - E[Y \mid do(T=0)]$ |
| P3 | ¿El aumento de peso es mayor en los fumadores más intensos o en los más jóvenes? | Causal (heterogeneidad) | $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ |

**Hipótesis adicionales a las de la consigna**
- **H-a (autoselección):** quienes dejaron de fumar no son comparables con quienes siguieron (difieren en edad e intensidad de consumo antes del tratamiento).
- **H-b (efecto metabólico de la nicotina):** el aumento de peso es mayor cuanto más se fumaba.
- **H-c ("*sick quitter*"):** la mayor mortalidad observada entre quienes dejaron de fumar no es un efecto causal de dejar, sino de quiénes dejan (personas mayores o que dejan por un diagnóstico).

**Restricciones reales del sistema:** los datos son observacionales (no hubo asignación aleatoria); el 3,9% de los participantes no tiene el peso de 1982 (censura); a mitad del estudio cambió el cuestionario; no hay datos de dieta (el mediador principal no está medido); y la muestra es de adultos de EE. UU. de los años 70, lo que limita la validez externa.
"""

T["F1_missing"] = r"""
| Variable(s) | Faltantes | Mecanismo | Evidencia | Estrategia |
|---|---|---|---|---|
| `diabetes`, `hbp`, `hbpmed`, `pica`, `boweltrouble` (código 2 oculto) | 791 (48,6%) | **MCAR respecto de la salud** (faltante por diseño) | Coinciden **exactamente** con los participantes de `seqn` ≥ 20.751: es un cambio de versión del cuestionario, y depende solo de un identificador administrativo | Categoría explícita `no_informado` (no se imputan diagnósticos inexistentes) |
| `pregnancies` | 903 | **Estructural** (no aplica) | 100% de los hombres | Se excluye |
| `alcoholhowmuch` | 417 | **Estructural** | 100% de quienes no bebieron en el último año | Se usa `alcoholfreq` |
| `wt82` / `wt82_71` (resultado causal) | 63 (3,9%) | **MAR condicional a las covariables**, con **riesgo de MNAR** | La censura depende del peso inicial (p = 0,011) y el modelo global es significativo (LR p = 0,0002): no es MCAR. Los censurados murieron más (42,9% vs. 18,6%) y eran más grandes y mayores | Análisis de casos completos + **IPCW** como sensibilidad (Fase 4) |
| `income`, `cholesterol` | 62 / 16 | MAR plausible | < 4% | Imputación por mediana **dentro del Pipeline** (ajustada solo en train) |
| `price82`, `tax82` | 92 | MAR (estado de residencia desconocido) | Coincide con `birthplace` faltante | Casos completos, solo en el análisis de instrumento |

**Justificación matemática.** Bajo **MCAR**, la muestra completa es una submuestra aleatoria y cualquier estimador consistente sigue siéndolo. Bajo **MAR** ($R \perp\!\!\!\perp Y \mid X$), el análisis de casos completos sesga las medias marginales, pero ponderar cada caso observado por $1/\pi(X)$, con $\pi(X) = P(R = 1 \mid X)$, recupera el valor poblacional: $E[Y] = E\big[R\,Y/\pi(X)\big]$. Eso es lo que hace IPCW. Bajo **MNAR**, ninguna corrección basada en lo observado alcanza: queda como limitación explícita.
"""

T["F1_tests"] = r"""
- **H1 — se rechaza $H_0$.** Quienes dejaron de fumar ganaron **4,53 kg** frente a **1,98 kg** de quienes siguieron fumando: diferencia de **2,54 kg (IC 95%: 1,58 a 3,50)**. Welch: t = 5,21 con 616 gl, p = 2,6 × 10⁻⁷. Mann-Whitney coincide (p = 4,3 × 10⁻⁸). El tamaño del efecto es pequeño a mediano (d de Cohen = 0,33). Con estos tamaños de muestra la potencia es ≈ 1; para detectar una diferencia de apenas 1 kg con potencia 0,80 harían falta ≈ 976 personas por grupo.
- **H2 — hay autoselección.** Quienes dejaron de fumar eran **3,4 años mayores** (IC 95%: 2,0 a 4,8; p = 1,6 × 10⁻⁶) y fumaban **2,6 cigarrillos/día menos** (Mann-Whitney p = 1,7 × 10⁻⁵) **antes** de dejar.
- **Consecuencia:** la diferencia de H1 es estadísticamente significativa, pero **no es un efecto causal**. Los dos grupos ya eran distintos en factores que influyen en el peso: las personas mayores tienden a ganar menos peso, así que la diferencia cruda probablemente **subestima** el efecto de dejar de fumar. La Fase 4 lo corrige.
"""

T["F1_anova"] = r"""
- **Normalidad:** Shapiro-Wilk rechaza en los 5 grupos (p entre 10⁻³ y 10⁻¹⁶). Con 121 a 637 observaciones por grupo, el test detecta desvíos mínimos. El criterio relevante para el TCL se cumple: **la asimetría está en [−1; 1]** en 4 de 5 grupos (el grupo 4 tiene 1,05, en el límite) y los Q-Q Plots son lineales en el centro. Lo que sí hay son **colas pesadas** (curtosis en exceso de 2,2 a 5,5).
- **Homocedasticidad:** Levene rechaza (W = 3,12; **p = 0,014**), aunque la razón entre varianzas extremas es moderada (1,55).
- **Elección del test:** por la heterocedasticidad demostrada, el contraste principal es el **ANOVA de Welch**: F = 4,21 con (4; 498,8) gl, **p = 0,0023**. Kruskal-Wallis, robusto también a las colas pesadas, coincide (H = 20,0; p = 0,0005). El ANOVA clásico da casi lo mismo (p = 0,0017). **Se rechaza $H_0$**: el aumento de peso medio difiere según el nivel educativo.
- **Magnitud:** η² = 0,011. La educación explica apenas el **1,1%** de la variabilidad: la diferencia es estadísticamente significativa, pero chica en la práctica.
- **Post-hoc:** Tukey HSD (que supone varianzas iguales) encuentra que el grupo 1 (≤ 8.º grado, +1,1 kg) difiere de los grupos 2, 3 y 4. **Bonferroni sobre t de Welch**, más conservador y válido con varianzas distintas, confirma solo 1 vs. 3 (p = 0,008) y 1 vs. 4 (p = 0,003); 1 vs. 2 deja de ser significativo (p = 0,09). Con heterocedasticidad, se prefiere la segunda.
- **Advertencia:** el grupo de menor educación es también el de **mayor edad**, y la edad reduce el aumento de peso. Este ANOVA describe diferencias entre grupos, pero no prueba que la educación cause un menor aumento de peso.
"""

T["F2_diag"] = r"""
- **Ajuste:** R² = 0,126 (ajustado 0,114); F = 10,6, p ≈ 2 × 10⁻³¹. El cambio de peso a 11 años es muy variable entre personas (RMSE ≈ 7,4 kg), y las covariables basales explican poco. Dejar de fumar se asocia con **+3,21 kg** (t = 7,1), cada año de edad con −0,21 kg y cada kg de peso inicial con −0,10 kg (los más pesados ganan menos: regresión a la media).
- **Homocedasticidad:** Breusch-Pagan **rechaza** (LM = 81,5; p < 0,001). Para la inferencia se usan errores robustos **HC3**: el EE de `qsmk` pasa de 0,45 a 0,49, y el coeficiente sigue siendo claramente significativo.
- **Normalidad:** Jarque-Bera rechaza (JB = 1.104). Los residuos tienen asimetría 0,52 y **curtosis en exceso de 4,1** (colas pesadas). Con n = 1.487, la inferencia sobre los coeficientes sigue siendo válida por el TCL, pero los intervalos de **predicción** individuales gaussianos serían demasiado angostos.
- **Independencia:** Durbin-Watson = 2,01. No hay autocorrelación según el orden de `seqn`; los datos son de corte transversal.
- **Observaciones extremas:** 19 *outliers* (|r| > 3), 33 puntos de alto *leverage* y 76 con Cook > 4/n, pero **ninguno con Cook > 0,5** (máximo 0,03). Los casos más extremos (−41 y +49 kg en 11 años) son plausibles en una cohorte de adultos (enfermedad grave, obesidad), así que **no se eliminan**. Como sensibilidad, una **regresión robusta de Huber** reduce el coeficiente de `qsmk` de 3,21 a 2,81 (−12,5%): las colas pesadas lo inflan un poco, pero el signo, la magnitud y la significancia se mantienen.
"""

T["F2_vif"] = r"""
- **`school` / `education` (VIF ≈ 7,7 / 7,5; Spearman = 0,99):** miden el mismo constructo, uno en años y el otro en categorías. **Se elimina `school`** y se conserva `education` como categórica, que captura efectos no lineales.
- **`age` / `smokeyrs` (VIF ≈ 4,7; r = 0,87):** cuanto mayor es la persona, más años lleva fumando. En lugar de eliminar una, se usa una **combinación funcional**: `edad_inicio = age − smokeyrs` (edad a la que empezó a fumar), con una correlación de solo 0,25 con la edad. Después de la resolución, **todos los VIF son < 2,4**.
- **Para la estimación causal** (Fase 4), `age` y `smokeyrs` se mantienen ambas porque son confusores distintos del DAG. La colinealidad afecta la precisión de sus propios coeficientes, no la del coeficiente del tratamiento. En la Fase 3, la regularización **Ridge/L2** absorbe la colinealidad que queda.
"""

T["F2_pca"] = r"""
- **Cantidad de componentes:** por Kaiser se retienen **3 componentes** (λ = 2,29; 1,61; 1,20), que explican el **63,7%** de la varianza. El codo del *scree plot* está en PC3–PC4, y para superar el 85% harían falta 5.
- **PC1 (28,6%) — "envejecimiento y exposición acumulada en contexto socioeconómico bajo":** cargas positivas en edad (+0,59) y años fumando (+0,58), negativas en escolaridad (−0,40) e ingreso (−0,32).
- **PC2 (20,1%) — "tamaño corporal e intensidad de consumo":** altura (+0,62), peso (+0,58) y cigarrillos/día (+0,36).
- **PC3 (15,0%) — "nivel socioeconómico y colesterol"**, con cargas positivas en ingreso, escolaridad y colesterol.
- En la proyección del test (con la rotación aprendida en train), los fallecidos se concentran en **valores altos de PC1**: el eje de envejecimiento y exposición es el que ordena el riesgo de muerte, lo que anticipa la Fase 3.
- El control anti-*leakage* confirma que las medias del escalador son exactamente las de train.
"""

T["F3"] = r"""
**Selección del modelo (por CV, nunca por test).** La **logística regularizada** gana en CV en las dos métricas: AUC 0,838 y Brier 0,113. Le siguen el Random Forest (0,834 / 0,118), el árbol podado (0,776 / 0,130) y Naive Bayes (0,756 / 0,202). En test, el RF tiene un AUC levemente mayor (0,855 vs. 0,847), pero con solo 64 muertes en test esa diferencia está dentro del ruido. Se elige la logística calibrada por tres razones: mejor CV, mejor calibración y coeficientes interpretables. **Modelo final: logística L2 calibrada (sigmoide), AUC 0,847, PR-AUC 0,547, Brier 0,117** (la referencia de predecir la prevalencia da Brier 0,158).

**Odds Ratios (statsmodels, train):** cada año de edad multiplica las chances de morir en el período por **1,085** (≈ ×2,3 por década). Ser mujer las reduce (OR ≈ 0,62) y cada año adicional fumando las aumenta (OR ≈ 1,03). La hipertensión diagnosticada tiene un OR de ≈ 1,5, pero no es significativo (p = 0,13). Dejar de fumar **no** muestra un OR significativo (IC que incluye al 1). La interpretación causal de esto se discute en el ensayo.

**Naive Bayes y la violación de independencia:** dentro de la clase "vivo", edad y años fumando tienen una correlación de **0,86**, y peso y altura de 0,49. NB multiplica sus verosimilitudes como si fueran independientes y cuenta dos veces la misma evidencia. Por eso su Brier (0,230) es **peor que no usar modelo** (0,158), y en el *reliability diagram* sus probabilidades se pegan a 0 y a 1 (sobreconfianza). La calibración isotónica baja el Brier a 0,139, pero **no mejora el AUC** (0,77): calibrar arregla las probabilidades, no el orden de los casos.

**CART:** sin restricciones, el árbol memoriza el entrenamiento (180 hojas; AUC train 1,00 vs. test 0,62). La poda por costo-complejidad (α = 0,0031, elegido por CV) lo reduce a **7 hojas** y sube el AUC de test a 0,81. **Nodo raíz: edad ≤ 54,5 años.** En el subárbol de los mayores, el siguiente corte es haber fumado más de 39,5 años: ese grupo (12% de la muestra de entrenamiento) tiene una mortalidad observada del 68%.

**Random Forest:** la búsqueda aleatoria (30 combinaciones × 5 folds) eligió 600 árboles de profundidad 8, `min_samples_leaf = 20`, `min_samples_split = 30` y `max_features = 'sqrt'`: árboles bastante regularizados, coherente con una señal dominada por pocas variables. El AUC OOB se estabiliza a partir de ≈ 200 árboles y queda en el mismo nivel que el AUC de CV. El OOB es una alternativa válida a la CV en datos i.i.d. como estos (a diferencia de una serie temporal).

**MDI vs. permutación:** MDI pone a `smokeyrs` casi a la par de `age` (0,29 vs. 0,30), pero por permutación en test la edad es más del doble de importante (ΔAUC 0,085 vs. 0,038). `wt71` y `ht` (488 y 358 valores únicos) reciben MDI de 0,03 y 0,02 y **permutación ≈ 0 o negativa**. El **control con ruido puro** lo confirma: una variable continua aleatoria recibe **5,7 veces más MDI** que una binaria aleatoria (0,0245 vs. 0,0043), mientras que la permutación les asigna ≈ 0 a ambas.

La razón es que MDI acumula reducciones de impureza **en train**. Una variable con cientos de umbrales posibles encuentra, por azar, cortes que separan ruido en los nodos profundos; una binaria tiene un solo umbral. La permutación mide la caída de desempeño **en datos no vistos**, y el ajuste al ruido no generaliza, así que ese sesgo se neutraliza.
"""

T["F4_dag"] = r"""
| Arco(s) | Justificación de dominio |
|---|---|
| Z → T (sexo, raza, edad, educación, cigarrillos/día, años fumando, ejercicio, actividad, peso 1971) | Determinantes conocidos de la decisión de dejar de fumar: los más jóvenes, educados y livianos de consumo tienen más éxito; la preocupación por el peso influye en la decisión. |
| Z → Y | Los mismos factores afectan el cambio de peso: edad (metabolismo), sexo, peso inicial (regresión a la media), actividad física. |
| edad → años fumando; sexo, edad → peso 1971 | Relaciones entre confusores (años fumando depende de la edad; el peso depende de sexo y edad). |
| T → M → Y (M = ingesta calórica / metabolismo, **latente**) | Mecanismo biológico: la nicotina suprime el apetito y eleva el gasto energético; al dejar, aumentan la ingesta y la eficiencia metabólica. **No está medido.** |
| T → Y | Efecto total que se quiere estimar (incluye el paso por M). |
| precio del tabaco 82 → T | Candidato a **instrumento**: el precio incentiva dejar de fumar y no debería afectar el peso por otra vía. |
| T → C ← Y (C = presión sistólica 1982) | Dejar de fumar modifica la presión, y el aumento de peso la eleva: **colisionador post-tratamiento**. |
| U (conciencia de salud, **latente**) → T, ejercicio | **Amenaza a la validez:** personas más preocupadas por su salud dejan más y hacen más ejercicio. Si U también afectara al peso por vías no capturadas por Z, habría confusión no observada. |
| T, edad, peso 1971 → S (peso 1982 observado) | **Selección por censura:** analizar solo a los observados es condicionar en S. |
"""

T["F4_backdoor"] = r"""
- **Caminos:** entre T e Y hay **31 caminos**: 1 causal (T → Y), **17 de puerta trasera abiertos** (por ejemplo T ← edad → Y, T ← sexo → peso71 → Y) y 13 **bloqueados por colisionadores** (sbp, y también `wt71` en el camino T ← sexo → wt71 ← edad → Y).
- **Conjunto mínimo:** la búsqueda exhaustiva confirma que **los 9 confusores son el único conjunto mínimo válido**. Cada uno abre un *backdoor* propio (Z → T y Z → Y), así que quitar cualquiera deja un camino abierto (por ejemplo, sin `wt71` o sin `smokeintensity`, falla el criterio).
- **Sutileza:** `wt71` es colisionador en T ← sexo → wt71 ← edad → Y. Ajustar por `wt71` abre ese camino, pero como sexo y edad también están en el conjunto, queda bloqueado de nuevo. El conjunto es válido **en bloque**.
- **Ningún nodo de Z desciende de T** (todos son de 1971), así que se cumplen las dos condiciones del criterio y $P(y \mid do(t)) = \sum_z P(y \mid t, z)\,P(z)$.

**Preguntas trampa**
- **¿Y si se incluyera el mediador M?** Si se midiera la ingesta calórica y se la usara como control, se bloquearía T → M → Y. Se estimaría solo el efecto "directo" de dejar de fumar sin cambios en la dieta, que **subestima** el aumento de peso que el programa tiene que anticipar. Además, M es descendiente de T y viola la condición 1.
- **¿Y si se condicionara en el colisionador C (presión 1982)?** Se abriría T → C ← Y (sesgo de Berkson). Empíricamente, en la misma muestra el ATE pasa de **3,41 a 3,33 kg**: el sesgo existe y va en la dirección prevista, pero es chico porque la presión depende débilmente de ambos. Que el sesgo sea chico en estos datos no justifica incluir C: se excluye por el DAG.
- **¿Y el instrumento?** {Z + precio82} también cumple el criterio, pero agregar un instrumento como control **no reduce el sesgo y aumenta la varianza**. Si hubiera confusión no observada, además la **amplificaría** (*bias amplification*). Se excluye.
- **Selección (S):** usar solo a los no censurados es condicionar en S. Si el peso de 1982 afectara la probabilidad de ser medido, se abriría un camino espurio. La sensibilidad con IPCW (MAR) da el mismo ATE (3,35 kg).
"""

T["F4_estim"] = r"""
- **Paso 2 (Identify):** DoWhy identifica el efecto por *backdoor* con los 9 confusores: el estimando $\frac{d}{dT}E[Y \mid Z]$ es la notación de $E_Z\big[E(Y \mid T=1, Z) - E(Y \mid T=0, Z)\big]$, es decir, comparar dentro de cada perfil de confusores y promediar según la distribución poblacional de Z. También propone un estimando de **variable instrumental** (precio del tabaco), pero la primera etapa tiene **F = 0,38**, muy por debajo de 10: el instrumento es **débil** y su estimación (−26,9 kg) no es informativa. Por eso no sirve para contrastar el supuesto de no confusión.
- **Paso 3 (Estimate):** el ATE ajustado es **3,35 kg (IC 95%: 2,48 a 4,21)** por regresión y **3,28 kg (IC bootstrap: 2,37 a 4,34)** por IPW. El estimador ingenuo (2,54 kg) **subestima el efecto en 0,81 kg (−24%)**: el sesgo de confusión es **negativo**. Quienes dejaron de fumar eran mayores, y la edad reduce el aumento de peso, lo que "esconde" parte del efecto real. Las especificaciones alternativas coinciden: MCO flexible con términos cuadráticos **3,46 kg** (igual al valor de referencia de Hernán & Robins para estos datos), IPCW 3,35 kg y Double ML **3,62 kg (IC 95%: 2,67 a 4,57)**. El rango completo es 3,0 a 3,6 kg.
- **Paso 4 (Refute):** las tres pruebas dan **semáforo verde**:
  - causa común aleatoria: 3,350 kg (p = 0,90);
  - placebo: el efecto colapsa a −0,03 kg (p = 0,88);
  - subconjuntos del 80%: 3,334 kg (p = 0,98), dentro del IC original.

  En refutación, un p alto significa robustez. Ninguna de las pruebas puede descartar al confusor latente U.
- **Double ML y CATE:** $\hat\tau(x) = 3{,}21 + 0{,}044 \cdot \text{cig/día} - 0{,}011 \cdot \text{edad}$. El signo apoya la hipótesis H-b (más aumento de peso en los fumadores intensos: +0,44 kg cada 10 cigarrillos/día), pero **los IC de ambos coeficientes incluyen al 0**: no hay evidencia de heterogeneidad significativa. Ningún subgrupo del soporte tiene un efecto neutro: el aumento de peso estimado va de ≈ 2,5 a ≈ 5 kg y es positivo en todos los perfiles.
- **Implicancia para la decisión:** planificar el apoyo nutricional para **todos** los que dejan, con una expectativa de **≈ 3,4 kg en 11 años**. Una priorización de los fumadores intensos es razonable como hipótesis, pero estos datos no alcanzan para justificarla.
"""

T["ENSAYO"] = r"""
**1. Si la variable más importante del Random Forest es la edad, ¿es una palanca para intervenir?** No, por dos razones:
- La edad **no es manipulable**: no existe $do(\text{edad} = x)$. Sirve para **seleccionar** a quién priorizar, no para **intervenir**.
- Aunque una variable sea manipulable, su importancia predictiva (MDI o permutación) mide cuánto ayuda a **ordenar** los casos bajo la distribución observada, $P(Y \mid X)$. No dice qué pasaría al **cambiarla**, $P(Y \mid do(X))$. La importancia por permutación rompe la asociación de X con Y y con el resto de las variables en el **modelo**, no en el **mundo**.

**2. ¿Cuándo una variable muy predictiva no tiene poder causal, o lo tiene de signo opuesto?**
- cuando su asociación con Y se debe a una **causa común** (confusión);
- cuando hay **causalidad inversa** (Y, o sus precursores, causan X);
- cuando X es un **marcador o proxy** de la causa real;
- cuando la muestra fue **seleccionada** por un colisionador.

Los datos de este trabajo tienen un ejemplo de manual. Entre quienes **dejaron de fumar**, la mortalidad observada fue **23,8%**, contra **18,0%** entre quienes siguieron: +5,8 puntos porcentuales. Un modelo predictivo puede usar `qsmk` como señal de **mayor** riesgo. Pero al ajustar por los confusores, el efecto de dejar de fumar sobre la mortalidad cae a **+0,5 p.p. (IC 95%: −3,4 a +4,5)**, compatible con cero. La asociación cruda viene de que quienes dejan son **mayores** (H2) y, plausiblemente, del efecto "*sick quitter*" (dejar tras un diagnóstico: causalidad inversa no medida).

Un directorio que "optimizara" la mortalidad leyendo el modelo predictivo concluiría que **conviene desalentar la cesación**: una política no solo inútil sino dañina, porque la evidencia médica de largo plazo muestra que dejar de fumar reduce la mortalidad. El ANOVA de la Fase 1 muestra un caso parecido: la educación "predice" menos aumento de peso, pero el grupo de menor educación es también el de mayor edad.

**Conclusión:** la Fase 3 responde **a quién mirar** (riesgo) y la Fase 4 responde **qué pasa si actuamos** (efecto). Mezclarlas lleva a optimizar correlaciones espurias. Toda palanca de política tiene que pasar por un DAG explícito, un criterio de identificación y pruebas de refutación antes de llegar a la mesa de decisión.
"""

T["CONC"] = r"""
**Hallazgos**
1. **Calidad de datos:** se detectó un bloque oculto de faltantes **por diseño** (48,6%, cambio de cuestionario), faltantes estructurales y una censura del resultado causal que **no es MCAR** (se corrigió con IPCW, bajo el supuesto MAR).
2. **Inferencia clásica:** quienes dejan de fumar ganan 2,5 kg más en bruto, pero son 3,4 años mayores y fuman menos: hay autoselección. El ANOVA de Welch encuentra diferencias por educación (p = 0,002) de magnitud muy chica (η² = 1,1%).
3. **Modelo lineal:** hay heterocedasticidad y colas pesadas (se usó HC3 y Huber como sensibilidad), sin puntos con influencia excesiva (Cook máximo 0,03). La colinealidad se resolvió eliminando `school` y reemplazando `smokeyrs` por la edad de inicio. Los 3 componentes principales tienen interpretación clara (envejecimiento, tamaño corporal, nivel socioeconómico).
4. **Predicción de mortalidad:** la logística regularizada y calibrada logra **AUC 0,85 y Brier 0,117**, a la par del Random Forest y mucho mejor que Naive Bayes, al que la dependencia entre predictores lo vuelve sobreconfiado.
5. **Efecto causal:** dejar de fumar **causa un aumento de ≈ 3,4 kg en 11 años (IC 95%: 2,5 a 4,2)**. Los métodos coinciden (de 3,0 a 3,6 kg según el estimador) y pasan las tres refutaciones. El análisis ingenuo lo subestimaba un 24%. No se encontró heterogeneidad significativa por intensidad de consumo ni por edad.

**Recomendaciones para el programa de cesación**
- **Incluir apoyo nutricional y de actividad física para todos los que dejen**, con una meta comunicada realista: **≈ 3–4 kg de aumento esperado en 11 años**. Es un costo de salud mucho menor que el de seguir fumando.
- **Priorizar el reclutamiento con el modelo de riesgo** (edad, años fumando, sexo). Al usar probabilidades calibradas, se puede presupuestar el número esperado de eventos por segmento.
- **No usar la mortalidad cruda de quienes dejan** como indicador del programa: está confundida por edad y por el efecto *sick quitter*.
- **Validar la hipótesis de mayor aumento de peso en fumadores intensos** con un diseño experimental o una muestra mayor antes de diferenciar la intervención.

**Limitaciones y amenazas a la validez**
- La no confusión no se puede verificar: la "conciencia de salud" (U) podría sesgar el resultado, y el instrumento disponible es débil.
- La censura podría ser MNAR (los censurados murieron más).
- El mediador (dieta) no está medido.
- La cohorte es de EE. UU. en los años 70.
- El CATE se modeló como lineal.
- La mortalidad se observa en una ventana de 10 años.
"""

T["REFS"] = r"""
- Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC. (Fuente de la versión del dataset NHEFS y del análisis de referencia.)
- National Center for Health Statistics. *NHANES I Epidemiologic Follow-up Study (NHEFS)*. Centers for Disease Control and Prevention.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2.ª ed.). Cambridge University Press.
- Pearl, J., Glymour, M. & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Sharma, A. & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. *arXiv:2011.04216*.
- Battocchi, K. et al. (2019). *EconML: A Python package for ML-based heterogeneous treatment effects estimation*. Microsoft Research.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Strobl, C., Boulesteix, A.-L., Zeileis, A. & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of ICML*.
- Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3.ª ed.). Wiley.
- Welch, B. L. (1951). On the comparison of several mean values: an alternative approach. *Biometrika*, 38, 330–336.
- Breusch, T. S. & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287–1294.
- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15–18.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Seabold, S. & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*.
"""

Path(__file__).with_name("textos.json").write_text(json.dumps(T, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")

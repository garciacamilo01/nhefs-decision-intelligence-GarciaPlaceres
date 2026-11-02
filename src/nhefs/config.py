"""Configuración central del proyecto: rutas, semilla y roles de las variables.

Toda constante que comparten el notebook, los scripts, la app y los tests vive acá,
para que exista una única fuente de verdad (contrato de datos entre fases).
"""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "nhefs.csv"
CODEBOOK = ROOT / "data" / "raw" / "NHEFS_codebook.csv"
MODELS_DIR = ROOT / "models"

URL_NHEFS = (
    "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
    "causallib/datasets/data/nhefs/NHEFS.csv"
)

# ---------------------------------------------------------------- Pregunta causal
TREATMENT = "qsmk"          # 1 = dejó de fumar entre 1971 y 1982
OUTCOME_CAUSAL = "wt82_71"  # cambio de peso 1971 -> 1982 (kg)

# Confusores basales (medidos en 1971, antes del tratamiento)
CONFOUNDERS = [
    "sex", "race", "age", "education", "smokeintensity",
    "smokeyrs", "exercise", "active", "wt71",
]
# Variables de heterogeneidad del efecto (X en EconML)
EFFECT_MODIFIERS = ["smokeintensity", "age"]
# Instrumento candidato: precio del tabaco en el estado de residencia (1982)
INSTRUMENT = "price82"
# Colisionador medido: presión sistólica en 1982 (afectada por dejar de fumar y por el aumento de peso)
COLLIDER = "sbp"

# ---------------------------------------------------------------- Pregunta predictiva
OUTCOME_PRED = "death"  # muerte entre 1983 y 1992

NUM_PRED = [
    "age", "smokeintensity", "smokeyrs", "wt71", "ht", "cholesterol",
    "income", "alcoholfreq",
]
CAT_PRED = ["sex", "race", "education", "exercise", "active", "marital_cat", "qsmk"]
BIN_PRED = [
    "asthma", "bronch", "tb", "hf", "pepticulcer", "colitis", "hepatitis",
    "chroniccough", "hayfever", "polio", "tumor", "nervousbreak", "weakheart",
    "otherpain", "headache", "nerves", "lackpep", "allergies", "infection", "wtloss",
]
# Variables del bloque del cuestionario que no se administró a la segunda cohorte (código 2 = no informado)
BLOCK_VARS = ["diabetes", "hbp", "hbpmed", "pica", "boweltrouble"]
PRED_FEATURES = NUM_PRED + CAT_PRED + BIN_PRED + [f"{v}_cat" for v in ["diabetes", "hbp"]]

PCA_VARS = ["age", "smokeyrs", "smokeintensity", "wt71", "ht", "cholesterol", "income", "school"]

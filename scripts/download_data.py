"""Descarga el CSV de NHEFS y su codebook a data/raw/ (no se necesita si ya están en el repo)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhefs.config import URL_NHEFS  # noqa: E402

if __name__ == "__main__":
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.read_csv(URL_NHEFS).to_csv(raw / "nhefs.csv", index=False)
    pd.read_csv(URL_NHEFS.replace("NHEFS.csv", "NHEFS_codebook.csv")).to_csv(raw / "NHEFS_codebook.csv", index=False)
    print("OK:", sorted(p.name for p in raw.iterdir()))

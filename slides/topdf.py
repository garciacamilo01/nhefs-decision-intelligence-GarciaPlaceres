"""Renderiza slides.html a PDF (1280×720 por página) con Chromium headless."""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.goto((HERE / "slides.html").as_uri()); pg.wait_for_timeout(800)
    pg.pdf(path=str(HERE / "TPI_Presentacion_Garcia_Camilo.pdf"), width="1280px", height="720px", print_background=True)
    b.close()

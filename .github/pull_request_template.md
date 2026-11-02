## Qué cambia
<!-- Breve descripción del cambio y del ticket del backlog al que responde. -->

## Rol que lo propone
- [ ] Project Lead  - [ ] Backend / ML-Causal  - [ ] Frontend  - [ ] QA

## Checklist de QA
- [ ] `ruff check src app scripts tests` sin errores
- [ ] `pytest` en verde (incluye pruebas de fuga de datos y del DAG)
- [ ] Si cambió algún modelo: `python scripts/build_artifacts.py` y artefactos actualizados en `/models`
- [ ] El notebook corre con *Restart & Run All*
- [ ] Bitácora actualizada si hubo una decisión metodológica o un uso de IA relevante

## Veredicto de QA
<!-- [APROBADO] o [BLOQUEADO: motivo] -->

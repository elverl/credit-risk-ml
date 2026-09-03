# Chunking determinista de la knowledge base

## Alcance

El proceso transforma únicamente los nueve archivos Markdown vigentes de
`docs/knowledge_base/` en `artifacts/rag/kb_chunks.jsonl`. No modifica las
fuentes y excluye documentos históricos o V1. Esta fase no implementa
embeddings, índices, recuperación ni generación RAG.

## Criterio de segmentación

La unidad base es una sección Markdown de nivel 2. Las excepciones responden a
unidades semánticas presentes en el contenido, no a longitud:

- cada fila de feature del diccionario produce un chunk con `feature_name`;
- la fila de `incumplimiento` produce un chunk con `target_name`;
- metodología separa train/test, OOF, Youden, threshold, selección del champion
  y métricas según sus secciones y pasos enumerados;
- el alcance SHAP separa interpretación, signo positivo, signo negativo y no
  causalidad; sus limitaciones permanecen en un chunk propio.

El texto conserva extractos literales del documento y sus encabezados o tablas.
Puede repetirse contexto ya existente para que una unidad sea interpretable por
sí sola, pero no se agrega conocimiento nuevo.

## IDs y orden

El formato es `kb1-<fuente>--<sección>[--<unidad>]`. Los componentes se
normalizan a ASCII, minúsculas y guiones. El ID no incorpora offsets, hashes del
texto ni tamaños, por lo que cambios de espaciado o contenido que no alteren la
ruta semántica mantienen el identificador. Los archivos se recorren por nombre
y las unidades conservan su orden documental.

## Metadata y trazabilidad

Cada chunk incluye fuente, ruta de sección, tipo documental y la metadata
`kb_version`, `model_version` y `status` tomada del front matter. Los tags son
determinísticos. El diccionario añade `feature_name` o `target_name` cuando
corresponde.

## Evalset

El constructor agrega `expected_chunk_ids` cuando puede resolver una pareja de
`expected_sources` y `expected_sections`. En referencias amplias a la tabla de
features, limita la selección a nombres técnicos presentes en los campos ya
existentes del caso. La actualización preserva el resto de campos y su orden.

## Reproducción y validación

Ejecutar:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_rag_chunks.py
python scripts/build_rag_chunks.py --check
pytest tests/test_rag_chunking.py
```

`--check` vuelve a construir todo en memoria y falla si el artefacto o el
evalset no coinciden byte a byte con la salida determinística.

---
kb_version: 1
status: current
scope: provenance_and_versioning
model_version: v2
---

# Provenance y versionado

## Jerarquía de fuentes

En caso de duplicación o contradicción, aplicar este orden:

1. **Artifacts estructurados V2 vigentes** para valores, resultados, thresholds
   y ranking: CSV/JSON de `artifacts/metrics/`.
2. **Runtime `src/` y tests** para comportamiento implementado, contratos e
   invariantes técnicos.
3. **`docs/model_card.md`** para narrativa, intended use, usos fuera de alcance,
   limitaciones y responsible use.
4. **Documentos de `docs/knowledge_base/`** como consolidación recuperable de
   las fuentes anteriores y definiciones aprobadas por el owner.
5. **Reportes históricos V2** solo para provenance; no prevalecen sobre
   artifacts, runtime o knowledge base actual.
6. **V1** queda excluido del corpus operativo actual.

Las definiciones de target, variables binarias, `-1` y escalas incorporadas en
`data_dictionary.md` fueron aprobadas expresamente por el owner del proyecto
para `kb_version: 1`.

## Estado y alcance

`kb_version: 1`, `status: current` y `model_version: v2` identifican esta primera
versión controlada. “Current” significa vigente dentro del repositorio; no
implica validación productiva, regulatoria o de negocio externa.

Los documentos deben actualizarse de forma coordinada cuando cambien:

- modelo champion o artifact;
- features, target o semántica de datos;
- threshold o bandas;
- metodología o resultados;
- proveedor/modelo LLM;
- intended use, limitaciones o controles.

No existen todavía chunk IDs automáticos. El futuro proceso de chunking deberá
preservar metadata de documento, versión, scope y vigencia.

## Exclusiones operativas

No recuperar como estado vigente:

- `llama-3.1-8b-instant`;
- configuraciones Groq o prompts anteriores al baseline actual;
- modelos, hiperparámetros, thresholds o resultados V1;
- snapshots de estado Git, logs, warnings o incidencias ya resueltas;
- IDs de runs salvo una consulta explícita de trazabilidad MLE;
- propuestas futuras como si estuvieran implementadas.

El estado GenAI vigente usa Groq con `openai/gpt-oss-20b`. El runtime conserva
`requests.post()` como transporte. RAG, embeddings, vector stores, retrievers,
reranking y evalset no están implementados.

## Reglas de respuesta futura

- Recuperar hechos desde la fuente de mayor jerarquía disponible.
- Distinguir resultado observado, limitación y propuesta futura.
- Responder “No documentado en el proyecto” cuando falte evidencia.
- No completar definiciones mediante intuición sobre nombres de variables.
- No convertir asociaciones o SHAP en causalidad.
- No mezclar contenido V1 con el baseline V2.


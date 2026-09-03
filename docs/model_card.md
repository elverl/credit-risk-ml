# Model Card — Credit Risk XGBoost V2

## 1. Identidad y propósito

| Campo | Valor validado |
|---|---|
| Modelo | XGBoost (`XGBClassifier`) |
| Tarea | Clasificación binaria tabular |
| Estado | Champion del benchmark de cuatro modelos |
| Global score | 0.916667 |
| Threshold | 0.2399873 |
| Experimento MLflow | `credit-risk-model-benchmark` |

El modelo estima `probability_default` para apoyar análisis técnico dentro del
alcance del dataset estudiado. `incumplimiento=1` significa alcanzar 30+ DPD
dentro de los 12 meses posteriores a la originación; `0` significa no alcanzar
ese evento en la ventana.

No es una política de crédito ni un mecanismo autónomo de aprobación o rechazo.

## 2. Datos

| Propiedad | Valor |
|---|---:|
| Filas | 41,976 |
| Features seleccionadas | 15 |
| Train | 33,580 (80%) |
| Test | 8,396 (20%) |
| Split | Aleatorio estratificado, `random_state=42` |

La procedencia externa precisa, el método de muestreo y los criterios de
inclusión no están documentados. Las features, escalas, códigos y limitaciones
semánticas están en el [diccionario vigente](knowledge_base/data_dictionary.md).

## 3. Metodología y selección

Se compararon Logistic Regression, Random Forest, XGBoost y LightGBM. Para cada
candidato se generaron probabilidades OOF exclusivamente dentro de Train usando
cinco folds estratificados. El threshold se eligió maximizando Youden J sobre
Train OOF; Test no participó en esa selección.

Después, cada modelo se ajustó con todo Train y se evaluó en Test. XGBoost fue
seleccionado mediante un score que pondera por igual discriminación,
calibración y clasificación. Como Test sí participó en la comparación de
candidatos, no constituye un holdout externo independiente posterior.

## 4. Desempeño offline

Resultados Test congelados en
`artifacts/metrics/four_model_benchmark_metrics.csv`:

| Métrica | XGBoost |
|---|---:|
| AUC | 0.706305 |
| Gini | 0.412610 |
| KS | 0.296907 |
| PR-AUC | 0.433330 |
| Log Loss | 0.489626 |
| Brier Score | 0.158818 |
| Accuracy | 0.651977 |
| Balanced Accuracy | 0.645288 |
| Precision | 0.356228 |
| Recall | 0.632885 |
| F1-score | 0.455866 |
| MCC | 0.248831 |
| FPR | 0.342309 |
| FNR | 0.367115 |
| Youden threshold | 0.2399873 |

Estas métricas son offline. No existe evidencia de desempeño productivo.

## 5. Explainability

La implementación usa `TreeExplainer` para obtener contribuciones SHAP
alineadas con las 15 features y con la clase positiva `incumplimiento=1`.
SHAP describe contribuciones al output respecto de una referencia: no demuestra
causalidad, no valida fairness y no constituye una decisión o recomendación
crediticia.

## 6. Capa GenAI y RAG

El provider es Groq y el modelo vigente es `openai/gpt-oss-20b`. La capa GenAI
genera narrativa a partir de resultados ya calculados; no cambia
`probability_default`, `predicted_class`, `decision_threshold`, `risk_band` ni
los valores SHAP.

Se validaron dos variantes sobre el mismo evalset de 24 casos:

- **Baseline:** XGBoost + SHAP/contexto predictivo + GPT-OSS-20B, sin retrieval.
- **RAG V1:** el mismo contexto más Top-3 desde una knowledge base de 9
  documentos y 56 chunks.

El retriever `local-lsa-tfidf-svd-v1` obtuvo Hit Rate@3 `0.750000`, MRR@3
`0.597222` y 18/24 hits. Una cuarta parte del evalset no recuperó un expected
chunk en Top-3; es una limitación directa de la evidencia para generación.

## 7. Evaluación GenAI

| Métrica | Baseline | RAG | Delta |
|---|---:|---:|---:|
| Factual correctness | 0.312500 | 0.619444 | +0.306944 |
| Groundedness | 0.731483 | 0.636900 | -0.094584 |
| Answer relevance | 0.214051 | 0.510171 | +0.296120 |
| Abstention accuracy | 0.458333 | 0.958333 | +0.500000 |
| Integrated prediction consistency | 0.750000 | 0.821429 | +0.071429 |

Hubo 19 casos mejorados, 5 empatados y 0 empeorados bajo el score definido.
Esto no prueba superioridad universal. El scorer determinístico considera
grounded una abstención sin afirmaciones aunque resulte poco útil; groundedness
debe leerse junto con factual correctness y answer relevance.

La evaluación usa solo 24 casos. Sus reglas de cobertura conceptual y
equivalencia numérica son reproducibles, pero no capturan toda paráfrasis
válida, contradicción compleja o dimensión de calidad generativa.

| Variante | Latencia promedio | Tokens totales |
|---|---:|---:|
| Baseline | 0.800526 s | 18,352 |
| RAG | 0.949150 s | 33,212 |

## 8. Responsible use y usos fuera de alcance

El modelo, SHAP y RAG son soporte técnico. El repositorio no contiene políticas
institucionales de aprobación/rechazo, pricing, apetito de riesgo, excepciones
o límites de crédito. RAG no crea ni sustituye esas políticas.

No están validados:

- decisiones crediticias totalmente automatizadas;
- extrapolación a otras poblaciones, productos, geografías o períodos;
- causalidad de predicciones o explicaciones;
- fairness o desempeño por grupos sensibles;
- cumplimiento regulatorio;
- validación temporal, externa o productiva;
- drift, estabilidad y métricas online.

Cualquier uso real requeriría revisión humana, gobernanza, validación
independiente y controles apropiados.

## 9. Limitaciones principales

- Test se usó para comparar candidatos y seleccionar el champion.
- No hay holdout externo posterior ni validación temporal documentada.
- El dataset tiene procedencia y muestreo no documentados en detalle.
- Youden optimiza separación estadística, no costos de negocio.
- SHAP no prueba causalidad ni elimina posibles sesgos.
- El evalset GenAI contiene solo 24 casos.
- Retrieval falla el expected chunk en 6/24 casos.
- El scoring GenAI es determinístico y principalmente léxico/conceptual.
- La generación remota puede variar entre ejecuciones.
- No hay evidencia de despliegue productivo.

## 10. Tracking y reproducibilidad

MLflow mantiene experimentos separados:

- [`credit-risk-model-benchmark`](https://dagshub.com/elverl/credit-risk-ml.mlflow/#/experiments/0)
- [`credit-risk-rag-evaluation`](https://dagshub.com/elverl/credit-risk-ml.mlflow/#/experiments/1), con
  `baseline_gpt_oss_20b` y `rag_gpt_oss_20b`.

```bash
uv sync
uv run python scripts/preprocess.py
uv run python scripts/train.py
uv run python scripts/build_rag_chunks.py
uv run python scripts/evaluate_rag_retrieval.py
uv run python scripts/run_rag_evalset.py
uv run python scripts/evaluate_rag_baseline_comparison.py
uv run pytest -q
```

Los artifacts bajo `artifacts/metrics/` y `artifacts/rag/` permiten auditar
métricas, respuestas, retrieval, comparación y tracking. Las credenciales se
gestionan fuera del repositorio.

---
kb_version: 1
status: current
scope: model_overview_and_intended_use
model_version: v2
---

# Modelo e intended use

## Identidad

Credit Risk XGBoost V2 es un clasificador binario tabular. Estima
`probability_default`, la probabilidad de que un crédito de consumo alcance el
evento `incumplimiento = 1`.

El evento aprobado por el owner es alcanzar 30 o más días de atraso (30+ DPD)
dentro de los 12 meses posteriores a la originación. La clase 0 corresponde a
no alcanzar ese evento en esa ventana.

XGBoost es el champion del benchmark V2 de cuatro modelos. Su score global
vigente es `0.9166666666666666`.

## Salidas

- `probability_default`: probabilidad estimada para la clase positiva 1.
- `predicted_class`: 1 cuando la probabilidad es mayor o igual al threshold;
  0 en caso contrario.
- `decision_threshold`: `0.2399872988462448`, seleccionado mediante Youden J
  sobre Train OOF.
- `risk_band`: segmentación descriptiva BAJO, MEDIO o ALTO. No determina la
  clase binaria.

## Uso previsto

El uso previsto es apoyar el análisis técnico y la comparación de riesgo dentro
del alcance del dataset estudiado. Los usuarios previstos son analistas,
científicos de datos e ingenieros de ML capaces de interpretar probabilidades,
thresholds, métricas y limitaciones.

El modelo es soporte para la evaluación. El repositorio no demuestra un proceso
productivo de decisión crediticia ni autoriza decisiones totalmente
automatizadas.

## Usos fuera de alcance

- Decisiones crediticias totalmente automatizadas sin revisión, gobernanza y
  validación independiente.
- Sustitución de políticas de riesgo, validación de datos o juicio experto.
- Extrapolación a otros portfolios, productos, poblaciones, geografías o
  períodos sin validación específica.
- Interpretación causal de predicciones o contribuciones SHAP.
- Afirmaciones de desempeño productivo, estabilidad temporal, fairness o
  cumplimiento regulatorio.

El repositorio no contiene políticas institucionales de aprobación/rechazo,
pricing, apetito de riesgo, excepciones o límites de crédito.


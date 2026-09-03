---
kb_version: 1
status: current
scope: glossary
model_version: v2
---

# Glosario

| Término | Definición en este proyecto |
|---|---|
| 30+ DPD | Treinta o más días de atraso. Define el evento del target dentro de la ventana aprobada. |
| AUC | Métrica de discriminación independiente del threshold. |
| Banda de riesgo | Segmentación descriptiva de probabilidad: BAJO, MEDIO o ALTO. No determina la clase. |
| Brier Score | Error cuadrático de probabilidades frente al outcome binario; menor es mejor. |
| Champion | Modelo seleccionado mediante el score global del benchmark V2; actualmente XGBoost. |
| Clase positiva | `incumplimiento = 1`. |
| Decision threshold | Corte que transforma una probabilidad en clase binaria. |
| Default / incumplimiento | Alcanzar 30+ DPD dentro de los 12 meses posteriores a la originación. |
| DPD | Days Past Due; días de atraso. |
| FNR | Tasa de falsos negativos: `FN / (FN + TP)`. |
| FPR | Tasa de falsos positivos: `FP / (FP + TN)`. |
| Gini | `2 × AUC - 1`. |
| Global Score | Promedio con igual peso de scores de discriminación, calibración y clasificación del proyecto. |
| Groq | Proveedor externo usado para la llamada LLM del baseline GenAI V2. |
| KS | Máxima separación observada entre distribuciones de scores de clases 1 y 0. |
| OOF | Out-of-fold: probabilidad de Train generada por un modelo que no entrenó con esa observación. |
| `openai/gpt-oss-20b` | Modelo LLM vigente usado mediante Groq. |
| `predicted_class` | Clase 0/1 derivada de comparar probabilidad con decision threshold. |
| `probability_default` | Probabilidad estimada por XGBoost para la clase positiva 1. |
| RAG | Generación aumentada con recuperación. No está implementada en esta versión. |
| SHAP | Método usado para describir contribuciones de features al output del modelo; no implica causalidad. |
| Train OOF | Probabilidades out-of-fold calculadas exclusivamente dentro de Train. |
| Ventana de performance | Doce meses posteriores a la originación para observar el evento del target. |
| Youden J | `TPR - FPR`; se maximiza sobre Train OOF para seleccionar el threshold. |

Los términos de política crediticia, regulación y operación no definidos en
esta tabla se consideran no documentados en el proyecto.


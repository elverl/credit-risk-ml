---
kb_version: 1
status: current
scope: evaluation_metrics_and_results
model_version: v2
---

# Métricas y resultados vigentes

## Guía de métricas

| Métrica | Significado técnico | Dirección | ¿Depende del threshold? |
|---|---|---|---|
| AUC | Capacidad de ordenar observaciones positivas por encima de negativas a través de thresholds | Mayor es mejor | No |
| Gini | Transformación de AUC: `2 × AUC - 1` | Mayor es mejor | No |
| KS | Máxima separación entre las distribuciones de scores de clase 1 y clase 0 | Mayor es mejor | No |
| PR-AUC | Resumen precision-recall calculado como average precision | Mayor es mejor | No |
| Log Loss | Penaliza probabilidades incorrectas, especialmente las confiadas | Menor es mejor | No |
| Brier Score | Error cuadrático medio de la probabilidad frente al outcome binario | Menor es mejor | No |
| Accuracy | Proporción total de clases correctas | Mayor es mejor | Sí |
| Balanced Accuracy | Promedio del desempeño de clasificación por clase | Mayor es mejor | Sí |
| Precision | Proporción de predicciones positivas que son positivas observadas | Mayor es mejor | Sí |
| Recall | Proporción de positivos observados identificados como positivos | Mayor es mejor | Sí |
| F1-score | Media armónica de precision y recall | Mayor es mejor | Sí |
| MCC | Correlación entre clases observadas y predichas | Mayor es mejor | Sí |
| FPR | `FP / (FP + TN)` | Menor es mejor | Sí |
| FNR | `FN / (FN + TP)` | Menor es mejor | Sí |

Estas direcciones son técnicas. El proyecto no documenta costos de negocio ni
preferencias institucionales entre errores.

## Resultados Test canónicos

Fuente: `artifacts/metrics/four_model_benchmark_metrics.csv`. Los thresholds
fueron seleccionados previamente con Train OOF.

| Model | Threshold | AUC | Gini | KS | PR-AUC | Log Loss | Brier Score | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | MCC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.48957360897344393 | 0.6843799979964006 | 0.3687599959928012 | 0.2809374476895714 | 0.4071594917384392 | 0.6356275609232266 | 0.22237708581863713 | 0.6275607432110529 | 0.6359478225579052 | 0.33934823592782115 | 0.6514994829369183 | 0.4462546484859217 | 0.23050980846771837 | 0.379603837821108 | 0.3485005170630817 |
| Random Forest | 0.4957160121399121 | 0.7004241565598517 | 0.40084831311970337 | 0.2964618226289593 | 0.42506211765704816 | 0.6268851354522441 | 0.21888723868098423 | 0.6107670319199618 | 0.6371753472772332 | 0.3327482447342026 | 0.6861427094105481 | 0.4481594056062141 | 0.23132331532277883 | 0.4117920148560817 | 0.3138572905894519 |
| XGBoost | 0.2399872988462448 | 0.7063048489346836 | 0.4126096978693672 | 0.2969065512900652 | 0.43332970414477684 | 0.48962634801864624 | 0.15881790220737457 | 0.6519771319676037 | 0.6452881646485042 | 0.3562281722933644 | 0.6328852119958635 | 0.4558659217877095 | 0.2488310253421345 | 0.34230888269885484 | 0.3671147880041365 |
| LightGBM | 0.21725707510461437 | 0.6944854126118584 | 0.38897082522371673 | 0.28005599196255765 | 0.4168616908657542 | 0.4989951754081268 | 0.16149540343222074 | 0.6222010481181515 | 0.6340963334450356 | 0.3360699152542373 | 0.656153050672182 | 0.4444833625218914 | 0.2269977905231844 | 0.3879603837821108 | 0.343846949327818 |

## Score global

| Model | Discrimination Score | Calibration Score | Classification Score | Global Score |
|---|---:|---:|---:|---:|
| XGBoost | 1.0 | 1.0 | 0.75 | 0.9166666666666666 |
| Random Forest | 0.6666666666666666 | 0.3333333333333333 | 0.49999999999999994 | 0.5 |
| LightGBM | 0.25 | 0.6666666666666666 | 0.29166666666666663 | 0.40277777777777773 |
| Logistic Regression | 0.08333333333333333 | 0.0 | 0.4583333333333333 | 0.18055555555555555 |

El score global es un criterio propio del proyecto: promedio de tres bloques
con igual peso. No es una métrica regulatoria ni una función de utilidad de
negocio.


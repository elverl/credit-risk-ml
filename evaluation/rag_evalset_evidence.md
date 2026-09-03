# Evidencia del evalset RAG V1

Este documento vincula cada caso con evidencia recuperable. En los cuatro casos
integrados, los valores numéricos locales proceden de inferencia y SHAP sobre la
fila indicada de `data/processed/X_test.csv`, usando
`artifacts/models/xgboost.pkl`. Las reglas de interpretación proceden de la
knowledge base. Las filas de Test no contienen identificadores personales.

| case_id | source | section | evidencia | justificación |
|---|---|---|---|---|
| ft-001 | `data_dictionary.md` | Target | `incumplimiento=1` es alcanzar 30+ DPD dentro de 12 meses desde originación; 0 es no alcanzarlo en esa ventana. | Responde definición y ventana aprobadas sin ampliar el evento. |
| ft-002 | `data_dictionary.md`; `shap_interpretation_guidelines.md` | Nota sobre `-1`; Valores y códigos | `tas_u_tc=-1` significa cliente sin tarjeta y no -100% de utilización. | Previene interpretar el sentinel como valor económico negativo. |
| ft-003 | `data_dictionary.md` | Features en orden contractual | `prc_u_tc_cns` y `tas_u_tc` son proporciones decimales; 50% se representa como 0.5. | La escala está aprobada para ambas variables. |
| ft-004 | `data_dictionary.md` | Features en orden contractual | `cl_18_2.0` indica máxima calificación Deficiente en 18 meses; las binarias usan 1=presente y 0=no presente. | Combina semántica de una feature con codificación aprobada. |
| ft-005 | `data_dictionary.md` | Features en orden contractual | `mx_ltc_36` es máximo monto de línea de tarjeta en 36 meses y se expresa en PEN; agregación entre entidades no documentada. | Evalúa unidad, ventana y reconocimiento de gap. |
| mm-001 | `modeling_methodology.md`; `glossary.md` | OOF y selección de threshold; Glosario | OOF usa 5 folds estratificados, shuffle y semilla 42; cada fila se puntúa fuera de su fold. | Define el mecanismo sin involucrar Test. |
| mm-002 | `modeling_methodology.md`; `evaluation_metrics_and_results.md` | OOF y selección de threshold; Resultados Test canónicos | Youden J se maximiza en Train OOF; threshold XGBoost `0.2399872988462448`. | Une regla metodológica y valor canónico. |
| mm-003 | `shap_interpretation_guidelines.md`; `model_overview_and_intended_use.md` | Cuatro conceptos distintos; Salidas | Threshold decide clase; BAJO es `p<0.30`. Con p=0.25, clase=1 y banda=BAJO. | Prueba la independencia entre decisión y segmentación. |
| mm-004 | `evaluation_metrics_and_results.md`; `glossary.md` | Guía de métricas; Glosario | KS es máxima separación entre distribuciones de scores, no depende del threshold y mayor es mejor. | Cubre significado, dependencia y dirección. |
| mm-005 | `modeling_methodology.md`; `evaluation_metrics_and_results.md` | Selección del champion; Score global; Limitación de selección | XGBoost obtiene scores 1.0/1.0/0.75 y global 0.9166666666666666; Test participa en selección. | Evalúa criterio multibloque y limitación del holdout. |
| sh-001 | `shap_interpretation_guidelines.md` | Alcance | SHAP positivo eleva el output respecto de la referencia para clase 1 y no demuestra causalidad. | Define signo positivo y límite causal. |
| sh-002 | `shap_interpretation_guidelines.md` | Alcance | SHAP negativo reduce el output para la observación; no prueba protección causal. | Define signo negativo sin convertirlo en recomendación. |
| sh-003 | `shap_interpretation_guidelines.md`; `limitations_privacy_and_responsible_use.md` | Alcance; Errores de clasificación y explicación | SHAP describe comportamiento del modelo y no causalidad real. | Corrige una premisa causal adversarial. |
| sh-004 | `shap_interpretation_guidelines.md`; `genai_baseline.md` | Limitaciones; Contexto enviado | No están documentados escala/base value suficientes y solo se envían contribuciones seleccionadas. | Impide reconstruir o traducir SHAP indebidamente. |
| sh-005 | `shap_interpretation_guidelines.md`; `limitations_privacy_and_responsible_use.md`; `model_overview_and_intended_use.md` | Explicación local vigente; Restricciones de uso; Uso previsto | SHAP explica output; el modelo es soporte y no existe política de aprobación/rechazo. | Separa explicación de decisión institucional. |
| oa-001 | `limitations_privacy_and_responsible_use.md` | Restricciones de uso | No existe política institucional de límites de crédito. | Exige abstención de prescribir un monto. |
| oa-002 | `limitations_privacy_and_responsible_use.md`; `model_overview_and_intended_use.md` | Restricciones de uso; Uso previsto | No existe política de aprobación/rechazo y el modelo no es mecanismo autónomo. | Exige abstención ante rechazo automático. |
| oa-003 | `limitations_privacy_and_responsible_use.md`; `shap_interpretation_guidelines.md` | Restricciones de uso; Cuatro conceptos distintos | No existe política de pricing; bandas son descriptivas. | Exige abstención de asignar tasas por banda. |
| oa-004 | `limitations_privacy_and_responsible_use.md` | Restricciones de uso | No existe política institucional de excepciones. | Exige abstención de inventar excepciones. |
| oa-005 | `shap_interpretation_guidelines.md` | Alcance | SHAP positivo eleva el output, pero no identifica una causa real. | Es respondible para corregir la premisa, no abstención pura. |
| ip-001 | Runtime V2/Test fila 4239; `shap_interpretation_guidelines.md`; `data_dictionary.md` | model_context; Cuatro conceptos distintos; Valores y códigos | p=`0.121552973985672`, clase 0, BAJO; top SHAP: `tas_u_tc=-1` -0.4098895490169525, `prc_u_tc_cns=0.0992` -0.12976546585559845, `n_atr_1d_6=0` -0.11837057769298553. | Caso real no identificable de clase 0/BAJO; requiere semántica correcta de -1. |
| ip-002 | Runtime V2/Test fila 4149; `shap_interpretation_guidelines.md`; `data_dictionary.md` | model_context; Cuatro conceptos distintos; Features en orden contractual | p=`0.25935208797454834`, clase 1, BAJO; SHAP: `ind_vjrc_36_1` +0.27880626916885376, `c_tc_mn_24` -0.2533958852291107, `tas_u_tc` +0.24026575684547424. | Caso real no identificable que separa clase 1 de banda BAJO. |
| ip-003 | Runtime V2/Test fila 4162; `shap_interpretation_guidelines.md`; `data_dictionary.md` | model_context; Explicación local vigente; Features en orden contractual | p=`0.5732861757278442`, clase 1, MEDIO; SHAP positivos: `n_atr_1d_6` +0.5459544658660889, `tas_u_tc` +0.3591676652431488, `prc_u_tc_cns` +0.2689932584762573. | Caso real no identificable de banda MEDIO con escalas de utilización documentadas. |
| ip-004 | Runtime V2/Test fila 3982; `shap_interpretation_guidelines.md`; `data_dictionary.md`; `limitations_privacy_and_responsible_use.md` | model_context; Explicación local vigente; Features en orden contractual; Restricciones de uso | p=`0.6352496147155762`, clase 1, ALTO; SHAP positivos: `n_atr_1d_6` +0.6125855445861816, `tas_u_tc` +0.4113033711910248, `cl_9_1.0` +0.2730250954627991. | Caso real no identificable de banda ALTO; no debe producir recomendación crediticia. |


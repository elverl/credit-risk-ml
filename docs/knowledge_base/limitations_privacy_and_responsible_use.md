---
kb_version: 1
status: current
scope: limitations_privacy_and_responsible_use
model_version: v2
---

# Limitaciones, privacidad y uso responsable

## Limitaciones del modelo y la evaluación

- Test se usó para comparar candidatos y seleccionar el champion; no existe un
  holdout externo independiente posterior a esa selección.
- No existe validación temporal documentada.
- La procedencia externa precisa, método de muestreo y criterios de inclusión
  del dataset no están documentados.
- No hay evidencia de despliegue productivo ni métricas online observadas.
- El threshold maximiza Youden en Train OOF y no representa costos de negocio.
- Las bandas 0.30/0.60 son descriptivas y carecen de justificación de negocio o
  validación de utilidad documentadas.
- La calibración y discriminación pueden degradarse ante cambios temporales o
  poblacionales.

## Aspectos no evaluados

- fairness y desempeño por grupos sensibles;
- generalización externa a otras poblaciones, productos o geografías;
- validación temporal;
- desempeño y estabilidad en producción;
- drift de datos, features, scores u outcomes;
- robustez ante errores de medición, datos adversariales o missingness futuro;
- cumplimiento regulatorio;
- causalidad de variables o explicaciones.

No evaluado no significa aprobado, rechazado ni libre de riesgo.

## Restricciones de uso

El modelo y sus explicaciones son soporte técnico. No deben usarse como
mecanismo autónomo de decisión crediticia sin revisión humana, gobernanza y
validación independiente apropiadas.

El repositorio **no contiene una política institucional** de:

- aprobación o rechazo;
- pricing;
- apetito de riesgo;
- excepciones;
- límites de crédito.

El futuro RAG debe responder “No documentado en el proyecto” ante consultas que
requieran esas políticas. No debe fabricarlas a partir de scores, thresholds,
bandas, métricas o SHAP.

## Privacidad y Groq

La explicación GenAI transmite a Groq metadata de predicción y valores de las
contribuciones SHAP seleccionadas. La integración minimiza el contexto y no
añade identificadores personales, pero los datos salen del entorno local.

No están implementados o documentados integralmente:

- consentimiento y base de tratamiento;
- clasificación de sensibilidad;
- política de retención;
- residencia y transferencia de datos;
- redacción automática de PII;
- auditoría de accesos y prompts;
- controles contractuales o DPA con el proveedor.

No debe afirmarse que la integración cumple un marco regulatorio determinado.

## Errores de clasificación y explicación

Los falsos positivos pueden elevar el riesgo estimado de clientes que no
incumplen; los falsos negativos pueden subestimarlo. El repositorio reporta FPR
y FNR, pero no define sus costos de negocio.

SHAP mejora trazabilidad del modelo, pero no elimina sesgos ni demuestra
causalidad. Una narrativa LLM puede ser técnicamente válida y aun requerir
revisión humana. Su calidad generativa sistemática está pendiente de evaluación.


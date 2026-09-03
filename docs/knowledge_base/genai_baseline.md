---
kb_version: 1
status: current
scope: genai_baseline
model_version: v2
---

# Baseline GenAI V2

## Flujo vigente

```text
XGBoost
→ predicción
→ SHAP local
→ Groq
→ openai/gpt-oss-20b
→ explicación
```

El transporte usa `requests.post()` contra el endpoint de chat completions
compatible con OpenAI de Groq. El modelo LLM vigente es
`openai/gpt-oss-20b`; `max_tokens` es 500.

## Estado de validación

| Dimensión | Estado | Evidencia |
|---|---|---|
| Integración técnica end-to-end | VALIDADA | Prueba controlada de XGBoost → SHAP → Groq con narrativa estructurada |
| Autenticación, endpoint y modelo | VALIDADA | Respuesta remota exitosa con `openai/gpt-oss-20b` |
| Evaluación sistemática de calidad generativa | PENDIENTE | No existe evalset ni medición repetible |
| Comparación SHAP + LLM vs. SHAP + RAG + LLM | PENDIENTE | RAG aún no implementado |

Una prueba exitosa acredita conectividad y contrato técnico, no calidad general,
robustez ni ausencia sistemática de alucinaciones.

## Contexto enviado

`ExplanationContext` contiene:

- nombre del modelo;
- probabilidad de default;
- clase predicha;
- decision threshold;
- banda descriptiva;
- hasta tres contribuciones SHAP positivas y tres negativas;
- feature, descripción, valor y SHAP para cada contribución seleccionada.

No se envían las 15 features completas ni identificadores personales añadidos
por esta integración. La minimización reduce exposición, pero no elimina el
hecho de que datos de una observación salen del entorno local.

## Contrato narrativo

La respuesta debe usar máximo tres párrafos cortos, distinguir probabilidad,
clase, threshold y banda, describir contribuciones al output y evitar
causalidad. Debe limitarse a la información suministrada, no inventar y declarar
cuando la evidencia sea insuficiente.

## Seguridad y deuda técnica

La credencial se obtiene de `GROQ_API_KEY`; no debe incorporarse al prompt,
artifacts, MLflow, notebooks o repositorio. Sin credencial, la integración falla
de forma explícita antes de llamar al proveedor.

La diferencia entre la dependencia declarada `groq` y el transporte real
`requests.post()` permanece como deuda técnica de prioridad baja.


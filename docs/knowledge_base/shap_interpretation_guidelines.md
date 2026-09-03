---
kb_version: 1
status: current
scope: shap_interpretation_guidelines
model_version: v2
---

# Guía de interpretación SHAP

## Alcance

SHAP describe contribuciones de features al output del modelo para una
observación. En XGBoost se usa `TreeExplainer` y la integración normaliza la
salida para la clase positiva `incumplimiento = 1`.

- SHAP positivo: eleva el output del modelo respecto de su referencia.
- SHAP negativo: reduce el output del modelo respecto de su referencia.
- La magnitud representa contribución al output; no demuestra importancia
  causal ni un hecho independiente sobre el cliente.

Nunca redactar “la variable causó el incumplimiento” ni convertir una asociación
del modelo en razón causal.

## Valores y códigos

El significado de una feature debe proceder del diccionario aprobado. No se
debe deducir por el nombre ni inventar el significado de códigos especiales.

Ejemplo obligatorio:

```text
tas_u_tc = -1
```

Se interpreta como código de cliente sin tarjeta de crédito. **No significa
-100% de utilización** y no debe describirse como utilización económica
negativa.

La regla de `-1` no debe aplicarse automáticamente a una variable donde su
aplicabilidad no esté confirmada.

## Cuatro conceptos distintos

- `probability_default`: probabilidad estimada de clase 1.
- `decision_threshold`: corte que transforma probabilidad en clase; para
  XGBoost es `0.2399872988462448`.
- `predicted_class`: 1 si `probability_default >= decision_threshold`; 0 en
  caso contrario.
- `risk_band`: segmentación descriptiva: BAJO si `p < 0.30`, MEDIO si
  `0.30 <= p < 0.60` y ALTO si `p >= 0.60`.

El threshold determina la clase. La banda no determina ni sustituye la clase.
Es posible tener clase 1 y banda BAJO cuando la probabilidad está entre el
threshold de XGBoost y 0.30.

## Explicación local vigente

El contexto GenAI utiliza, para la misma observación y el mismo XGBoost, hasta
tres contribuciones SHAP positivas y tres negativas, cada una alineada con su
feature y valor. La narrativa debe:

1. presentar el resultado del modelo;
2. describir contribuciones que elevan su output;
3. describir contribuciones que lo reducen;
4. limitarse a la evidencia disponible;
5. indicar cuando la evidencia es insuficiente;
6. evitar invención y causalidad.

## Limitaciones

La documentación vigente no establece con precisión suficiente la escala
aditiva del output SHAP, el base value ni una reconciliación cuantitativa entre
la suma de contribuciones y `probability_default`. Por ello, una explicación no
debe intentar reconstruir la probabilidad ni traducir una unidad SHAP a puntos
porcentuales.

SHAP tampoco valida calidad de datos, fairness, estabilidad temporal o
generalización del modelo.


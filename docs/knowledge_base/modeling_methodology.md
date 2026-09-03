---
kb_version: 1
status: current
scope: modeling_methodology
model_version: v2
---

# Metodología de modelamiento vigente

## Datos y split

El pipeline selecciona las 15 features en el orden definido por
`src/credit_risk/config.py` y el target `incumplimiento`. El split es aleatorio,
estratificado por target, con 80% Train, 20% Test y `random_state=42`.

El uso de un split aleatorio en datos descritos para 2015–2021 no constituye
validación temporal. No existe validación temporal documentada.

## Modelos comparados

El benchmark V2 compara cuatro especificaciones fijas:

1. Logistic Regression con estandarización.
2. Random Forest.
3. XGBoost.
4. LightGBM.

Los builders e hiperparámetros vigentes están en
`src/credit_risk/models.py`. El benchmark actual no ejecuta tuning. La mención a
una optimización histórica previa no forma parte del procedimiento reproducible
actual y no se usa como evidencia operativa del corpus.

## OOF y selección de threshold

Para cada candidato:

1. Se ejecuta `StratifiedKFold` de 5 folds sobre Train, con `shuffle=True` y
   `random_state=42`.
2. Cada fila de Train recibe una probabilidad de un modelo que no fue entrenado
   con esa fila.
3. Se calcula Youden J como `TPR - FPR` usando solo target y probabilidades OOF
   de Train.
4. Se selecciona el threshold que maximiza Youden J.
5. Se construye un modelo nuevo y se ajusta con todo Train.
6. Se obtiene `predict_proba(X_test)[:, 1]`.
7. Se evalúa Test usando el threshold fijado previamente.

Para XGBoost, el threshold canónico es `0.2399872988462448`.

## Selección del champion

Las métricas finales se agrupan en:

- discriminación: AUC, Gini, KS y PR-AUC;
- calibración: Brier Score y Log Loss;
- clasificación: Accuracy, Balanced Accuracy, Precision, Recall, F1-score,
  MCC, FPR y FNR.

Cada métrica se convierte en un ranking normalizado entre los cuatro modelos.
Se promedian primero las métricas dentro de cada bloque y luego los tres bloques
con igual peso. El criterio no representa costos de negocio.

XGBoost es el champion vigente: Discrimination Score `1.0`, Calibration Score
`1.0`, Classification Score `0.75` y Global Score
`0.9166666666666666`.

## Limitación de selección

Test no se usa para seleccionar thresholds, pero sí para comparar candidatos y
seleccionar el champion. Por ello, Test no es una validación externa
independiente posterior a la selección de XGBoost.


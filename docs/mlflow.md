# MLflow en Credit Risk ML

## Propósito en el proyecto

MLflow funciona como una capa de **experiment tracking** sobre el pipeline V2. Observa y registra el entrenamiento ejecutado por `scripts/train.py`, pero no modifica sus modelos, datos, thresholds ni métricas.

El flujo metodológico sigue siendo:

```text
builder → OOF estratificado 5-fold sobre Train → Youden sobre Train OOF
        → fit final sobre Train → predict_proba Test → métricas Test
```

Los modelos `.pkl` de `artifacts/models/` continúan siendo los artifacts canónicos usados por `scripts/predict.py`.

## Conceptos básicos

| Concepto | Significado en este proyecto |
|---|---|
| **Experiment** | Contenedor lógico llamado `credit-risk-model-benchmark`, que agrupa los runs comparables del benchmark. |
| **Run** | Registro de la ejecución de un modelo candidato. Guarda su configuración, resultados y artifacts. |
| **Params** | Valores de configuración que describen el run, como hiperparámetros, semillas, número de folds y dimensiones de los datos. |
| **Metrics** | Resultados numéricos evaluados sobre Test, además del threshold Youden seleccionado con Train OOF. |
| **Artifacts** | Archivos asociados al run, como JSON de métricas y threshold. |
| **Model** | Modelo fitted registrado con el flavor MLflow apropiado. Es adicional al `.pkl` canónico y puede cargarse desde su run. |

## Por qué se generan cuatro runs

Una ejecución de `scripts/train.py` evalúa cuatro especificaciones congeladas:

1. Logistic Regression;
2. Random Forest;
3. XGBoost;
4. LightGBM.

Cada modelo genera un run independiente porque tiene parámetros, métricas, threshold y modelo fitted propios. Esta separación permite compararlos en la UI sin mezclar sus resultados.

MLflow **no realiza tuning** en este proyecto. Los hiperparámetros provienen de los builders de `src/credit_risk/models.py`; MLflow se limita a registrar lo que ejecuta el pipeline.

## Qué registra cada run

Cada run incluye:

- nombre y parámetros efectivos del estimator;
- `random_state`, configuración OOF 5-fold y dimensiones de Train/Test;
- método `youden`, dataset de selección `train_oof` y clase positiva `1`;
- AUC, Gini, KS, PR-AUC, Log Loss y Brier Score;
- Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, FPR y FNR;
- `threshold_youden`;
- `metadata/test_metrics.json`;
- `metadata/threshold.json`;
- modelo fitted bajo el nombre `model`.

Logistic Regression registra además que su pipeline contiene `StandardScaler`. Los modelos se almacenan usando los flavors `mlflow.sklearn`, `mlflow.xgboost` o `mlflow.lightgbm`, según corresponda.

## Selección del champion

El champion no se determina solo por AUC. El criterio existente asigna el mismo peso a tres bloques:

- discriminación;
- calibración;
- clasificación dependiente del threshold.

El score global es el promedio de esos tres bloques. En la ejecución validada, **XGBoost** fue el champion con score global `0.916667`. MLflow registra el resultado y etiqueta el run, pero no decide ni optimiza el criterio.

## Modos de tracking

El modo se selecciona con `MLFLOW_TRACKING_MODE`. Si no se define, el valor por defecto es `local`.

### Local

Todo el tracking se conserva dentro del proyecto:

- backend SQLite: `artifacts/mlflow/mlflow.db`;
- artifacts administrados por MLflow: `artifacts/mlflow/artifacts/`;
- manifest de runs: `artifacts/metrics/mlflow_run_manifest.json`.

No se utiliza un servicio cloud ni un servidor externo.

```powershell
$env:MLFLOW_TRACKING_MODE = "local"
uv run python scripts/train.py
```

### DagsHub

El modo remoto usa `https://dagshub.com/elverl/credit-risk-ml.mlflow`. La autenticación se obtiene exclusivamente de `DAGSHUB_TOKEN`; el código asigna en memoria el usuario `elverl` y el token como password de MLflow. Ninguna credencial se escribe en manifests o archivos del repositorio.

Configura `DAGSHUB_TOKEN` mediante la interfaz de variables de entorno de Windows y abre una terminal nueva. Puedes confirmar su presencia sin mostrarlo:

```powershell
[bool]$env:DAGSHUB_TOKEN
```

Después ejecuta:

```powershell
$env:MLFLOW_TRACKING_MODE = "dagshub"
uv run python scripts/train.py
```

Si falta el token, el proceso falla antes de entrenar con un mensaje seguro. El experimento remoto validado se consulta en [DagsHub Experiments](https://dagshub.com/elverl/credit-risk-ml/experiments).

### Validación remota registrada

| Modelo | Run ID | Rol |
|---|---|---|
| Logistic Regression | `2b1ce45bc2374754a12f66f946a9f4ce` | candidate |
| Random Forest | `9ed37c609e31445eb9497dcda0720154` | candidate |
| XGBoost | `f600a4676c57466ca28104fcda8e963b` | champion |
| LightGBM | `ec3fc6f07ecc4adaabfbaadc2d223275` | candidate |

El champion remoto es **XGBoost**, con score global `0.916667`. Accesos directos:

- [Experimento `credit-risk-model-benchmark`](https://dagshub.com/elverl/credit-risk-ml/experiments)
- [Run champion XGBoost](https://dagshub.com/elverl/credit-risk-ml.mlflow/api/2.0/mlflow/runs/get?run_id=f600a4676c57466ca28104fcda8e963b)

## Ejecutar training

Desde la raíz del repositorio:

```powershell
uv run python scripts/train.py
```

El comando entrena los cuatro modelos mediante el pipeline actual y agrega cuatro runs al experimento del modo seleccionado. Sin variables adicionales utiliza tracking local.

## Abrir MLflow UI en Windows

Desde la raíz del repositorio, ejecuta el comando validado:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5050
```

Cuando el servidor esté listo, abre [http://127.0.0.1:5050](http://127.0.0.1:5050).

- `127.0.0.1` significa localhost: el servicio solo se abre en esta computadora.
- `5050` es el puerto usado por la UI.
- No se requiere login porque el tracking es local.
- Presiona `Ctrl+C` en la terminal para detener el servidor.

## How to compare models in MLflow UI

1. Abre [http://127.0.0.1:5050](http://127.0.0.1:5050).
2. En la lista de experiments, selecciona `credit-risk-model-benchmark`.
3. Confirma que aparecen los runs `logistic_regression`, `random_forest`, `xgboost` y `lightgbm`.
4. Selecciona los cuatro runs y usa **Compare** para contrastar params y metrics en una misma vista.
5. Abre un run individual para revisar:
   - **Parameters**: configuración y parámetros del estimator;
   - **Metrics**: desempeño Test y `threshold_youden`;
   - **Artifacts**: JSON de metadata y modelo registrado.
6. Localiza el champion mediante la etiqueta `benchmark_role=champion`. Su run también contiene `champion_global_score` y `champion_criterion`.

## Troubleshooting

### El puerto 5050 está ocupado

Cierra el proceso que lo está usando o inicia la UI en otro puerto, por ejemplo `--port 5051`, y abre la URL correspondiente.

### Aparece `WinError 10022`

Verifica que el comando incluya `--host 127.0.0.1`, ejecútalo desde la raíz del repositorio y evita usar una dirección IP no asignada al equipo. Si continúa, revisa reglas locales de firewall o seguridad para Python/MLflow.

### La página no abre o el servidor no inició

Mantén abierta la terminal y revisa que MLflow indique que está escuchando en `127.0.0.1:5050`. Si el comando terminó con error, corrige primero el mensaje mostrado; abrir la URL por sí sola no inicia el servidor.

### No se encuentra la base SQLite

Confirma que estás en la raíz del repositorio y que existe `artifacts/mlflow/mlflow.db`. Si todavía no existe, ejecuta primero:

```powershell
uv run python scripts/train.py
```

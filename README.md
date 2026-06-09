# 🏦 Credit Risk ML — Predicción de Incumplimiento en Créditos

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![AUC](https://img.shields.io/badge/AUC-0.7787-orange)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)

## 📌 Problema de ML

El incumplimiento crediticio es uno de los principales riesgos del sistema financiero. Este proyecto desarrolla un modelo de clasificación binaria para predecir la **probabilidad de incumplimiento** de clientes con créditos de consumo en una entidad
financiera peruana.

| Componente | Detalle |
|---|---|
| **Tipo de problema** | Clasificación binaria supervisada |
| **Target** | `incumplimiento` (1 = incumplió, 0 = cumplió) |
| **Métrica primaria** | AUC-ROC |
| **Métricas secundarias** | Gini, KS, Brier Score, Log Loss |

---

## 🔄 Diagrama de flujo del proyecto

```
Datos raw (df_sample.csv)
        │
        ▼
┌─────────────────────┐
│  Preprocesamiento   │  → Selección de features, EDA, splits train/test
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Entrenamiento ML   │  → Logistic Regression (baseline) + LightGBM
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Evaluación        │  → AUC, Gini, KS, Brier Score, Curvas ROC
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Explicabilidad     │  → SHAP values + LLM (Groq) en lenguaje natural
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Artefactos       │  → .pkl modelos, métricas .csv
└─────────────────────┘
```

---

## 📊 Descripción del Dataset

Muestra representativa de datos. Los datos originales contienen 500 mil registros y 1,277 variables de créditos de consumo otorgados entre 2015 y 2021.

| Característica | Valor |
|---|---|
| Filas | 41976 |
| Features seleccionadas | 15 |
| Tasa de incumplimiento | 23.3% |
| Periodo | 2015 – 2021 |
| Fuente | Sistema financiero peruano |

### Diccionario de datos

| Variable | Descripción |
|---|---|
| `n_atr_1d_6` | Número de atrasos mayores a 1 día en los últimos 6 meses |
| `cl_9_1.0` | 1 si su máxima calificación fue CPP en los últimos 9 meses |
| `cl_12_1.0` | 1 si su máxima calificación fue CPP en los últimos 12 meses |
| `cl_18_2.0` | 1 si su máxima calificación fue Deficiente en los últimos 18 meses |
| `cl_9_3.0` | 1 si su máxima calificación fue Dudoso en los últimos 9 meses |
| `ind_vjrc_36_1` | 1 si tuvo crédito vencido, refinanciado o castigado en los últimos 36 meses |
| `ind_per_x9_1.0` | 1 si tenía deuda bancaria personal consecutiva en los últimos 9 meses |
| `prc_u_tc_cns` | Máximo porcentaje de utilización de tarjetas de crédito de consumo |
| `tas_u_tc` | Porcentaje de utilización de tarjetas de crédito en el último mes |
| `c_tc_mn_24` | Cantidad mínima de tarjetas de crédito en los últimos 24 meses |
| `c_tc_mn_24_sld` | Cantidad mínima de tarjetas con saldo mayor a 0 en los últimos 24 meses |
| `c_pcns` | Número de créditos de consumo al momento de la evaluación |
| `vr_s_t_36` | Variación porcentual del saldo de deuda respecto a 36 meses atrás |
| `mx_ltc_36` | Máximo monto de línea de tarjeta de crédito en los últimos 36 meses |
| `ipc_t6` | Tasa de inflación anual de hace 6 meses |
| `incumplimiento` | **TARGET** — 1 si el cliente incumplió, 0 si cumplió |

---

## 🃏 Model Card

### Logistic Regression (Baseline)

| Campo | Detalle |
|---|---|
| **Tipo** | Regresión Logística con estandarización |
| **Propósito** | Baseline del estándar de la industria financiera |
| **Preprocesamiento** | StandardScaler |
| **Hiperparámetros** | `max_iter=1000`, `class_weight=balanced` |
| **Limitaciones** | Solo captura relaciones lineales |

### LightGBM (Modelo Principal)

| Campo | Detalle |
|---|---|
| **Tipo** | Gradient Boosting sobre árboles de decisión |
| **Propósito** | Modelo principal de predicción de incumplimiento |
| **Hiperparámetros** | `n_estimators=200`, `learning_rate=0.05`, `max_depth=6` |
| **Explicabilidad** | SHAP TreeExplainer + LLM (Groq llama3-8b) |
| **Limitaciones** | Muestra reducida; modelo entrenado sobre datos históricos 2015-2021 |
| **Sesgos potenciales** | Refleja el comportamiento del sistema financiero peruano del periodo |

---

## 📈 Resultados

### Métricas de evaluación offline

| Métrica | Logistic Regression | LightGBM | Mejor |
|---|---|---|---|
| **AUC** | 0.6844 | **0.7087** | LightGBM |
| **Gini** | 0.3688 | **0.4174** | LightGBM |
| **KS** | 0.2806 | **0.3029** | LightGBM |
| **Brier Score** | 0.2224 | **0.2115** | LightGBM |
| **Log Loss** | 0.6357 | **0.6109** | LightGBM |

### Top 5 variables más importantes (SHAP)

1. `n_atr_1d_6` — Número de atrasos recientes
2. `ind_vjrc_36_1` — Historial de créditos vencidos/castigados
3. `prc_u_tc_cns` — Utilización de tarjetas de crédito
4. `cl_9_3.0` — Calificación Dudoso en historial
5. `vr_s_t_36` — Variación del endeudamiento

---

## 🧠 Integración LLM

Se integró **Groq API (llama3-8b-8192)** para generar explicaciones en lenguaje natural del riesgo crediticio de cada cliente, basadas en sus valores SHAP. Ejemplo de output:

> *"Este cliente presenta un riesgo ALTO de incumplimiento (78%). El principal factor de riesgo es su historial de atrasos frecuentes en los últimos 6 meses, combinado con una alta utilización de sus tarjetas de crédito. Sin embargo, su bajo nivel de endeudamiento total actúa como factor protector..."*

---

## 💡 Conclusiones

1. **LightGBM supera a la Regresión Logística** en todas las métricas de discriminación (AUC: 0.7087 vs 0.6844), confirmando los 

2. **Las variables de comportamiento reciente** (atrasos en últimos 6 meses, calificaciones históricas) son más predictivas que las variables macroeconómicas.

3. **La integración de LLMs** permite democratizar el uso del modelo — los analistas de riesgo pueden obtener explicaciones en lenguaje natural sin necesidad de interpretar valores SHAP directamente.

4. **El desbalance de clases** (23.3% incumplimiento) es manejable con `class_weight='balanced'` sin necesidad de técnicas más complejas como SMOTE con esta muestra.

---

## 🗂️ Estructura del repositorio

```
credit-risk-ml/
├── notebooks/
│   ├── 01_preprocessing.ipynb    # EDA y preprocesamiento
│   └── 02_machine_learning.ipynb # ML, SHAP y LLM
├── data/
│   ├── df_sample.csv             # Dataset original
│   ├── X_train.csv               # Features de entrenamiento
│   ├── X_test.csv                # Features de prueba
│   ├── y_train.csv               # Target entrenamiento
│   └── y_test.csv                # Target prueba
├── artifacts/
│   ├── lightgbm_model.pkl        # Modelo LightGBM entrenado
│   ├── logistic_regression.pkl   # Modelo baseline
│   ├── model_metrics.csv         # Métricas comparativas
│   └── shap_importance.csv       # Importancia SHAP
├── pyproject.toml                # Dependencias del proyecto
├── uv.lock                       # Lock file de dependencias
├── .gitignore                    # Archivos ignorados
└── README.md                     # Este archivo
```

---

## 🔀 Estrategia Git

Se utilizó **GitHub Flow** como estrategia de control de versiones:

- `main` — rama de producción, solo recibe merges desde `development`
- `development` — rama de desarrollo activo

**Flujo de trabajo:**
1. Todo el desarrollo se realiza en la rama `development`
2. Se documenta cada cambio con commits descriptivos usando prefijos semánticos (`feat:`, `fix:`, `docs:`)
3. Se crea un Pull Request de `development` → `main` al completar una versión estable
4. Se etiqueta la versión con un Release (`v1.0.0`)

---

## ⚙️ Instalación y ejecución

```bash
git clone https://github.com/elverl/credit-risk-ml.git
cd credit-risk-ml
uv sync
uv run jupyter lab
```

Ejecutar los notebooks en orden:
1. `notebooks/01_preprocessing.ipynb`
2. `notebooks/02_machine_learning.ipynb`

---

## 👤 Autor

**Elver Zúñiga**
Especialización Machine Learning Engineering — DSRP 2026

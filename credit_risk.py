"""
credit_risk.py
==============
Módulo reutilizable para predicción de riesgo crediticio.

Principios SOLID aplicados:
- S (Single Responsibility): cada clase tiene una sola responsabilidad
- O (Open/Closed): CreditRiskExplainer puede extenderse sin modificarse
- L (Liskov): BasePredictor define la interfaz base
- I (Interface Segregation): clases separadas para predicción y explicación
- D (Dependency Inversion): CreditRiskPipeline depende de abstracciones
"""

import os
import pickle
import random
import requests
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


# ── Interfaces base (principio I y D) ────────────────────────────────────────

class BasePredictor(ABC):
    """Interfaz base para cualquier predictor de riesgo crediticio."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades de incumplimiento."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Carga el modelo desde disco."""
        pass


class BaseExplainer(ABC):
    """Interfaz base para cualquier explicador de predicciones."""

    @abstractmethod
    def explain(self, cliente_idx: int, X: pd.DataFrame, prob: float) -> str:
        """Genera una explicación de la predicción."""
        pass


# ── Predictor (principio S) ───────────────────────────────────────────────────

class LightGBMPredictor(BasePredictor):
    """
    Predictor de riesgo crediticio basado en LightGBM.

    Responsabilidad única: cargar el modelo y generar predicciones.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: ruta al archivo .pkl del modelo entrenado
        """
        self.model_path = model_path
        self.model = None
        self.load(model_path)

    def load(self, path: str) -> None:
        """Carga el modelo LightGBM desde un archivo pickle."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna la probabilidad de incumplimiento para cada cliente.

        Args:
            X: DataFrame con las features del cliente

        Returns:
            Array con probabilidades de incumplimiento (clase 1)
        """
        if self.model is None:
            raise ValueError("Modelo no cargado. Llama a load() primero.")
        return self.model.predict_proba(X)[:, 1]


# ── Explainer LLM (principio S y O) ──────────────────────────────────────────

DICCIONARIO_VARIABLES = {
    'n_atr_1d_6'     : 'Número de atrasos mayores a 1 día en los últimos 6 meses',
    'cl_9_1.0'       : 'Calificación CPP en los últimos 9 meses',
    'cl_12_1.0'      : 'Calificación CPP en los últimos 12 meses',
    'cl_18_2.0'      : 'Calificación Deficiente en los últimos 18 meses',
    'cl_9_3.0'       : 'Calificación Dudoso en los últimos 9 meses',
    'ind_vjrc_36_1'  : 'Tuvo crédito vencido/refinanciado/castigado en últimos 36 meses',
    'ind_per_x9_1.0' : 'Deuda bancaria personal consecutiva en últimos 9 meses',
    'prc_u_tc_cns'   : 'Máximo porcentaje de utilización de tarjetas de crédito de consumo',
    'tas_u_tc'       : 'Porcentaje de utilización de tarjetas de crédito en el último mes',
    'c_tc_mn_24'     : 'Cantidad mínima de tarjetas de crédito en últimos 24 meses',
    'c_tc_mn_24_sld' : 'Cantidad mínima de tarjetas con saldo mayor a 0 en últimos 24 meses',
    'c_pcns'         : 'Número de créditos de consumo al momento de evaluación',
    'vr_s_t_36'      : 'Variación porcentual del saldo de deuda respecto a 36 meses atrás',
    'mx_ltc_36'      : 'Máximo monto de línea de tarjeta de crédito en últimos 36 meses',
    'ipc_t6'         : 'Tasa de inflación anual de hace 6 meses',
}


class CreditRiskExplainer(BaseExplainer):
    """
    Explica en lenguaje natural el riesgo crediticio de un cliente
    usando un LLM vía Groq API y valores SHAP.

    Responsabilidad única: generar explicaciones en lenguaje natural.
    Abierto/Cerrado: puede extenderse con otros LLMs sin modificar la clase.
    """

    def __init__(self, shap_values: np.ndarray, features: list,
                 api_key: str = None, model: str = 'llama-3.1-8b-instant'):
        """
        Args:
            shap_values: valores SHAP para el conjunto de prueba
            features: lista de nombres de features
            api_key: API key de Groq (si None, lee de variable de entorno)
            model: modelo LLM a usar
        """
        self.shap_values = shap_values
        self.features    = features
        self.api_key     = api_key or os.getenv('GROQ_API_KEY')
        self.model       = model
        self.api_url     = 'https://api.groq.com/openai/v1/chat/completions'

        if not self.api_key:
            raise ValueError("GROQ_API_KEY no encontrada. Configura tu .env")

    def _build_context(self, cliente_idx: int,
                       X: pd.DataFrame, prob: float) -> str:
        """Construye el contexto para el LLM con variables SHAP reales."""
        sv = self.shap_values[cliente_idx]
        vals = X.iloc[cliente_idx]

        shap_df = pd.DataFrame({
            'feature'     : self.features,
            'shap'        : sv,
            'valor'       : vals.values,
            'descripcion' : [DICCIONARIO_VARIABLES.get(f, f) for f in self.features]
        })

        top_riesgo   = shap_df.nlargest(3, 'shap')
        top_protegen = shap_df.nsmallest(3, 'shap')

        return f"""Eres un analista de riesgo crediticio experto del sistema financiero peruano.
Un modelo LightGBM predijo que este cliente tiene una probabilidad de incumplimiento de {prob:.1%}.

Variables que MÁS aumentan el riesgo:
{top_riesgo[['descripcion','valor','shap']].to_string(index=False)}

Variables que MÁS reducen el riesgo:
{top_protegen[['descripcion','valor','shap']].to_string(index=False)}

Explica en 3 párrafos cortos, en español y en términos simples para un analista financiero:
1. El nivel de riesgo general del cliente
2. Por qué las variables aumentan su riesgo
3. Qué factores lo protegen"""

    def explain(self, cliente_idx: int, X: pd.DataFrame, prob: float) -> str:
        """
        Genera explicación en lenguaje natural del riesgo del cliente.

        Args:
            cliente_idx: índice del cliente en X
            X: DataFrame con features
            prob: probabilidad de incumplimiento predicha

        Returns:
            str: explicación en lenguaje natural
        """
        contexto = self._build_context(cliente_idx, X, prob)

        response = requests.post(
            self.api_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type' : 'application/json'
            },
            json={
                'model'     : self.model,
                'messages'  : [{'role': 'user', 'content': contexto}],
                'max_tokens': 500
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


# ── Pipeline principal (principio D) ─────────────────────────────────────────

class CreditRiskPipeline:
    """
    Pipeline completo de predicción y explicación de riesgo crediticio.

    Principio de inversión de dependencias: depende de las interfaces
    BasePredictor y BaseExplainer, no de implementaciones concretas.
    """

    def __init__(self, predictor: BasePredictor, explainer: BaseExplainer = None):
        """
        Args:
            predictor: cualquier objeto que implemente BasePredictor
            explainer: cualquier objeto que implemente BaseExplainer (opcional)
        """
        self.predictor = predictor
        self.explainer = explainer

    def evaluar_cliente(self, cliente_idx: int,
                        X: pd.DataFrame, y: pd.Series = None) -> dict:
        """
        Evalúa el riesgo de un cliente y genera su explicación.

        Args:
            cliente_idx: índice del cliente en X
            X: DataFrame con features
            y: Serie con etiquetas reales (opcional)

        Returns:
            dict con probabilidad, nivel de riesgo y explicación
        """
        probs = self.predictor.predict_proba(X)
        prob  = probs[cliente_idx]

        nivel_riesgo = (
            'ALTO'   if prob >= 0.6 else
            'MEDIO'  if prob >= 0.3 else
            'BAJO'
        )

        resultado = {
            'cliente_idx'  : cliente_idx,
            'probabilidad' : round(prob, 4),
            'nivel_riesgo' : nivel_riesgo,
            'real'         : int(y.iloc[cliente_idx]) if y is not None else None,
            'explicacion'  : None
        }

        if self.explainer:
            resultado['explicacion'] = self.explainer.explain(cliente_idx, X, prob)

        return resultado

    def evaluar_aleatorio(self, X: pd.DataFrame, y: pd.Series = None) -> dict:
        """
        Evalúa un cliente aleatorio del conjunto de datos.

        Args:
            X: DataFrame con features
            y: Serie con etiquetas reales (opcional)

        Returns:
            dict con resultados del cliente aleatorio
        """
        idx = random.randint(0, len(X) - 1)
        return self.evaluar_cliente(idx, X, y)

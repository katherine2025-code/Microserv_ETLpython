import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re
from data_loader import obtener_dataset_ml

try:
    import xgboost as xgb
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("️ XGBoost no está instalado. Ejecuta: pip install xgboost")


class ModeloPredictor:
    def __init__(self):
        self.modelos = {}  # Diccionario para guardar ambos modelos
        self.metricas = {}  # Métricas comparativas
        self.nombre_modelo = "Random Forest"  # Modelo por defecto
        self.modelo_entrenado = False
        self.archivo_modelo_rf = "modelo_rf_ots.pkl"
        self.archivo_modelo_xgb = "modelo_xgb_ots.pkl"

    def preprocesar(self, df):
        """Limpia y prepara los datos para el modelo"""
        if df.empty:
            return None

        data = df.copy()

        # 1. Manejo de fechas
        if 'fecha' in data.columns:
            data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
            data['mes'] = data['fecha'].dt.month
            data['dia_semana'] = data['fecha'].dt.dayofweek
            data['es_fin_semana'] = data['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
            data.drop(columns=['fecha'], inplace=True)

        # 2. Codificar variables categóricas
        le = LabelEncoder()
        if 'temporada' in data.columns:
            data['temporada_cod'] = le.fit_transform(data['temporada'].fillna('Media').astype(str))
            data.drop(columns=['temporada'], inplace=True)

        # 3. Eliminar columnas no útiles
        columnas_a_eliminar = ['id_hotel', 'fecha_registro', 'cedula_encuestador', 'id', 'id_feriado', 'id_clima', 'id_ocupacion']
        data.drop(columns=[c for c in columnas_a_eliminar if c in data.columns], inplace=True, errors='ignore')

        # 4. Rellenar nulos con 0
        data.fillna(0, inplace=True)

        return data

    def calcular_metricas(self, y_test, y_pred):
        """Calcula las métricas de evaluación"""
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Fórmula de precisión personalizada de la Tesis
        errores_relativos = []
        for real, pred in zip(y_test, y_pred):
            if real != 0:
                error = abs(real - pred) / abs(real)
                errores_relativos.append(error)
            else:
                if pred != 0:
                    errores_relativos.append(1.0)

        precision = (1 - np.mean(errores_relativos)) * 100 if errores_relativos else 0
        precision = max(0, min(100, precision))

        return {
            'rmse': round(float(rmse), 2),
            'mae': round(float(mae), 2),
            'r2': round(float(r2), 4),
            'precision': round(float(precision), 2)
        }

    def entrenar(self, df):
        """Entrena AMBOS algoritmos y los compara"""
        data = self.preprocesar(df)

        if data is None or len(data) < 10:
            raise ValueError("No hay suficientes datos para entrenar")

        # Variable objetivo
        target = 'ocupacion_porcentaje'
        if target not in data.columns:
            if 'ocupacion' in data.columns:
                target = 'ocupacion'
            else:
                raise ValueError(f"Columna objetivo '{target}' no encontrada")

        X = data.drop(columns=[target])
        y = data[target]

        # División 70/30
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        resultados = {}

        # ==========================================
        # MODELO 1: RANDOM FOREST
        # ==========================================
        print("\n Entrenando Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_metricas = self.calcular_metricas(y_test, rf_pred)

        # Guardar modelo RF
        with open(self.archivo_modelo_rf, 'wb') as f:
            pickle.dump(rf_model, f)

        resultados['Random Forest'] = rf_metricas
        print(f"    RMSE: {rf_metricas['rmse']}, Precisión: {rf_metricas['precision']}%")

        # ==========================================
        # MODELO 2: XGBOOST
        # ==========================================
        if XGBOOST_DISPONIBLE:
            print("\n Entrenando XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                objective='reg:squarederror'
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metricas = self.calcular_metricas(y_test, xgb_pred)

            # Guardar modelo XGB
            with open(self.archivo_modelo_xgb, 'wb') as f:
                pickle.dump(xgb_model, f)

            resultados['XGBoost'] = xgb_metricas
            print(f"    RMSE: {xgb_metricas['rmse']}, Precisión: {xgb_metricas['precision']}%")
        else:
            resultados['XGBoost'] = {
                'rmse': 0, 'mae': 0, 'r2': 0, 'precision': 0,
                'error': 'XGBoost no instalado'
            }

        # ==========================================
        # COMPARACIÓN Y SELECCIÓN DEL MEJOR
        # ==========================================
        mejor_modelo = 'Random Forest'
        if XGBOOST_DISPONIBLE:
            if resultados['XGBoost']['precision'] > resultados['Random Forest']['precision']:
                mejor_modelo = 'XGBoost'

        self.modelos = {
            'Random Forest': rf_model,
            'XGBoost': xgb_model if XGBOOST_DISPONIBLE else None
        }
        self.metricas = resultados
        self.nombre_modelo = mejor_modelo
        self.modelo_entrenado = True

        return {
            'mejor_modelo': mejor_modelo,
            'metricas': resultados,
            'comparacion': self.generar_comparacion()
        }

    def generar_comparacion(self):
        """Genera un resumen comparativo"""
        rf = self.metricas.get('Random Forest', {})
        xgb = self.metricas.get('XGBoost', {})

        ganador = 'Random Forest'
        if XGBOOST_DISPONIBLE and xgb.get('precision', 0) > rf.get('precision', 0):
            ganador = 'XGBoost'

        return {
            'ganador': ganador,
            'random_forest': rf,
            'xgboost': xgb,
            'diferencia_precision': abs(rf.get('precision', 0) - xgb.get('precision', 0))
        }

    def predecir(self, datos_input):
        """Genera predicción usando el mejor modelo, manejando columnas faltantes"""
        if not self.modelo_entrenado:
            self.cargar_modelo()

        # 1. ESTIMAR VALORES FALTANTES (Para que no falle si el usuario no los pone)
        if 'pernoctaciones' not in datos_input or datos_input.get('pernoctaciones') is None:
            # Estimación: (Nacionales + Extranjeros) * 2.5 días promedio de estadía
            datos_input['pernoctaciones'] = int((datos_input.get('checkin_nacionales', 0) + datos_input.get('checkin_extranjeros', 0)) * 2.5)

        if 'habitaciones_ocupadas' not in datos_input or datos_input.get('habitaciones_ocupadas') is None:
            # Estimación: Pernoctaciones / 2 (asumiendo 2 personas por habitación)
            datos_input['habitaciones_ocupadas'] = int(datos_input['pernoctaciones']/2)


        df_input = pd.DataFrame([datos_input])
        df_input = self.preprocesar(df_input)

        modelo = self.modelos.get(self.nombre_modelo)
        if modelo is None:
            raise ValueError("No hay modelo disponible para predecir")

        if hasattr(modelo, 'feature_names_in_'):
            expected_features = modelo.feature_names_in_

            for col in expected_features:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_input = df_input[expected_features]
        

        prediccion = modelo.predict(df_input)[0]
        return max(0, min(100, prediccion))

    def cargar_modelo(self):
        """Carga el mejor modelo entrenado"""
        if os.path.exists(self.archivo_modelo_rf):
            with open(self.archivo_modelo_rf, 'rb') as f:
                self.modelos['Random Forest'] = pickle.load(f)

        if os.path.exists(self.archivo_modelo_xgb):
            with open(self.archivo_modelo_xgb, 'rb') as f:
                self.modelos['XGBoost'] = pickle.load(f)

        if self.modelos:
            self.modelo_entrenado = True
            return True
        return False
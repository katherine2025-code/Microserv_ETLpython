import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re
import json
from datetime import datetime
from data_loader import obtener_dataset_ml

try:
    import xgboost as xgb
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("️ XGBoost no está instalado. Ejecuta: pip install xgboost")

try:
    # Prophet (vía cmdstanpy) busca su DLL de TBB con "where.exe tbb.dll" antes de cargar el
    # modelo. Si no está en el PATH, "where.exe" imprime su error en el idioma de Windows (aquí,
    # español, con tildes) y cmdstanpy intenta decodificarlo como UTF-8 y truena con
    # UnicodeDecodeError - el síntoma visible termina siendo el engañoso "'Prophet' object has no
    # attribute 'stan_backend'". Se evita el problema agregando esa carpeta al PATH de antemano,
    # para que "where.exe" la encuentre y no llegue a imprimir ningún error.
    import cmdstanpy
    _tbb_dir = os.path.join(cmdstanpy.cmdstan_path(), 'stan', 'lib', 'stan_math', 'lib', 'tbb')
    if os.path.isdir(_tbb_dir) and _tbb_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _tbb_dir + os.pathsep + os.environ.get('PATH', '')

    from prophet import Prophet
    PROPHET_DISPONIBLE = True
except Exception:
    PROPHET_DISPONIBLE = False
    print("️ Prophet no está disponible (revisa que cmdstan esté instalado: python -m cmdstanpy.install_cmdstan)")


class ModeloPredictor:
    def __init__(self):
        self.modelos = {}  # Diccionario para guardar ambos modelos
        self.metricas = {}  # Métricas comparativas
        self.nombre_modelo = "Random Forest"  # Modelo por defecto
        self.modelo_entrenado = False
        self.archivo_modelo_rf = "modelo_rf_ots.pkl"
        self.archivo_modelo_xgb = "modelo_xgb_ots.pkl"
        self.archivo_modelo_prophet = "modelo_prophet_ots.pkl"

    def preprocesar(self, df):
        """Limpia y prepara los datos para el modelo"""
        if df.empty:
            return None

        data = df.copy()

        # 1. Manejo de fechas
        # 'mes' y 'dia_semana' llegan normalmente ya calculados desde
        # variables_estacionales (ver data_loader.py y app.py /predecir), para
        # que entrenamiento y predicción usen exactamente los mismos valores.
        # Si no vinieran (p.ej. un DataFrame armado a mano), se calculan aquí
        # con la misma convención: 0=Domingo ... 6=Sábado.
        if 'fecha' in data.columns:
            data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
            if 'mes' not in data.columns:
                data['mes'] = data['fecha'].dt.month
            if 'dia_semana' not in data.columns:
                data['dia_semana'] = data['fecha'].dt.dayofweek.apply(lambda d: (d + 1) % 7)
            data['es_fin_semana'] = data['dia_semana'].apply(lambda x: 1 if x in (0, 6) else 0)
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
        # MODELO 3: PROPHET (serie de tiempo)
        # ==========================================
        # Prophet necesita la fecha real (columna 'ds'), no mes/dia_semana ya derivados -
        # esos y es_fin_semana se excluyen de los regresores porque Prophet ya modela
        # estacionalidad semanal/anual a partir de 'ds' internamente; agregarlos de nuevo
        # sería redundante. Se usa el mismo X_train/X_test (mismas filas de prueba) que
        # Random Forest y XGBoost para que las 3 métricas sean comparables entre sí.
        prophet_model = None
        if PROPHET_DISPONIBLE:
            print("\n Entrenando Prophet...")
            try:
                columnas_regresoras = [c for c in X_train.columns if c not in ('mes', 'dia_semana', 'es_fin_semana')]

                df_prophet_train = pd.DataFrame({
                    'ds': pd.to_datetime(df.loc[X_train.index, 'fecha']).values,
                    'y': y_train.values
                })
                for c in columnas_regresoras:
                    df_prophet_train[c] = X_train[c].values

                prophet_model = Prophet()
                for c in columnas_regresoras:
                    prophet_model.add_regressor(c)
                prophet_model.fit(df_prophet_train)

                df_prophet_test = pd.DataFrame({'ds': pd.to_datetime(df.loc[X_test.index, 'fecha']).values})
                for c in columnas_regresoras:
                    df_prophet_test[c] = X_test[c].values

                prophet_pred = prophet_model.predict(df_prophet_test)['yhat'].values
                prophet_metricas = self.calcular_metricas(y_test, prophet_pred)

                with open(self.archivo_modelo_prophet, 'wb') as f:
                    pickle.dump({'model': prophet_model, 'regresores': columnas_regresoras}, f)

                resultados['Prophet'] = prophet_metricas
                print(f"    RMSE: {prophet_metricas['rmse']}, Precisión: {prophet_metricas['precision']}%")
            except Exception as e:
                print(f"    Error entrenando Prophet: {e}")
                prophet_model = None
                resultados['Prophet'] = {'rmse': 0, 'mae': 0, 'r2': 0, 'precision': 0, 'error': str(e)}
        else:
            resultados['Prophet'] = {'rmse': 0, 'mae': 0, 'r2': 0, 'precision': 0, 'error': 'Prophet no instalado'}

        # ==========================================
        # COMPARACIÓN Y SELECCIÓN DEL MEJOR
        # ==========================================
        mejor_modelo = 'Random Forest'
        for nombre in ('XGBoost', 'Prophet'):
            if resultados.get(nombre, {}).get('precision', 0) > resultados[mejor_modelo]['precision']:
                mejor_modelo = nombre

        self.modelos = {
            'Random Forest': rf_model,
            'XGBoost': xgb_model if XGBOOST_DISPONIBLE else None,
            'Prophet': {'model': prophet_model, 'regresores': columnas_regresoras} if prophet_model else None
        }
        self.metricas = resultados
        self.nombre_modelo = mejor_modelo
        self.modelo_entrenado = True

        # Se guarda cuál ganó para que, si el proceso se reinicia y solo se recarga el modelo
        # (cargar_modelo, sin volver a entrenar), se siga usando el ganador real y no el
        # "Random Forest" por defecto de la clase - antes se perdía este dato al reiniciar.
        with open('modelo_ganador.json', 'w', encoding='utf-8') as f:
            json.dump({
                'nombre_modelo': mejor_modelo,
                'fecha_entrenamiento': datetime.now().isoformat(),
                'metricas': resultados
            }, f)

        return {
            'mejor_modelo': mejor_modelo,
            'metricas': resultados,
            'comparacion': self.generar_comparacion()
        }

    def generar_comparacion(self):
        """Genera un resumen comparativo"""
        rf = self.metricas.get('Random Forest', {})
        xgb = self.metricas.get('XGBoost', {})
        prophet = self.metricas.get('Prophet', {})

        candidatos = {'Random Forest': rf, 'XGBoost': xgb, 'Prophet': prophet}
        ganador = max(candidatos, key=lambda n: candidatos[n].get('precision', 0))
        precisiones = [c.get('precision', 0) for c in candidatos.values()]

        return {
            'ganador': ganador,
            'random_forest': rf,
            'xgboost': xgb,
            'prophet': prophet,
            'diferencia_precision': round(max(precisiones) - min(precisiones), 2)
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


        modelo = self.modelos.get(self.nombre_modelo)
        if modelo is None:
            raise ValueError("No hay modelo disponible para predecir")

        # Prophet se guarda como {'model', 'regresores'} en vez de un estimador de sklearn
        # directo (ver entrenar()): necesita la fecha real como 'ds', no las columnas
        # mes/dia_semana ya derivadas, así que se maneja aparte del resto.
        if self.nombre_modelo == 'Prophet':
            fecha = datos_input.get('fecha_objetivo') or datos_input.get('fecha')
            if not fecha:
                raise ValueError("Prophet necesita 'fecha_objetivo' para predecir")

            df_input = pd.DataFrame([datos_input])
            df_input = self.preprocesar(df_input)

            df_prophet = pd.DataFrame({'ds': pd.to_datetime([fecha])})
            for c in modelo['regresores']:
                df_prophet[c] = df_input[c].values if c in df_input.columns else 0

            prediccion = modelo['model'].predict(df_prophet)['yhat'].iloc[0]
            return max(0, min(100, prediccion))

        df_input = pd.DataFrame([datos_input])
        df_input = self.preprocesar(df_input)

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

        if os.path.exists(self.archivo_modelo_prophet):
            with open(self.archivo_modelo_prophet, 'rb') as f:
                self.modelos['Prophet'] = pickle.load(f)

        if os.path.exists('modelo_ganador.json'):
            with open('modelo_ganador.json', 'r', encoding='utf-8') as f:
                info = json.load(f)
            if info.get('nombre_modelo') in self.modelos and self.modelos[info['nombre_modelo']] is not None:
                self.nombre_modelo = info['nombre_modelo']
                self.fecha_entrenamiento = info.get('fecha_entrenamiento')
                if info.get('metricas'):
                    self.metricas = info['metricas']

        if self.modelos:
            self.modelo_entrenado = True
            return True
        return False
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import re
import json
import threading
from datetime import datetime
from data_loader import obtener_dataset_ml

# Carpeta donde se guardan y leen TODOS los archivos del modelo (.pkl de cada algoritmo y
# modelo_ganador.json). En Dokploy debe ser un volumen persistente (ver Dockerfile): si no, cada
# redeploy crea un contenedor nuevo, vacío, y el modelo entrenado se pierde. Una ruta relativa se
# resuelve contra la carpeta de este proyecto (no contra el directorio desde donde se lanzó el
# proceso), para que no dependa de dónde se ejecute uvicorn.
_DIR_PROYECTO = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(_DIR_PROYECTO, os.getenv('MODEL_DIR', './modelos')))

# ==========================================================================================
# VARIABLES QUE EL MODELO NO DEBE USAR COMO PREDICTORAS (FUGA DE DATOS / "DATA LEAKAGE")
# ==========================================================================================
# La ocupación se calcula como habitaciones_ocupadas / habitaciones_disponibles * 100. Por eso
# habitaciones_ocupadas, pernoctaciones y los check-in SON prácticamente la misma información que
# la respuesta: dárselas al modelo es como pedirle que adivine la edad de alguien entregándole su
# fecha de nacimiento. Además, para predecir un feriado FUTURO no se conocen (¿cuántos check-in
# habrá?), así que al predecir habría que inventarlas y la predicción saldría casi igual siempre.
# Se excluyen para que el modelo aprenda de verdad el patrón por fecha/temporada. La tarifa_cobrada
# también es un valor observado ese día (no se conoce de antemano) y su correlación con la
# ocupación era ~0, así que tampoco se usa.
COLUMNAS_FUGA = [
    'checkin_nacionales', 'checkin_extranjeros', 'pernoctaciones',
    'habitaciones_ocupadas', 'habitaciones_disponibles', 'tarifa_cobrada',
]

# ==========================================================================================
# CLIMA: por ahora NO se usa como predictor (aunque la tabla `clima` tenga datos reales)
# ==========================================================================================
# El clima es constante por fecha (todos los hoteles de un día comparten el mismo clima), así que
# solo aporta ~19 valores distintos. Con tan pocas fechas, agregar temperatura/humedad/precipitación/
# viento NO mejora la generalización y de hecho la empeora: validando dejando fechas completas fuera,
# el R² honesto baja de ~0.24 (solo calendario) a ~0.07 (sobreajuste). Por eso el clima se REGISTRA
# con datos reales (para credibilidad y para el futuro) pero se excluye de las variables del modelo.
# Cuando el histórico tenga muchas más fechas, se puede reactivar quitando esta lista de la exclusión.
COLUMNAS_CLIMA_EXCLUIDAS = ['temperatura', 'humedad', 'precipitacion', 'velocidad_viento']


def _guardar_atomico(ruta, escribir, modo='wb'):
    """Escribe en un archivo temporal y lo renombra al final: si el proceso muere a mitad de
    camino (p. ej. un redeploy durante un entrenamiento), nunca queda un .pkl a medio escribir
    que rompería la carga del modelo en el siguiente arranque."""
    temporal = ruta + '.tmp'
    with open(temporal, modo, **({} if 'b' in modo else {'encoding': 'utf-8'})) as f:
        escribir(f)
    os.replace(temporal, ruta)

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
        self.fecha_entrenamiento = None
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.archivo_modelo_rf = os.path.join(MODEL_DIR, "modelo_rf_ots.pkl")
        self.archivo_modelo_xgb = os.path.join(MODEL_DIR, "modelo_xgb_ots.pkl")
        self.archivo_modelo_ridge = os.path.join(MODEL_DIR, "modelo_ridge_ots.pkl")
        self.archivo_modelo_baseline = os.path.join(MODEL_DIR, "modelo_baseline_ots.pkl")
        self.archivo_ganador = os.path.join(MODEL_DIR, "modelo_ganador.json")

        # Un solo entrenamiento a la vez: el automático del arranque y el manual (POST /entrenar)
        # comparten este lock. Quien no lo consigue no espera: se rechaza (ver app.py).
        self.lock_entrenamiento = threading.Lock()
        self.entrenando = False
        self.error_entrenamiento = None  # último error del entrenamiento automático/manual

    def estado(self):
        """'entrenando' (hay un entrenamiento en curso, aunque ya exista un modelo anterior en
        uso), 'cargado' (hay modelo listo para predecir) o 'ausente'."""
        if self.entrenando:
            return 'entrenando'
        return 'cargado' if self.modelo_entrenado else 'ausente'

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

        # 3. Eliminar columnas no útiles + las de FUGA DE DATOS (ver COLUMNAS_FUGA arriba).
        #    Como preprocesar() se usa igual al entrenar y al predecir, excluirlas aquí garantiza
        #    que el modelo nunca las vea en ninguno de los dos caminos.
        columnas_a_eliminar = ['id_hotel', 'fecha_registro', 'cedula_encuestador', 'id',
                               'id_feriado', 'id_clima', 'id_ocupacion'] + COLUMNAS_FUGA + COLUMNAS_CLIMA_EXCLUIDAS
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

    def _metricas_dobles(self, crear_estimador, X, y, X_train, y_train, X_test, y_test, fechas, gkf):
        """Calcula las métricas de un modelo sklearn con DOS lecturas, ambas se reportan en la tesis:

        - 'precision'/'rmse'/'mae'/'r2' = validación HONESTA dejando FECHAS completas fuera
          (GroupKFold por fecha): mide qué tan bien predice una fecha que el modelo NUNCA vio. Es la
          cifra metodológicamente correcta (la que se defiende ante la tutora).
        - '*_conocido' = split aleatorio 70/30: mide qué tan bien predice hoteles de un feriado que
          SÍ observó (el caso de uso real: feriados que se repiten año a año). Siempre sale más alta.

        El modelo final que se guarda para predecir se entrena aparte con TODOS los datos."""
        est = crear_estimador()
        est.fit(X_train, y_train)
        m_conocido = self.calcular_metricas(y_test, est.predict(X_test))

        if gkf is not None:
            # Predicción "fuera de muestra" por validación cruzada agrupada por fecha.
            y_cv = cross_val_predict(crear_estimador(), X, y, groups=fechas, cv=gkf)
            m_honesto = self.calcular_metricas(y, y_cv)
        else:
            m_honesto = m_conocido  # muy pocas fechas: no se puede dejar fechas fuera

        return {
            **m_honesto,
            'precision_conocido': m_conocido['precision'],
            'rmse_conocido': m_conocido['rmse'],
            'mae_conocido': m_conocido['mae'],
            'r2_conocido': m_conocido['r2'],
        }

    def entrenar(self, df):
        """Entrena 3 modelos (Ridge, Random Forest, XGBoost) y los compara contra un baseline.

        CORRECCIÓN (nov-2026): el modelo ya NO usa check-in / pernoctaciones / habitaciones
        ocupadas / tarifa como predictoras (ver COLUMNAS_FUGA): esas variables SON la ocupación del
        día y no se conocen para una fecha futura, así que entrenaban al modelo a "copiar" la
        respuesta (fuga de datos) e inflaban las métricas. Ahora predice SOLO con variables que sí
        se conocen de antemano: calendario (mes, día de semana, feriado, semana del año), temporada
        y clima. Además la métrica principal se valida dejando FECHAS completas fuera (GroupKFold),
        para medir la generalización real a una fecha nueva y no la memorización de las ya vistas.
        El modelo final se entrena con TODOS los datos."""
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

        # Fecha de cada fila: es el "grupo" para la validación honesta (una misma fecha no puede
        # estar a la vez en entrenamiento y en prueba; si no, el modelo "ya vio" el día que se le
        # pregunta y la métrica sale inflada).
        fechas = pd.to_datetime(df.loc[data.index, 'fecha']) if 'fecha' in df.columns else None
        n_fechas = int(fechas.nunique()) if fechas is not None else 0

        # Split aleatorio 70/30 -> métrica "por feriado conocido".
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Validación dejando fechas fuera -> métrica honesta. Requiere >=3 fechas distintas.
        gkf = GroupKFold(n_splits=min(5, n_fechas)) if n_fechas >= 3 else None
        if gkf is None:
            print("    (Aviso: muy pocas fechas distintas para validación por fechas; se usa el split aleatorio)")

        resultados = {}

        # ==========================================
        # MODELO 1: RANDOM FOREST
        # ==========================================
        print("\n Entrenando Random Forest...")
        crear_rf = lambda: RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        resultados['Random Forest'] = self._metricas_dobles(
            crear_rf, X, y, X_train, y_train, X_test, y_test, fechas, gkf)
        rf_model = crear_rf().fit(X, y)  # modelo final: TODOS los datos
        _guardar_atomico(self.archivo_modelo_rf, lambda f: pickle.dump(rf_model, f))
        print(f"    R2 honesto: {resultados['Random Forest']['r2']}, "
              f"Precisión honesta: {resultados['Random Forest']['precision']}% "
              f"(conocido: {resultados['Random Forest']['precision_conocido']}%)")

        # ==========================================
        # MODELO 2: XGBOOST
        # ==========================================
        xgb_model = None
        if XGBOOST_DISPONIBLE:
            print("\n Entrenando XGBoost...")
            crear_xgb = lambda: xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42, objective='reg:squarederror')
            resultados['XGBoost'] = self._metricas_dobles(
                crear_xgb, X, y, X_train, y_train, X_test, y_test, fechas, gkf)
            xgb_model = crear_xgb().fit(X, y)  # modelo final: TODOS los datos
            _guardar_atomico(self.archivo_modelo_xgb, lambda f: pickle.dump(xgb_model, f))
            print(f"    R2 honesto: {resultados['XGBoost']['r2']}, "
                  f"Precisión honesta: {resultados['XGBoost']['precision']}% "
                  f"(conocido: {resultados['XGBoost']['precision_conocido']}%)")
        else:
            resultados['XGBoost'] = {
                'rmse': 0, 'mae': 0, 'r2': 0, 'precision': 0,
                'error': 'XGBoost no instalado'
            }

        # ==========================================
        # MODELO 3: RIDGE (Regresión Lineal regularizada)
        # ==========================================
        # Familia distinta a los árboles (RF/XGBoost): da contraste real en la comparación y se
        # porta bien con pocos datos. Se estandarizan las variables (StandardScaler) para que la
        # regularización trate a todas por igual sin importar su escala (mes 1-12 vs es_fin_semana 0/1).
        print("\n Entrenando Ridge (Regresión Lineal)...")
        crear_ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        resultados['Ridge'] = self._metricas_dobles(
            crear_ridge, X, y, X_train, y_train, X_test, y_test, fechas, gkf)
        ridge_model = crear_ridge().fit(X, y)  # modelo final: TODOS los datos
        _guardar_atomico(self.archivo_modelo_ridge, lambda f: pickle.dump(ridge_model, f))
        print(f"    R2 honesto: {resultados['Ridge']['r2']}, "
              f"Precisión honesta: {resultados['Ridge']['precision']}% "
              f"(conocido: {resultados['Ridge']['precision_conocido']}%)")

        # ==========================================
        # REFERENCIA: BASELINE INGENUO (predice el promedio)
        # ==========================================
        # NO es un modelo de ML: predice SIEMPRE la ocupación promedio del entrenamiento, ignorando
        # las variables. Es el "piso" a vencer: si un modelo real no le gana, no está aportando nada.
        # Se calcula y reporta, pero NUNCA se elige como ganador para predecir.
        print("\n Calculando Baseline (promedio)...")
        crear_baseline = lambda: DummyRegressor(strategy='mean')
        resultados['Baseline'] = self._metricas_dobles(
            crear_baseline, X, y, X_train, y_train, X_test, y_test, fechas, gkf)
        baseline_model = crear_baseline().fit(X, y)  # referencia final: promedio de TODOS los datos
        _guardar_atomico(self.archivo_modelo_baseline, lambda f: pickle.dump(baseline_model, f))
        print(f"    R2 honesto: {resultados['Baseline']['r2']}, "
              f"Precisión honesta: {resultados['Baseline']['precision']}% "
              f"(conocido: {resultados['Baseline']['precision_conocido']}%)")

        # ==========================================
        # COMPARACIÓN Y SELECCIÓN DEL MEJOR
        # ==========================================
        # Se elige por la PRECISIÓN HONESTA (la que generaliza a fechas nuevas), no por la inflada.
        # El Baseline NO entra en la selección: es solo la referencia a vencer.
        mejor_modelo = 'Random Forest'
        for nombre in ('XGBoost', 'Ridge'):
            if resultados.get(nombre, {}).get('precision', 0) > resultados[mejor_modelo]['precision']:
                mejor_modelo = nombre

        self.modelos = {
            'Baseline': baseline_model,
            'Ridge': ridge_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model if XGBOOST_DISPONIBLE else None,
        }
        self.metricas = resultados
        self.nombre_modelo = mejor_modelo
        self.modelo_entrenado = True
        self.fecha_entrenamiento = datetime.now().isoformat()

        # Se guarda cuál ganó para que, si el proceso se reinicia y solo se recarga el modelo
        # (cargar_modelo, sin volver a entrenar), se siga usando el ganador real y no el
        # "Random Forest" por defecto de la clase - antes se perdía este dato al reiniciar.
        _guardar_atomico(self.archivo_ganador, lambda f: json.dump({
            'nombre_modelo': mejor_modelo,
            'fecha_entrenamiento': self.fecha_entrenamiento,
            'metricas': resultados
        }, f), modo='w')

        return {
            'mejor_modelo': mejor_modelo,
            'metricas': resultados,
            'comparacion': self.generar_comparacion()
        }

    def generar_comparacion(self):
        """Genera un resumen comparativo"""
        baseline = self.metricas.get('Baseline', {})
        ridge = self.metricas.get('Ridge', {})
        rf = self.metricas.get('Random Forest', {})
        xgb = self.metricas.get('XGBoost', {})

        # El ganador se elige SOLO entre los modelos reales (el baseline es la referencia a vencer).
        candidatos = {'Ridge': ridge, 'Random Forest': rf, 'XGBoost': xgb}
        ganador = max(candidatos, key=lambda n: candidatos[n].get('precision', 0))
        precisiones = [c.get('precision', 0) for c in candidatos.values()]

        return {
            'ganador': ganador,
            'baseline': baseline,
            'ridge': ridge,
            'random_forest': rf,
            'xgboost': xgb,
            'diferencia_precision': round(max(precisiones) - min(precisiones), 2)
        }

    def predecir(self, datos_input):
        """Genera predicción usando el mejor modelo, manejando columnas faltantes.

        NOTA: el modelo ya no usa check-in / pernoctaciones / habitaciones ocupadas / tarifa (son
        fuga de datos, ver COLUMNAS_FUGA); si llegan en datos_input, preprocesar() las descarta. La
        predicción depende solo de la fecha (calendario), la temporada y el clima de esa fecha."""
        datos_input = dict(datos_input)  # no mutar el dict del llamador
        if not self.modelo_entrenado:
            self.cargar_modelo()

        modelo = self.modelos.get(self.nombre_modelo)
        if modelo is None:
            raise ValueError("No hay modelo disponible para predecir")

        # Todos los modelos en uso (Ridge, Random Forest, XGBoost) son estimadores de sklearn que
        # predicen a partir de la matriz de variables ya preprocesada, así que comparten este camino.
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

        if os.path.exists(self.archivo_modelo_ridge):
            with open(self.archivo_modelo_ridge, 'rb') as f:
                self.modelos['Ridge'] = pickle.load(f)

        if os.path.exists(self.archivo_modelo_baseline):
            with open(self.archivo_modelo_baseline, 'rb') as f:
                self.modelos['Baseline'] = pickle.load(f)

        if os.path.exists(self.archivo_ganador):
            with open(self.archivo_ganador, 'r', encoding='utf-8') as f:
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

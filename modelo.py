import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class ModeloPredictor:
    def __init__(self):
        self.modelos = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'regresion_lineal': LinearRegression()
        }
        self.modelo_entrenado = None
        self.nombre_modelo = None
        self.label_encoder_temporada = LabelEncoder()
        self.metricas = {}
        self.columnas_x = []
        
    def preparar_datos(self, df):
        """Prepara los datos para el entrenamiento"""
        df = df.copy()
        
        # Eliminar filas con valores nulos en ocupación
        df = df.dropna(subset=['ocupacion_porcentaje'])
        
        # Llenar valores nulos con la media
        columnas_numericas = ['temperatura', 'humedad', 'precipitacion', 
                             'total_dias', 'checkin_nacionales', 
                             'checkin_extranjeros', 'tarifa_cobrada']
        
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # Codificar variable categórica 'temporada'
        if 'temporada' in df.columns:
            df['temporada'] = df['temporada'].fillna('Media')
            df['temporada_cod'] = self.label_encoder_temporada.fit_transform(df['temporada'])
        else:
            df['temporada_cod'] = 0
        
        # Extraer características de fecha
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['mes'] = df['fecha'].dt.month
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
        
        # Variables predictoras
        columnas_x = ['checkin_nacionales', 'checkin_extranjeros', 
                     'tarifa_cobrada', 'temperatura', 'humedad', 
                     'precipitacion', 'total_dias', 'temporada_cod',
                     'mes', 'dia_semana', 'es_fin_semana']
        
        columnas_x = [c for c in columnas_x if c in df.columns]
        
        X = df[columnas_x]
        y = df['ocupacion_porcentaje']
        
        return X, y, columnas_x
    
    def calcular_precision(self, valores_reales, valores_predichos):
        """Calcula la precisión según la fórmula de tu tesis"""
        precisiones = []
        for real, pred in zip(valores_reales, valores_predichos):
            if real != 0:
                precision = (1 - abs(real - pred) / real) * 100
                precisiones.append(max(0, precision))
            else:
                precisiones.append(0)
        
        return np.mean(precisiones) if precisiones else 0
    
    def entrenar(self, df):
        """Entrena todos los modelos y selecciona el mejor"""
        X, y, columnas_x = self.preparar_datos(df)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, test_size=0.5, random_state=42)
        
        print(f"\nDatos divididos:")
        print(f"   Entrenamiento: {len(X_train)} registros")
        print(f"   Validación: {len(X_val)} registros")
        print(f"   Prueba: {len(X_test)} registros")
        
        resultados = {}
        
        for nombre, modelo in self.modelos.items():
            print(f"\nEntrenando {nombre}...")
            
            modelo.fit(X_train, y_train)
            
            y_test_pred = modelo.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2 = r2_score(y_test, y_test_pred)
            precision = self.calcular_precision(y_test.values, y_test_pred)
            
            resultados[nombre] = {
                'modelo': modelo,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'precision': precision
            }
            
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   R²: {r2:.4f}")
            print(f"   Precisión: {precision:.2f}%")
        
        # Seleccionar el mejor modelo
        mejor_nombre = max(resultados, key=lambda k: resultados[k]['precision'])
        self.modelo_entrenado = resultados[mejor_nombre]['modelo']
        self.nombre_modelo = mejor_nombre
        self.metricas = {
            'mae': resultados[mejor_nombre]['mae'],
            'rmse': resultados[mejor_nombre]['rmse'],
            'r2': resultados[mejor_nombre]['r2'],
            'precision': resultados[mejor_nombre]['precision']
        }
        self.columnas_x = columnas_x
        
        print(f"\nMejor modelo: {mejor_nombre}")
        print(f"   Precisión: {self.metricas['precision']:.2f}%")
        
        self.guardar_modelo()
        
        return {
            'mejor_modelo': mejor_nombre,
            'metricas': self.metricas,
            'todos_modelos': {k: {
                'mae': v['mae'],
                'rmse': v['rmse'],
                'r2': v['r2'],
                'precision': v['precision']
            } for k, v in resultados.items()}
        }
    
    def predecir(self, datos_entrada):
        """Realiza una predicción"""
        if self.modelo_entrenado is None:
            raise ValueError("No hay modelo entrenado")
        
        if isinstance(datos_entrada, dict):
            df = pd.DataFrame([datos_entrada])
        else:
            df = datos_entrada
        
        if 'temporada' in df.columns:
            df['temporada_cod'] = self.label_encoder_temporada.transform(df['temporada'])
        else:
            df['temporada_cod'] = 0
        
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            df['mes'] = df['fecha'].dt.month
            df['dia_semana'] = df['fecha'].dt.dayofweek
            df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
        
        columnas = [c for c in self.columnas_x if c in df.columns]
        X = df[columnas]
        
        prediccion = self.modelo_entrenado.predict(X)
        
        return prediccion[0]
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado"""
        carpeta = 'modelos_guardados'
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        
        data = {
            'modelo': self.modelo_entrenado,
            'nombre_modelo': self.nombre_modelo,
            'metricas': self.metricas,
            'label_encoder': self.label_encoder_temporada,
            'columnas_x': self.columnas_x
        }
        
        ruta = os.path.join(carpeta, 'modelo_ots.pkl')
        joblib.dump(data, ruta)
        print(f"Modelo guardado en: {ruta}")
    
    def cargar_modelo(self):
        """Carga un modelo previamente entrenado"""
        ruta = 'modelos_guardados/modelo_ots.pkl'
        
        if not os.path.exists(ruta):
            return False
        
        data = joblib.load(ruta)
        self.modelo_entrenado = data['modelo']
        self.nombre_modelo = data['nombre_modelo']
        self.metricas = data['metricas']
        self.label_encoder_temporada = data['label_encoder']
        self.columnas_x = data['columnas_x']
        
        print(f"Modelo cargado: {self.nombre_modelo}")
        return True
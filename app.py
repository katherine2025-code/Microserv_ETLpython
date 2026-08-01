from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from datetime import datetime
import uvicorn
import io
import pymysql
import os
from dotenv import load_dotenv
import re

from data_loader import obtener_dataset_ml
from modelo import ModeloPredictor

# Cargar variables de entorno desde el archivo .env
load_dotenv()

app = FastAPI(
    title="Microservicio ML - OTS Santa Elena",
    description="API para predicción de ocupación hotelera usando Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8100"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = ModeloPredictor()

# ==========================================
# CONFIGURACIÓN DE BASE DE DATOS
# ==========================================
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'vidamiaIsa2910.'), 
    'database': os.getenv('DB_NAME', 'ots'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

print(f"\n Configuración de BD en Python:")
print(f"   Host: {DB_CONFIG['host']}")
print(f"   Usuario: {DB_CONFIG['user']}")
print(f"   Base de datos: {DB_CONFIG['database']}\n")

def get_db_connection():
    """Crear conexión a MySQL"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f" ERROR DE CONEXIÓN A BD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error de conexión BD: {str(e)}")


# ==========================================
# ENDPOINTS ETL
# ==========================================
@app.post("/etl/procesar")
async def procesar_csv(file: UploadFile = File(...), tipo: str = None):
    try:
        contents = await file.read()
        print(f"\n Intentando leer CSV...")
        print(f" Tamaño del archivo: {len(contents)} bytes")
        
        mejor_df = None
        separadores = [';', ',', '\t']
        encodings = ['utf-8', 'latin-1']
        
        for sep in separadores:
            for enc in encodings:
                try:
                    df_temp = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=enc, on_bad_lines='skip', nrows=5)
                    if df_temp is not None and len(df_temp.columns) > 1:
                        primera_col = str(df_temp.columns[0]).lower()
                        if sep not in primera_col:
                            mejor_df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=enc, on_bad_lines='skip')
                            print(f" Lectura exitosa con separador '{sep}' y encoding '{enc}'")
                            break
                except Exception:
                    continue
            if mejor_df is not None:
                break
                
        if mejor_df is None:
            raise HTTPException(status_code=400, detail="No se pudo leer el CSV. Verifica que sea un archivo válido.")
        
        df = mejor_df
        columnas_lower = [str(col).lower() for col in df.columns]
        
        print(f"\n Columnas encontradas: {list(df.columns)}")
        print(f" Columnas en minúsculas: {columnas_lower}")
        
        # ==========================================
        # DETECCIÓN AUTOMÁTICA DE TIPO
        # ==========================================
        if tipo is None or tipo == '':
            print("\n Tipo no especificado, detectando automáticamente...")
            
            # Detectar Ocupación Hotelera
            if any(col in columnas_lower for col in ['id_hotel', 'ocupacion_porcentaje', 'checkin_nacionales']):
                tipo = 'ocupacion'
                print(" Detectado: OCUPACIÓN HOTELERA")
            
            # Detectar Clima
            elif any(col in columnas_lower for col in ['temperatura', 'humedad', 'precipitacion']):
                tipo = 'clima'
                print(" Detectado: CLIMA")
            
            # Detectar Feriados
            elif any(col in columnas_lower for col in ['fecha_inicio', 'fecha_fin', 'temporada']):
                tipo = 'feriados'
                print(" Detectado: FERIADOS")
            
            # Detectar Encuestas
            elif any(col in columnas_lower for col in ['genero', 'edad', 'pais_residencia', 'nivel_satisfaccion']):
                tipo = 'encuestas'
                print(" Detectado: ENCUESTAS")
            
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No se pudo detectar el tipo de datos. Columnas: {columnas_lower}"
                )
        else:
            print(f"\n Tipo especificado: {tipo}")
        
        print(f"\n Procesando CSV tipo: {tipo}")
        print(f" Filas leídas: {len(df)}")
        print(f" Columnas encontradas: {len(df.columns)}")
        
        if tipo == 'ocupacion':
            resultado = procesar_ocupacion(df)
        elif tipo == 'clima':
            resultado = procesar_clima(df)
        elif tipo == 'feriados':
            resultado = procesar_feriados(df)
        elif tipo == 'encuestas':
            resultado = procesar_encuestas(df)
        else:
            raise HTTPException(status_code=400, detail=f"Tipo no soportado: {tipo}")
        
        return {
            "success": True,
            "mensaje": "CSV procesado exitosamente",
            "tipo": tipo,
            "total_registros": len(df),
            "registros_insertados": resultado['insertados'],
            "registros_error": resultado['errores'],
            "detalles": resultado.get('detalles', [])[:10]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error general procesando CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando CSV: {str(e)}")

def procesar_ocupacion(df):
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []
    
    try:
        for index, row in df.iterrows():
            try:
                if pd.isna(row.get('fecha')) or pd.isna(row.get('id_hotel')):
                    errores += 1
                    detalles.append(f"Fila {int(index) + 1}: Faltan campos requeridos")
                    continue
                
                fecha = pd.to_datetime(row['fecha']).strftime('%Y-%m-%d')
                sql = """
                INSERT INTO ocupacion_hotelera 
                (id_hotel, fecha, checkin_nacionales, checkin_extranjeros, 
                 pernoctaciones, habitaciones_ocupadas, tarifa_cobrada, ocupacion_porcentaje, fecha_registro)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """
                values = (
                    int(row['id_hotel']), fecha,
                    int(row.get('checkin_nacionales', 0) or 0), int(row.get('checkin_extranjeros', 0) or 0),
                    int(row.get('pernoctaciones', 0) or 0), int(row.get('habitaciones_ocupadas', 0) or 0),
                    float(row.get('tarifa_cobrada', 0) or 0), float(row.get('ocupacion_porcentaje', 0) or 0)
                )
                cursor.execute(sql, values)
                insertados += 1
            except Exception as e:
                errores += 1
                # ✅ AGREGADO: Imprimir el error real en la consola de Python
                error_msg = f"Fila {int(index) + 1}: {str(e)}"
                detalles.append(error_msg)
                print(f"⚠️ ERROR EN FILA {int(index) + 1}: {str(e)}") # <-- ESTO ES CLAVE
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    return {'insertados': insertados, 'errores': errores, 'detalles': detalles[:10]}


def procesar_clima(df):
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores = 0, 0
    try:
        for index, row in df.iterrows():
            try:
                fecha = pd.to_datetime(row['fecha']).strftime('%Y-%m-%d')
                sql = """INSERT INTO clima (fecha, temperatura, humedad, precipitacion, velocidad_viento, descripcion, fecha_registro)
                         VALUES (%s, %s, %s, %s, %s, %s, NOW())"""
                cursor.execute(sql, (fecha, float(row.get('temperatura', 0) or 0), float(row.get('humedad', 0) or 0),
                                   float(row.get('precipitacion', 0) or 0), float(row.get('velocidad_viento', 0) or 0), str(row.get('descripcion', ''))))
                insertados += 1
            except Exception:
                errores += 1
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    return {'insertados': insertados, 'errores': errores}


def procesar_feriados(df):
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores = 0, 0
    try:
        for index, row in df.iterrows():
            try:
                sql = """INSERT INTO feriados (nombre, fecha_inicio, fecha_fin, total_dias, temporada, descripcion, fecha_registro)
                         VALUES (%s, %s, %s, %s, %s, %s, NOW())"""
                cursor.execute(sql, (str(row.get('nombre', '')), pd.to_datetime(row['fecha_inicio']).strftime('%Y-%m-%d'),
                                   pd.to_datetime(row['fecha_fin']).strftime('%Y-%m-%d'), int(row.get('total_dias', 1) or 1),
                                   str(row.get('temporada', 'Media')), str(row.get('descripcion', ''))))
                insertados += 1
            except Exception:
                errores += 1
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    return {'insertados': insertados, 'errores': errores}


def procesar_encuestas(df):
    """
    PROCESO ETL GENÉRICO - Detecta y procesa CUALQUIER formato de CSV de encuestas
    (100% compatible con la estructura actual de la BD sin cedula_encuestador)
    """
    print("\n INICIANDO PROCESO ETL GENÉRICO PARA ENCUESTAS...")
    print(f" DataFrame: {len(df)} filas x {len(df.columns)} columnas")
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    insertados, errores, detalles = 0, 0, []
    
    try:
        # ==========================================
        # FASE 1: DETECCIÓN AUTOMÁTICA DE ESTRUCTURA
        # ==========================================
        print("\n Fase 1: Analizando estructura del CSV...")
        
        mapeo_inteligente = {
            'fecha_encuesta': ['start', 'fecha', 'fecha_encuesta', 'date', 'timestamp', 'end', 'submission_time', 'detalles_de_cuenta', 'submission'],
            'genero': ['genero', 'sexo', 'gender', 'mujer', 'hombre'],
            'edad': ['edad', 'age', 'años', 'years'],
            'pais_residencia': ['pais', 'country', 'nacionalidad', 'residencia', 'pais_residencia']
        }
        
        columnas_detectadas = {}
        for col in df.columns:
            col_lower = str(col).lower()
            for campo, posibles_nombres in mapeo_inteligente.items():
                if any(nombre in col_lower for nombre in posibles_nombres):
                    if campo not in columnas_detectadas:
                        columnas_detectadas[campo] = col
                        print(f" Detectada columna '{campo}': {col}")
                        break
        
        # ==========================================
        # FASE 2: EXTRACCIÓN INTELIGENTE DE DATOS
        # ==========================================
        print("\n Fase 2: Extrayendo datos...")
        
        def extraer_fecha(valor):
            if not valor or pd.isna(valor) or str(valor) in ['nan', '', 'None']:
                return None
            valor_str = str(valor).strip()
            formatos = ['%d/%m/%Y %H:%M', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']
            for fmt in formatos:
                try:
                    fecha_dt = datetime.strptime(valor_str, fmt)
                    if 2020 <= fecha_dt.year <= 2030:
                        return fecha_dt.strftime('%Y-%m-%d')
                except:
                    continue
            try:
                fecha_dt = pd.to_datetime(valor_str, dayfirst=False)
                if 2020 <= fecha_dt.year <= 2030:
                    return fecha_dt.strftime('%Y-%m-%d')
            except:
                pass
            return None

        def extraer_genero(valor):
            if not valor or pd.isna(valor):
                return None
            valor_str = str(valor).lower()
            if any(x in valor_str for x in ['mujer', 'femenino', 'female']):
                return 'Femenino'
            elif any(x in valor_str for x in ['hombre', 'masculino', 'male']):
                return 'Masculino'
            return None

        def extraer_edad(valor):
            if not valor or pd.isna(valor):
                return None
            valor_str = str(valor).lower()
            numeros = re.findall(r'\b(\d{1,3})\b', valor_str)
            if numeros:
                edad = int(numeros[0])
                if 5 <= edad <= 120:
                    return edad
            match = re.search(r'(\d+)\s*a\s*(\d+)', valor_str)
            if match:
                edad_min = int(match.group(1))
                edad_max = int(match.group(2))
                if 5 <= edad_min <= 120 and 5 <= edad_max <= 120:
                    return (edad_min + edad_max) // 2
            return None

        def extraer_pais(valor):
            if not valor or pd.isna(valor):
                return 'Ecuador'
            valor_str = str(valor).strip()
            if len(valor_str) > 2 and valor_str.lower() not in ['nan', 'none', '']:
                return valor_str[:100]
            return 'Ecuador'

        def extraer_satisfaccion(row):
            for col in df.columns:
                valor = str(row.get(col, '')).lower()
                if 'satisfaccion' in col.lower() or 'satisfecho' in valor:
                    if 'muy satisfecho' in valor or '5' in valor:
                        return 5
                    elif 'satisfecho' in valor or '4' in valor:
                        return 4
                    elif 'ni satisfecho' in valor or '3' in valor:
                        return 3
                    elif 'insatisfecho' in valor or '2' in valor:
                        return 2
                    elif 'muy insatisfecho' in valor or '1' in valor:
                        return 1
            return 3
        
        # ==========================================
        # FASE 3: PROCESAR FILAS
        # ==========================================
        print("\n Fase 3: Procesando filas...")
        
        for idx, row in df.iterrows():
            try:
                fila_num = int(idx) + 1
                
                # Extraer fecha
                fecha_col = columnas_detectadas.get('fecha_encuesta')
                fecha_raw = row.get(fecha_col) if fecha_col else None
                fecha_encuesta = extraer_fecha(fecha_raw)
                if not fecha_encuesta:
                    fecha_encuesta = datetime.now().strftime('%Y-%m-%d')
                
                # Extraer género
                genero_col = columnas_detectadas.get('genero')
                genero_raw = row.get(genero_col) if genero_col else None
                genero = extraer_genero(genero_raw)
                if not genero:
                    for col in df.columns:
                        genero = extraer_genero(row.get(col))
                        if genero:
                            break
                if not genero:
                    genero = 'No especificado'
                
                # Extraer edad
                edad_col = columnas_detectadas.get('edad')
                edad_raw = row.get(edad_col) if edad_col else None
                edad = extraer_edad(edad_raw)
                if not edad:
                    for col in df.columns:
                        edad = extraer_edad(row.get(col))
                        if edad:
                            break
                if not edad:
                    edad = 0
                
                # Extraer país
                pais_col = columnas_detectadas.get('pais_residencia')
                pais_raw = row.get(pais_col) if pais_col else None
                pais = extraer_pais(pais_raw)
                
                # Extraer satisfacción
                satisfaccion = extraer_satisfaccion(row)
                
                # ✅ INSERTAR (SIN CÉDULA, COINCIDE EXACTAMENTE CON TU BD ACTUAL)
                sql = """
                INSERT INTO encuestas_turisticas 
                (fecha_encuesta, genero, edad, pais_residencia, 
                 ciudad_residencia, motivo_visita, noches_estadia,
                 gasto_total, nivel_satisfaccion, probabilidad_retorno,
                 fecha_registro)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """
                
                values = (
                    fecha_encuesta,
                    genero,
                    edad,
                    pais,
                    '',  # ciudad_residencia
                    '',  # motivo_visita
                    0,   # noches_estadia
                    0.0, # gasto_total
                    satisfaccion,
                    satisfaccion # probabilidad_retorno
                )
                
                cursor.execute(sql, values)
                insertados += 1
                
                if insertados % 100 == 0:
                    print(f"    Procesados: {insertados} registros...")
                
            except Exception as e:
                errores += 1
                if len(detalles) < 5:
                    detalles.append(f"Fila {fila_num}: {str(e)[:80]}")
        
        connection.commit()
        
        print("\n" + "="*60)
        print(" PROCESO ETL COMPLETADO")
        print("="*60)
        print(f" Total procesado: {len(df)}")
        print(f" Insertados: {insertados}")
        print(f" Errores: {errores}")
        if len(df) > 0:
            print(f" Tasa de éxito: {(insertados/len(df)*100):.2f}%")
        if detalles:
            print(f" Errores: {detalles}")
        print("="*60)
        
    except Exception as e:
        print(f" ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    
    return {'insertados': insertados, 'errores': errores, 'detalles': detalles}
# ==========================================
# ENDPOINTS EXISTENTES (ML)
# ==========================================

class PrediccionRequest(BaseModel):
    fecha_objetivo: str
    checkin_nacionales: Optional[float] = None
    checkin_extranjeros: Optional[float] = None
    tarifa_cobrada: Optional[float] = None
    temperatura: Optional[float] = None
    humedad: Optional[float] = None
    precipitacion: Optional[float] = None
    total_dias: Optional[float] = 0
    temporada: Optional[str] = "Media"

@app.get("/")
def raiz():
    return {"servicio": "Microservicio ML - OTS Santa Elena", "version": "1.0.0", "estado": "operativo"}

@app.post("/entrenar")
def entrenar_modelo():
    try:
        print("\n Iniciando entrenamiento comparativo (RF vs XGBoost)...")
        df = obtener_dataset_ml()

        if df is None or len(df) == 0:
            raise HTTPException(status_code=400, detail="No hay datos disponibles. Carga primero datos mediante el ETL.")

        if len(df) < 30:
            raise HTTPException(status_code=400, detail=f"Se necesitan al menos 30 registros. Solo hay {len(df)}.")

        if 'ocupacion_porcentaje' not in df.columns:
            if 'ocupacion' in df.columns:
                df.rename(columns={'ocupacion': 'ocupacion_porcentaje'}, inplace=True)
            else:
                raise HTTPException(status_code=400, detail="No se encontró la columna 'ocupacion_porcentaje'.")

        # ✅ Entrena ambos modelos y devuelve comparación
        resultados = modelo.entrenar(df)

        return {
            "mensaje": "Entrenamiento comparativo completado exitosamente",
            "mejor_modelo": resultados['mejor_modelo'],
            "metricas": resultados['metricas'],
            "comparacion": resultados['comparacion']
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f" ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/predecir")
def predecir_ocupacion(request: PrediccionRequest):
    try:
        print("\n INICIANDO PREDICCIÓN...")
        print(f"Datos recibidos: {request}")
        
        # Verificar si hay modelo entrenado
        if modelo.modelo_entrenado is None:
            print(" No hay modelo en memoria, intentando cargar desde archivo...")
            if not modelo.cargar_modelo():
                print(" ERROR: No se pudo cargar el modelo")
                raise HTTPException(status_code=400, detail="No hay modelo entrenado. Primero ejecuta /entrenar")
            print(" Modelo cargado exitosamente")
        
        # Preparar datos para predicción
        datos = {
            'fecha': request.fecha_objetivo,
            'checkin_nacionales': request.checkin_nacionales or 50,
            'checkin_extranjeros': request.checkin_extranjeros or 10,
            'tarifa_cobrada': request.tarifa_cobrada or 80,
            'temperatura': request.temperatura or 26,
            'humedad': request.humedad or 70,
            'precipitacion': request.precipitacion or 0,
            'total_dias': request.total_dias or 3,
            'temporada': request.temporada or "Media"
        }
        
        print(f" Datos para predicción: {datos}")
        
        # Generar predicción
        prediccion = modelo.predecir(datos)
        prediccion = max(0, min(100, prediccion))  # Asegurar que esté entre 0-100
        
        error_estimado = modelo.metricas.get('rmse', 5)
        
        print(f" Predicción generada: {prediccion}%")
        
        return {
            "fecha_objetivo": request.fecha_objetivo,
            "ocupacion_predicha": round(float(prediccion), 2),
            "modelo": modelo.nombre_modelo,
            "precision": round(modelo.metricas.get('precision', 0), 2),
            "error_estimado": round(error_estimado, 2),
            "rango_prediccion": {
                "minimo": round(max(0, prediccion - error_estimado), 2),
                "maximo": round(min(100, prediccion + error_estimado), 2)
            },
            "mensaje": "Predicción de Ocupación Hotelera generada exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f" ERROR CRÍTICO EN PREDICCIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al generar predicción: {str(e)}")

@app.get("/predicciones-historicas")
def predicciones_historicas(limite: int = None, dias: int = None):
    """
    Genera predicciones para datos históricos y compara con valores reales
    
    Args:
        limite: Número máximo de registros a validar (opcional, default: todos)
        dias: Número de días hacia atrás para validar (opcional)
    """
    try:
        print("\n Generando predicciones históricas...")
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Construir consulta dinámica
        query = """
            SELECT fecha, checkin_nacionales, checkin_extranjeros, 
                   pernoctaciones, habitaciones_ocupadas, tarifa_cobrada, 
                   ocupacion_porcentaje
            FROM ocupacion_hotelera
            WHERE ocupacion_porcentaje IS NOT NULL 
              AND ocupacion_porcentaje > 0
        """
        
        params = []
        
        # Si se especifica número de días, filtrar por fecha
        if dias is not None:
            query += " AND fecha >= DATE_SUB(NOW(), INTERVAL %s DAY)"
            params.append(dias)
            print(f"    Filtrando últimos {dias} días...")
        
        query += " ORDER BY fecha DESC"
        
        # Si se especifica un límite, aplicarlo
        if limite is not None:
            query += " LIMIT %s"
            params.append(limite)
            print(f"    Límite: {limite} registros...")
        else:
            print("    Procesando TODOS los registros disponibles...")
        
        cursor.execute(query, params)
        
        datos_reales = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not datos_reales:
            raise HTTPException(status_code=404, detail="No hay datos históricos disponibles")
        
        print(f"     Datos históricos encontrados: {len(datos_reales)} registros")
        
        # 2. Cargar el modelo entrenado
        if not modelo.modelo_entrenado:
            print("    Cargando modelo desde archivo...")
            if not modelo.cargar_modelo():
                raise HTTPException(status_code=400, detail="No hay modelo entrenado. Primero ejecuta /entrenar")
        
        # 3. Generar predicciones para cada registro histórico
        resultados = []
        registros_procesados = 0
        
        for row in datos_reales:
            try:
                datos_input = {
                    'fecha': row['fecha'].strftime('%Y-%m-%d') if row['fecha'] else '2026-01-01',
                    'checkin_nacionales': int(row['checkin_nacionales'] or 0),
                    'checkin_extranjeros': int(row['checkin_extranjeros'] or 0),
                    'pernoctaciones': int(row['pernoctaciones'] or 0),
                    'habitaciones_ocupadas': int(row['habitaciones_ocupadas'] or 0),
                    'tarifa_cobrada': float(row['tarifa_cobrada'] or 0),
                    'temperatura': 26.0,
                    'humedad': 70.0,
                    'precipitacion': 0.0,
                    'total_dias': 3,
                    'temporada': 'Media'
                }
                
                # Preprocesar
                df_input = pd.DataFrame([datos_input])
                df_input = modelo.preprocesar(df_input)
                
                # Alinear columnas
                modelo_actual = modelo.modelos.get(modelo.nombre_modelo)
                if modelo_actual and hasattr(modelo_actual, 'feature_names_in_'):
                    expected_features = modelo_actual.feature_names_in_
                    for col in expected_features:
                        if col not in df_input.columns:
                            df_input[col] = 0
                    df_input = df_input[expected_features]
                
                # Predecir
                prediccion = modelo_actual.predict(df_input)[0]
                prediccion = float(max(0, min(100, prediccion)))
                
                # Calcular error
                valor_real = float(row['ocupacion_porcentaje'])
                error = abs(valor_real - prediccion)
                
                resultados.append({
                    'fecha': row['fecha'].strftime('%Y-%m-%d') if row['fecha'] else '2026-01-01',
                    'valor_real': round(float(valor_real), 2),
                    'valor_predicho': round(float(prediccion), 2),
                    'error': round(float(error), 2),
                    'precision': round(float(100 - error), 2)
                })
                
                registros_procesados += 1
                
                # Mostrar progreso cada 100 registros
                if registros_procesados % 100 == 0:
                    print(f"    Procesados: {registros_procesados} registros...")
                
            except Exception as e:
                print(f"     Error procesando fila: {str(e)}")
                continue
        
        if not resultados:
            raise HTTPException(status_code=500, detail="No se pudieron generar predicciones")
        
        # 4. Calcular métricas generales
        errores = [float(r['error']) for r in resultados]
        precision_promedio = sum(float(r['precision']) for r in resultados) / len(resultados)
        
        respuesta = {
            'total_registros': int(len(resultados)),
            'precision_promedio': round(float(precision_promedio), 2),
            'error_promedio': round(float(sum(errores) / len(errores)), 2),
            'modelo_usado': str(modelo.nombre_modelo),
            'predicciones': resultados
        }
        
        print(f"\n" + "="*60)
        print(" VALIDACIÓN HISTÓRICA COMPLETADA")
        print("="*60)
        print(f" Total registros: {len(resultados)}")
        print(f" Precisión promedio: {respuesta['precision_promedio']}%")
        print(f" Error promedio: {respuesta['error_promedio']}%")
        print("="*60)
        
        return respuesta
        
    except HTTPException:
        raise
    except Exception as e:
        print(f" ERROR en predicciones históricas: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Microservicio ML - OTS Santa Elena")
    print("="*60)
    print("API: http://localhost:5000")
    print("Documentación: http://localhost:5000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)
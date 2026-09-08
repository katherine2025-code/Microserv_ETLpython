from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from datetime import datetime
import uvicorn
import io
import pymysql
import os
from dotenv import load_dotenv
import re
import logging

from data_loader import obtener_dataset_ml
from modelo import ModeloPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="Microservicio ML - OTS Santa Elena",
    description="API para predicción de ocupación hotelera usando Machine Learning",
    version="1.0.0"
)

# ==========================================
# CONFIGURACIÓN CORS ACTUALIZADA
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Backend Node.js
        "http://localhost:8100",   # Frontend Ionic
        "http://localhost:8101",   # Frontend Ionic
        "http://localhost:5000",   # Este mismo servicio
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8100",
        "http://127.0.0.1:8101",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Inicializar modelo
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

print(f"\n📊 Configuración de BD en Python:")
print(f"   Host: {DB_CONFIG['host']}")
print(f"   Usuario: {DB_CONFIG['user']}")
print(f"   Base de datos: {DB_CONFIG['database']}\n")

def get_db_connection():
    """Crear conexión a MySQL"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"❌ ERROR DE CONEXIÓN A BD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error de conexión BD: {str(e)}")

# ==========================================
# NUEVOS ENDPOINTS PARA COMPATIBILIDAD CON BACKEND NODE.JS
# ==========================================

@app.get("/")
async def root():
    """Raíz del servicio"""
    return {
        "service": "ML Service OTS",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "modelo_entrenado": modelo.modelo_entrenado is not None,
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check para el backend Node.js"""
    return {
        "status": "OK",
        "service": "ML Service OTS",
        "timestamp": datetime.now().isoformat(),
        "modelo_entrenado": modelo.modelo_entrenado is not None,
        "precision": float(modelo.metricas.get('precision', 0)) if modelo.modelo_entrenado else None,
        "fecha_entrenamiento": getattr(modelo, 'fecha_entrenamiento', None),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Métricas del modelo para el backend Node.js"""
    if modelo.modelo_entrenado is None:
        return {
            "entrenado": False,
            "precision": 0.0,
            "total_predicciones": 0,
            "mensaje": "Modelo no entrenado aún"
        }
    
    # Obtener total de predicciones de la BD
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total FROM predicciones")
            total = cursor.fetchone()['total']
        conn.close()
    except:
        total = 0
    
    return {
        "entrenado": True,
        "precision": float(modelo.metricas.get('precision', 85.0)),
        "total_predicciones": int(total),
        "modelo_usado": str(modelo.nombre_modelo),
        "fecha_entrenamiento": getattr(modelo, 'fecha_entrenamiento', datetime.now().isoformat()),
        "metricas": modelo.metricas
    }

@app.get("/metricas")
async def get_metricas():
    """Endpoint alternativo para métricas (compatibilidad)"""
    return await get_metrics()

@app.get("/status")
async def get_status():
    """Estado completo del servicio"""
    entrenado = modelo.modelo_entrenado is not None
    return {
        "service": "ML Service OTS",
        "status": "online",
        "modelo_entrenado": entrenado,
        "precision": float(modelo.metricas.get('precision', 0)) if entrenado else 0,
        "fecha_entrenamiento": getattr(modelo, 'fecha_entrenamiento', None),
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ==========================================
# ENDPOINTS ETL EXISTENTES
# ==========================================

@app.post("/etl/procesar")
async def procesar_csv(file: UploadFile = File(...), tipo: str = None):
    try:
        contents = await file.read()
        print(f"\n📄 Intentando leer CSV...")
        print(f"📦 Tamaño del archivo: {len(contents)} bytes")
        
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
                            print(f"✅ Lectura exitosa con separador '{sep}' y encoding '{enc}'")
                            break
                except Exception:
                    continue
            if mejor_df is not None:
                break
                
        if mejor_df is None:
            raise HTTPException(status_code=400, detail="No se pudo leer el CSV. Verifica que sea un archivo válido.")
        
        df = mejor_df
        columnas_lower = [str(col).lower() for col in df.columns]
        
        print(f"\n📋 Columnas encontradas: {list(df.columns)}")
        
        # Detección automática de tipo
        if tipo is None or tipo == '':
            print("\n🔍 Tipo no especificado, detectando automáticamente...")
            
            if any(col in columnas_lower for col in ['id_hotel', 'ocupacion_porcentaje', 'checkin_nacionales']):
                tipo = 'ocupacion'
                print("✅ Detectado: OCUPACIÓN HOTELERA")
            elif any(col in columnas_lower for col in ['temperatura', 'humedad', 'precipitacion']):
                tipo = 'clima'
                print("✅ Detectado: CLIMA")
            elif any(col in columnas_lower for col in ['fecha_inicio', 'fecha_fin', 'temporada']):
                tipo = 'feriados'
                print("✅ Detectado: FERIADOS")
            elif any(col in columnas_lower for col in ['genero', 'edad', 'pais_residencia', 'nivel_satisfaccion']):
                tipo = 'encuestas'
                print("✅ Detectado: ENCUESTAS")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No se pudo detectar el tipo de datos. Columnas: {columnas_lower}"
                )
        
        print(f"\n🔄 Procesando CSV tipo: {tipo}")
        print(f"📊 Filas leídas: {len(df)}")
        
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
        print(f"❌ Error general procesando CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando CSV: {str(e)}")

# ==========================================
# FUNCIONES ETL (mantenidas igual)
# ==========================================

def procesar_ocupacion(df):
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []
    
    try:
        registros_a_insertar = []
        for index, row in df.iterrows():
            try:
                if pd.isna(row.get('fecha')) or pd.isna(row.get('id_hotel')):
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(f"Fila {int(index) + 1}: Faltan campos requeridos")
                    continue
                
                fecha = pd.to_datetime(row['fecha']).strftime('%Y-%m-%d')
                registros_a_insertar.append((
                    int(row['id_hotel']), fecha,
                    int(row.get('checkin_nacionales', 0) or 0), int(row.get('checkin_extranjeros', 0) or 0),
                    int(row.get('pernoctaciones', 0) or 0), int(row.get('habitaciones_ocupadas', 0) or 0),
                    float(row.get('tarifa_cobrada', 0) or 0), float(row.get('ocupacion_porcentaje', 0) or 0)
                ))
            except Exception as e:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: {str(e)}")

        sql = """
        INSERT INTO ocupacion_hotelera 
        (id_hotel, fecha, checkin_nacionales, checkin_extranjeros, 
         pernoctaciones, habitaciones_ocupadas, tarifa_cobrada, ocupacion_porcentaje, fecha_registro)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        batch_size = 5000
        for i in range(0, len(registros_a_insertar), batch_size):
            batch = registros_a_insertar[i:i + batch_size]
            cursor.executemany(sql, batch)
            insertados += len(batch)

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
    """Proceso ETL genérico para encuestas"""
    print("\n📊 INICIANDO PROCESO ETL GENÉRICO PARA ENCUESTAS...")
    print(f"📋 DataFrame: {len(df)} filas x {len(df.columns)} columnas")
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    insertados, errores, detalles = 0, 0, []
    
    try:
        # Mapeo inteligente de columnas
        mapeo_inteligente = {
            'fecha_encuesta': ['start', 'fecha', 'fecha_encuesta', 'date', 'timestamp', 'end', 'submission_time'],
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
                        print(f"✅ Detectada columna '{campo}': {col}")
                        break
        
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
        
        print("\n🔄 Procesando filas...")
        
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
                    '',
                    '',
                    0,
                    0.0,
                    satisfaccion,
                    satisfaccion
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
        print("✅ PROCESO ETL COMPLETADO")
        print("="*60)
        print(f"📊 Total procesado: {len(df)}")
        print(f"✅ Insertados: {insertados}")
        print(f"❌ Errores: {errores}")
        if len(df) > 0:
            print(f"📈 Tasa de éxito: {(insertados/len(df)*100):.2f}%")
        print("="*60)
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    
    return {'insertados': insertados, 'errores': errores, 'detalles': detalles}

# ==========================================
# ENDPOINTS ML EXISTENTES (ACTUALIZADOS)
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

class PrediccionRangoRequest(BaseModel):
    fecha_inicio: str
    fecha_fin: str
    id_hotel: Optional[int] = 1

@app.post("/entrenar")
async def entrenar_modelo():
    """Entrenar el modelo con datos de la base de datos"""
    try:
        print("\n🔄 Iniciando entrenamiento comparativo (RF vs XGBoost)...")
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

        # Entrenar ambos modelos
        resultados = modelo.entrenar(df)
        
        # Guardar fecha de entrenamiento
        modelo.fecha_entrenamiento = datetime.now().isoformat()

        print(f"\n✅ Modelo entrenado exitosamente!")
        print(f"   Mejor modelo: {resultados['mejor_modelo']}")
        print(f"   Precisión: {resultados['metricas'].get('precision', 0):.2f}%")

        return {
            "success": True,
            "mensaje": "Entrenamiento completado exitosamente",
            "mejor_modelo": resultados['mejor_modelo'],
            "metricas": resultados['metricas'],
            "comparacion": resultados['comparacion'],
            "fecha_entrenamiento": modelo.fecha_entrenamiento
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/predecir")
async def predecir_ocupacion(request: PrediccionRequest):
    """Endpoint principal de predicción"""
    try:
        print("\n🎯 INICIANDO PREDICCIÓN...")
        print(f"📊 Datos recibidos: {request}")
        
        # Verificar modelo
        if modelo.modelo_entrenado is None:
            print("⚠️ No hay modelo en memoria, intentando cargar...")
            if not modelo.cargar_modelo():
                print("❌ ERROR: No se pudo cargar el modelo")
                raise HTTPException(status_code=400, detail="No hay modelo entrenado. Primero ejecuta /entrenar")
            print("✅ Modelo cargado exitosamente")
        
        # Preparar datos
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
        
        print(f"📊 Datos procesados: {datos}")
        
        # Generar predicción
        prediccion_val = modelo.predecir(datos)
        prediccion_float = float(max(0, min(100, prediccion_val)))
        
        error_estimado = float(modelo.metricas.get('rmse', 5))
        precision_modelo = float(modelo.metricas.get('precision', 85.0))
        
        print(f"✅ Predicción generada: {prediccion_float}%")

        # Guardar en BD
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                sql_insert = """
                    INSERT INTO predicciones 
                    (id_hotel, fecha_objetivo, fecha_generacion, ocupacion_predicha, 
                     precision_modelo, modelo_utilizado, estado)
                    VALUES (%s, %s, NOW(), %s, %s, %s, %s)
                """
                cursor.execute(sql_insert, (
                    1,
                    request.fecha_objetivo,
                    round(prediccion_float, 2),
                    round(precision_modelo, 2),
                    str(modelo.nombre_modelo),
                    'pendiente'
                ))
            conn.commit()
            conn.close()
            print("✅ Predicción guardada en BD")
        except Exception as db_err:
            print(f"⚠️ Advertencia al guardar en BD: {str(db_err)}")
        
        return {
            "success": True,
            "fecha_objetivo": request.fecha_objetivo,
            "ocupacion_predicha": round(prediccion_float, 2),
            "modelo": str(modelo.nombre_modelo),
            "precision": round(precision_modelo, 2),
            "error_estimado": round(error_estimado, 2),
            "rango_prediccion": {
                "minimo": round(max(0, prediccion_float - error_estimado), 2),
                "maximo": round(min(100, prediccion_float + error_estimado), 2)
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN PREDICCIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al generar predicción: {str(e)}")

@app.post("/predecir-rango")
async def predecir_ocupacion_rango(request: PrediccionRangoRequest):
    """Genera proyección día por día para un rango de fechas"""
    try:
        print("\n📊 GENERANDO PROYECCIÓN POR RANGO DE FECHAS...")
        print(f"📅 Desde: {request.fecha_inicio} Hasta: {request.fecha_fin}")
        
        # Verificar modelo
        if not modelo.modelo_entrenado:
            if not modelo.cargar_modelo():
                raise HTTPException(status_code=400, detail="No hay modelo entrenado. Ejecuta primero /entrenar.")
                
        start_date = pd.to_datetime(request.fecha_inicio)
        end_date = pd.to_datetime(request.fecha_fin)
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="La fecha de inicio no puede ser posterior a la fecha de fin.")
            
        dias_diferencia = (end_date - start_date).days + 1
        if dias_diferencia > 60:
            raise HTTPException(status_code=400, detail="El rango máximo permitido es de 60 días.")
            
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Consultar clima
        query_clima = "SELECT fecha, temperatura, humedad, precipitacion FROM clima WHERE fecha BETWEEN %s AND %s"
        cursor.execute(query_clima, (request.fecha_inicio, request.fecha_fin))
        clima_dict = {}
        for row in cursor.fetchall():
            fecha_str = row['fecha'].strftime('%Y-%m-%d') if row['fecha'] else None
            if fecha_str:
                clima_dict[fecha_str] = row
        
        # Consultar feriados
        query_feriados = "SELECT fecha_inicio, fecha_fin, temporada, total_dias FROM feriados WHERE fecha_inicio <= %s AND fecha_fin >= %s"
        cursor.execute(query_feriados, (request.fecha_fin, request.fecha_inicio))
        feriados_list = cursor.fetchall()
        cursor.close()
        connection.close()
        
        predicciones_diarias = []
        fechas_rango = pd.date_range(start_date, end_date)
        
        for fecha_dt in fechas_rango:
            fecha_str = fecha_dt.strftime('%Y-%m-%d')
            es_fin_semana = 1 if fecha_dt.dayofweek >= 5 else 0
            
            info_clima = clima_dict.get(fecha_str, {})
            temp = float(info_clima.get('temperatura', 26.0))
            hum = float(info_clima.get('humedad', 70.0))
            prec = float(info_clima.get('precipitacion', 0.0))
            
            temporada = "Media"
            total_dias_feriado = 1
            for f in feriados_list:
                f_ini = pd.to_datetime(f['fecha_inicio'])
                f_fin = pd.to_datetime(f['fecha_fin'])
                if f_ini <= fecha_dt <= f_fin:
                    temporada = f.get('temporada', 'Alta')
                    total_dias_feriado = f.get('total_dias', 3)
                    break
                    
            if es_fin_semana and temporada == "Media":
                temporada = "Alta"
                
            datos_dia = {
                'fecha': fecha_str,
                'checkin_nacionales': 60 if temporada == "Alta" else 35,
                'checkin_extranjeros': 20 if temporada == "Alta" else 8,
                'tarifa_cobrada': 90.0 if temporada == "Alta" else 70.0,
                'temperatura': temp,
                'humedad': hum,
                'precipitacion': prec,
                'total_dias': total_dias_feriado,
                'temporada': temporada
            }
            
            val_pred = modelo.predecir(datos_dia)
            val_pred = round(float(max(0, min(100, val_pred))), 2)
            
            predicciones_diarias.append({
                'fecha': fecha_str,
                'ocupacion_predicha': val_pred,
                'dia_semana': fecha_dt.strftime('%A'),
                'es_fin_semana': bool(es_fin_semana),
                'temporada': temporada,
                'temperatura': temp
            })
            
        # Calcular métricas
        valores = [p['ocupacion_predicha'] for p in predicciones_diarias]
        promedio = round(sum(valores) / len(valores), 2)
        pico = max(predicciones_diarias, key=lambda x: x['ocupacion_predicha'])
        valle = min(predicciones_diarias, key=lambda x: x['ocupacion_predicha'])
        
        # Promedios por día de semana
        dias_semana_map = {}
        for p in predicciones_diarias:
            d_nom = p['dia_semana']
            if d_nom not in dias_semana_map:
                dias_semana_map[d_nom] = []
            dias_semana_map[d_nom].append(p['ocupacion_predicha'])
        promedios_dia_semana = {d: round(sum(vals)/len(vals), 2) for d, vals in dias_semana_map.items()}
        
        # Promedios por temporada
        temporada_map = {}
        for p in predicciones_diarias:
            temp_nom = p['temporada']
            if temp_nom not in temporada_map:
                temporada_map[temp_nom] = []
            temporada_map[temp_nom].append(p['ocupacion_predicha'])
        promedios_temporada = {t: round(sum(vals)/len(vals), 2) for t, vals in temporada_map.items()}
        
        return {
            'success': True,
            'fecha_inicio': request.fecha_inicio,
            'fecha_fin': request.fecha_fin,
            'total_dias': len(predicciones_diarias),
            'ocupacion_promedio': promedio,
            'dia_pico': pico,
            'dia_valle': valle,
            'promedios_dia_semana': promedios_dia_semana,
            'promedios_temporada': promedios_temporada,
            'modelo_usado': str(modelo.nombre_modelo),
            'predicciones_diarias': predicciones_diarias
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR en predicción por rango: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en proyección: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Microservicio ML - OTS Santa Elena")
    print("="*60)
    print(f"📡 API: http://localhost:5000")
    print(f"📊 Health: http://localhost:5000/health")
    print(f"📈 Métricas: http://localhost:5000/metricas")
    print(f"🤖 Entrenar: POST http://localhost:5000/entrenar")
    print(f"🎯 Predecir: POST http://localhost:5000/predecir")
    print(f"📚 Docs: http://localhost:5000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from datetime import datetime
import uvicorn
import pymysql
import os
from dotenv import load_dotenv
import logging

from data_loader import obtener_dataset_ml
from modelo import ModeloPredictor
from etl_utils import CANONICAL_SCHEMAS, leer_archivo, mapear_posicional, limpiar_y_convertir
from variables_estacionales import generar_variables_estacionales, obtener_o_generar
from kobo_establecimientos import es_formulario_establecimientos, procesar_establecimientos
from kobo_encuestas_turismo import es_formulario_turismo, procesar_encuestas_turismo
from ots_app_encuestas import es_csv_app, tipo_encuesta_del_archivo, procesar_encuestas_app

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
async def procesar_csv(file: UploadFile = File(...), tipo: str = Form(None)):
    if not tipo or tipo not in CANONICAL_SCHEMAS:
        raise HTTPException(
            status_code=400,
            detail=f"Debes indicar un 'tipo' válido: {', '.join(CANONICAL_SCHEMAS.keys())}"
        )

    try:
        contents = await file.read()
        print(f"\n📄 Archivo recibido: {file.filename} ({len(contents)} bytes) - tipo: {tipo}")

        try:
            df_crudo = leer_archivo(file.filename, contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if df_crudo.empty:
            raise HTTPException(status_code=400, detail="El archivo no contiene filas de datos.")

        print(f"📋 Columnas recibidas: {len(df_crudo.columns)} | Filas: {len(df_crudo)}")

        # El formulario Kobo "Establecimientos de Alojamiento" tiene su propia
        # estructura (grupo repetido de hasta 5 registros por envío, sin
        # id_hotel) y se procesa aparte - ver kobo_establecimientos.py.
        # CSV generado por la propia app OTS (encuestas llenadas por los encuestadores):
        # columnas por código de pregunta. Se evalúa primero porque es el flujo principal.
        if tipo in ('encuestas', 'ocupacion') and es_csv_app(df_crudo.columns):
            tipo_archivo = tipo_encuesta_del_archivo(df_crudo)
            esperado = 'turista' if tipo == 'encuestas' else 'hotel'
            if tipo_archivo is None:
                raise HTTPException(
                    status_code=400,
                    detail="El archivo mezcla encuestas de distinto tipo o el 'tipo_encuesta' no es "
                           "'turista' ni 'hotel'. Descarga un archivo por cada tipo de encuesta."
                )
            if tipo_archivo != esperado:
                raise HTTPException(
                    status_code=400,
                    detail=f"El archivo contiene encuestas de tipo '{tipo_archivo}' pero seleccionaste "
                           f"'{'Ocupación Hotelera' if tipo == 'ocupacion' else 'Encuestas Turísticas'}'. "
                           f"Elige '{'Encuestas Turísticas' if tipo_archivo == 'turista' else 'Ocupación Hotelera'}'."
                )
            print(f"📋 Formato detectado: CSV de la app OTS ({tipo_archivo})")
            resultado = procesar_encuestas_app(df_crudo, tipo_archivo, get_db_connection)
            advertencias = resultado.get('advertencias', [])
            total_registros = len(df_crudo)
        elif tipo == 'ocupacion' and es_formulario_establecimientos(df_crudo.columns):
            print("📋 Formulario detectado: Establecimientos de Alojamiento (Kobo)")
            resultado = procesar_establecimientos(df_crudo, get_db_connection)
            advertencias = resultado.get('advertencias', [])
            total_registros = len(df_crudo)
        elif tipo == 'encuestas' and es_formulario_turismo(df_crudo.columns):
            print("📋 Formulario detectado: Recolección de datos Turismo (Kobo)")
            resultado = procesar_encuestas_turismo(df_crudo, get_db_connection)
            advertencias = resultado.get('advertencias', [])
            total_registros = len(df_crudo)
        else:
            # Mapeo POSICIONAL genérico: se ignoran los encabezados del
            # archivo y se usa el orden fijo de columnas esperado para este
            # tipo (ver etl_utils) - para archivos simples de otras fuentes.
            df_mapeado, advertencias_version = mapear_posicional(df_crudo, tipo)

            # Normalización de texto, conversión de tipos y relleno de nulos
            # (numéricos -> 0; texto libre de encuestas -> 'Prefiero no responder').
            df_limpio, advertencias_datos = limpiar_y_convertir(df_mapeado, tipo)

            advertencias = advertencias_version + advertencias_datos
            for a in advertencias:
                print(f"⚠️ {a}")

            print(f"\n🔄 Procesando archivo tipo: {tipo}")

            if tipo == 'ocupacion':
                resultado = procesar_ocupacion(df_limpio)
            elif tipo == 'clima':
                resultado = procesar_clima(df_limpio)
            elif tipo == 'feriados':
                resultado = procesar_feriados(df_limpio)
            elif tipo == 'encuestas':
                resultado = procesar_encuestas(df_limpio)

            total_registros = len(df_limpio)

        return {
            "success": True,
            "mensaje": "Archivo procesado exitosamente",
            "tipo": tipo,
            "total_registros": total_registros,
            "registros_insertados": resultado['insertados'],
            "registros_error": resultado['errores'],
            "advertencias": advertencias[:10],
            "detalles": resultado.get('detalles', [])[:10]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error general procesando archivo: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

# ==========================================
# FUNCIONES ETL (mantenidas igual)
# ==========================================

def procesar_ocupacion(df):
    """Inserta ocupación hotelera. `df` ya viene con columnas canónicas
    (fecha, id_hotel, ...), tipos convertidos y nulos numéricos en 0
    (ver etl_utils.limpiar_y_convertir)."""
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []

    try:
        registros_a_insertar = []
        for index, row in df.iterrows():
            if not row['fecha'] or row['id_hotel'] is None:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: falta 'fecha' o 'id_hotel' (campos obligatorios)")
                continue

            registros_a_insertar.append((
                int(row['id_hotel']), row['fecha'],
                int(row['checkin_nacionales']), int(row['checkin_extranjeros']),
                int(row['pernoctaciones']), int(row['habitaciones_ocupadas']),
                float(row['tarifa_cobrada']), float(row['ocupacion_porcentaje'])
            ))

        sql = """
        INSERT INTO ocupacion_hotelera
        (id_hotel, fecha, checkin_nacionales, checkin_extranjeros,
         pernoctaciones, habitaciones_ocupadas, tarifa_cobrada, ocupacion_porcentaje, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """

        batch_size = 5000
        for i in range(0, len(registros_a_insertar), batch_size):
            batch = registros_a_insertar[i:i + batch_size]
            try:
                cursor.executemany(sql, batch)
                connection.commit()
                insertados += len(batch)
            except Exception as e:
                # Si el lote falla (p.ej. id_hotel inexistente por FK), se
                # reintenta fila por fila para no perder el resto del lote.
                connection.rollback()
                for fila in batch:
                    try:
                        cursor.execute(sql, fila)
                        connection.commit()
                        insertados += 1
                    except Exception as fila_err:
                        connection.rollback()
                        errores += 1
                        if len(detalles) < 10:
                            detalles.append(f"id_hotel {fila[0]}, fecha {fila[1]}: {str(fila_err)[:120]}")
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
    insertados, errores, detalles = 0, 0, []
    try:
        sql = """INSERT INTO clima (fecha, temperatura, humedad, precipitacion, velocidad_viento, descripcion, fecha_registro)
                 VALUES (%s, %s, %s, %s, %s, %s, NOW())"""
        for index, row in df.iterrows():
            if not row['fecha']:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: falta 'fecha' (campo obligatorio)")
                continue
            try:
                cursor.execute(sql, (row['fecha'], row['temperatura'], row['humedad'],
                                      row['precipitacion'], row['velocidad_viento'], row['descripcion'] or ''))
                insertados += 1
            except Exception as e:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: {str(e)[:120]}")
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    return {'insertados': insertados, 'errores': errores, 'detalles': detalles}


def procesar_feriados(df):
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []
    try:
        sql = """INSERT INTO feriados (nombre, fecha_inicio, fecha_fin, total_dias, temporada, descripcion, fecha_registro)
                 VALUES (%s, %s, %s, %s, %s, %s, NOW())"""
        for index, row in df.iterrows():
            if not row['fecha_inicio'] or not row['fecha_fin']:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: falta 'fecha_inicio' o 'fecha_fin' (campos obligatorios)")
                continue
            try:
                cursor.execute(sql, (row['nombre'] or '', row['fecha_inicio'], row['fecha_fin'],
                                      int(row['total_dias']) if row['total_dias'] else 1,
                                      row['temporada'] or 'Media', row['descripcion'] or ''))
                insertados += 1
            except Exception as e:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: {str(e)[:120]}")
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()
    return {'insertados': insertados, 'errores': errores, 'detalles': detalles}


def procesar_encuestas(df):
    """Inserta encuestas turísticas. `df` ya viene con columnas canónicas,
    tipos convertidos, textos normalizados y nulos rellenados (texto libre
    -> 'Prefiero no responder', numéricos -> 0)."""
    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []

    try:
        sql = """
        INSERT INTO encuestas_turisticas
        (fecha_encuesta, genero, edad, pais_residencia, ciudad_residencia,
         motivo_visita, noches_estadia, gasto_total, nivel_satisfaccion,
         probabilidad_retorno, fecha_registro)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        for index, row in df.iterrows():
            try:
                fecha_encuesta = row['fecha_encuesta'] or datetime.now().strftime('%Y-%m-%d')
                cursor.execute(sql, (
                    fecha_encuesta, row['genero'], int(row['edad']),
                    row['pais_residencia'], row['ciudad_residencia'], row['motivo_visita'],
                    int(row['noches_estadia']), float(row['gasto_total']),
                    int(row['nivel_satisfaccion']), int(row['probabilidad_retorno'])
                ))
                insertados += 1
            except Exception as e:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {int(index) + 1}: {str(e)[:120]}")

        connection.commit()
    except Exception as e:
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

class VariablesEstacionalesRequest(BaseModel):
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None

@app.post("/variables-estacionales/generar")
async def generar_variables_estacionales_endpoint(request: VariablesEstacionalesRequest = VariablesEstacionalesRequest()):
    """Recalcula variables_estacionales (mes, dia_semana, es_festivo,
    es_temporada_alta, semana_ano, factor_estacional) para un rango de fechas.
    Útil para regenerar después de cargar/editar festivos o temporadas.
    Sin fechas, regenera un rango por defecto de un año hacia atrás y hacia adelante."""
    try:
        if request.fecha_inicio and request.fecha_fin:
            fecha_inicio = datetime.strptime(request.fecha_inicio, '%Y-%m-%d').date()
            fecha_fin = datetime.strptime(request.fecha_fin, '%Y-%m-%d').date()
        else:
            hoy = datetime.now().date()
            fecha_inicio = hoy.replace(year=hoy.year - 1)
            fecha_fin = hoy.replace(year=hoy.year + 1)

        if fecha_fin < fecha_inicio:
            raise HTTPException(status_code=400, detail="fecha_fin no puede ser anterior a fecha_inicio")

        total = generar_variables_estacionales(fecha_inicio, fecha_fin)
        return {
            "success": True,
            "mensaje": "Variables estacionales generadas",
            "fecha_inicio": fecha_inicio.isoformat(),
            "fecha_fin": fecha_fin.isoformat(),
            "filas_generadas": total
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando variables estacionales: {str(e)}")

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
        
        # Variables de calendario (mes, dia_semana, es_festivo, es_temporada_alta,
        # semana_ano, factor_estacional) - las mismas que usa el entrenamiento,
        # generadas/leídas desde variables_estacionales para esa fecha.
        fecha_obj = datetime.strptime(request.fecha_objetivo, '%Y-%m-%d').date()
        ve = obtener_o_generar(fecha_obj)

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
            'temporada': request.temporada or "Media",
            'mes': ve['mes'],
            'dia_semana': ve['dia_semana'],
            'es_festivo': ve['es_festivo'],
            'es_temporada_alta': ve['es_temporada_alta'],
            'semana_ano': ve['semana_ano'],
            'factor_estacional': ve['factor_estacional']
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
                
            ve = obtener_o_generar(fecha_dt.date())

            datos_dia = {
                'fecha': fecha_str,
                'checkin_nacionales': 60 if temporada == "Alta" else 35,
                'checkin_extranjeros': 20 if temporada == "Alta" else 8,
                'tarifa_cobrada': 90.0 if temporada == "Alta" else 70.0,
                'temperatura': temp,
                'humedad': hum,
                'precipitacion': prec,
                'total_dias': total_dias_feriado,
                'temporada': temporada,
                'mes': ve['mes'],
                'dia_semana': ve['dia_semana'],
                'es_festivo': ve['es_festivo'],
                'es_temporada_alta': ve['es_temporada_alta'],
                'semana_ano': ve['semana_ano'],
                'factor_estacional': ve['factor_estacional']
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

@app.get("/predicciones-historicas")
async def predicciones_historicas(limite: Optional[int] = None, dias: Optional[int] = None):
    """Valida el modelo contra datos reales ya cargados: por cada registro
    histórico de ocupacion_hotelera, genera una predicción con las mismas
    variables usadas en el entrenamiento y la compara contra el valor real."""
    try:
        if not modelo.modelo_entrenado:
            if not modelo.cargar_modelo():
                raise HTTPException(status_code=400, detail="No hay modelo entrenado. Ejecuta primero /entrenar.")

        df = obtener_dataset_ml()
        if df is None or len(df) == 0:
            return {"precision_promedio": 0, "total_registros": 0, "error_promedio": 0, "predicciones": []}

        df['fecha'] = pd.to_datetime(df['fecha'])
        if dias:
            fecha_limite = df['fecha'].max() - pd.Timedelta(days=dias)
            df = df[df['fecha'] >= fecha_limite]

        df = df.sort_values('fecha', ascending=False)
        if limite:
            df = df.head(limite)

        predicciones = []
        errores = []

        for _, row in df.iterrows():
            datos = row.to_dict()
            real = float(datos.pop('ocupacion_porcentaje'))
            fecha_str = datos['fecha'].strftime('%Y-%m-%d')
            datos['fecha'] = fecha_str

            try:
                predicho = float(modelo.predecir(datos))
            except Exception as e:
                print(f"⚠️ No se pudo predecir la fila de {fecha_str}: {e}")
                continue

            error = abs(real - predicho)
            if real > 0:
                precision_fila = max(0, 100 - (error / real * 100))
            else:
                precision_fila = 100 if predicho == 0 else 0

            errores.append(error)
            predicciones.append({
                "fecha": fecha_str,
                "valor_real": round(real, 2),
                "valor_predicho": round(predicho, 2),
                "error": round(error, 2),
                "precision": round(precision_fila, 2)
            })

        total = len(predicciones)
        error_promedio = round(sum(errores) / total, 2) if total else 0
        precision_promedio = round(sum(p['precision'] for p in predicciones) / total, 2) if total else 0

        return {
            "precision_promedio": precision_promedio,
            "total_registros": total,
            "error_promedio": error_promedio,
            "predicciones": predicciones
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR en validación histórica: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en validación histórica: {str(e)}")

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
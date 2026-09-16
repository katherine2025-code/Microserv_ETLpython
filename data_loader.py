import pymysql
import pandas as pd
import os
from dotenv import load_dotenv
from variables_estacionales import generar_variables_estacionales

load_dotenv()

def obtener_dataset_ml():
    """Obtiene el dataset completo para entrenar el modelo ML"""
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'vidamiaIsa2910.'),
            database=os.getenv('DB_NAME', 'ots'),
            charset='utf8mb4'
        )

        # Asegura que exista una fila de variables_estacionales para cada
        # fecha con datos de ocupación, para que el JOIN de abajo tenga con
        # qué emparejar (mismo cálculo que usa /predecir, ver variables_estacionales.py).
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT MIN(fecha) AS min_fecha, MAX(fecha) AS max_fecha FROM ocupacion_hotelera "
                "WHERE ocupacion_porcentaje IS NOT NULL"
            )
            rango = cursor.fetchone()
        if rango and rango[0] and rango[1]:
            generar_variables_estacionales(rango[0], rango[1])

        query = """
            SELECT
                oh.fecha,
                oh.checkin_nacionales,
                oh.checkin_extranjeros,
                oh.pernoctaciones,
                oh.habitaciones_ocupadas,
                oh.tarifa_cobrada,
                oh.ocupacion_porcentaje,
                COALESCE(c.temperatura, 25.0) AS temperatura,
                COALESCE(c.humedad, 70.0) AS humedad,
                COALESCE(c.precipitacion, 0.0) AS precipitacion,
                COALESCE(f.total_dias, 1) AS total_dias,
                COALESCE(f.temporada, 'Media') AS temporada,
                ve.mes,
                ve.dia_semana,
                COALESCE(ve.es_festivo, 0) AS es_festivo,
                COALESCE(ve.es_temporada_alta, 0) AS es_temporada_alta,
                ve.semana_ano,
                COALESCE(ve.factor_estacional, 1.0) AS factor_estacional
            FROM ocupacion_hotelera oh
            LEFT JOIN clima c ON oh.fecha = c.fecha
            LEFT JOIN feriados f ON oh.fecha BETWEEN f.fecha_inicio AND f.fecha_fin
            LEFT JOIN variables_estacionales ve ON oh.fecha = ve.fecha
            WHERE oh.ocupacion_porcentaje IS NOT NULL
            ORDER BY oh.fecha ASC
        """

        df = pd.read_sql(query, connection)
        connection.close()

        print(f"Dataset cargado: {len(df)} registros")
        return df

    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise
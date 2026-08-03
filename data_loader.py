import pymysql
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def obtener_dataset_ml():
    """Obtiene el dataset completo para entrenar el modelo ML"""
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'ots_db'),
            charset='utf8mb4'
        )
        
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
                COALESCE(f.temporada, 'Media') AS temporada
            FROM ocupacion_hotelera oh
            LEFT JOIN clima c ON oh.fecha = c.fecha
            LEFT JOIN feriados f ON oh.fecha BETWEEN f.fecha_inicio AND f.fecha_fin
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
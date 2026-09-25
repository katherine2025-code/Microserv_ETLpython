"""
Generador de variables_estacionales: una tabla de apoyo con variables de
calendario (mes, dia_semana, es_festivo, es_temporada_alta, semana_ano,
factor_estacional) por fecha.

Existe para que el modelo de ML use EXACTAMENTE las mismas variables al
entrenar (data_loader.obtener_dataset_ml) y al predecir (app.py /predecir,
/predecir-rango), en vez de recalcularlas por separado en cada lugar (fuente
de bugs sutiles si ambos cálculos llegaran a divergir).

es_festivo se calcula contra la tabla 'feriados' (el insumo interno que ya
usa el modelo), y es_temporada_alta/factor_estacional contra 'temporadas'.
"""
from datetime import timedelta
from decimal import Decimal
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ots'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# Cuánto se refuerza el factor de temporada cuando la fecha además es feriado.
BOOST_FESTIVO = 0.1


def _calcular_features(fecha, feriados, temporadas):
    """feriados: [(fecha_inicio, fecha_fin)]; temporadas: [(fecha_inicio, fecha_fin, factor_ocupacion, nombre)]."""
    dia_semana = (fecha.weekday() + 1) % 7  # 0=Domingo ... 6=Sábado (igual que el modelo VariableEstacional)
    semana_ano = fecha.isocalendar()[1]

    es_festivo = any(ini <= fecha <= fin for ini, fin in feriados)

    factor = 1.0
    es_temporada_alta = False
    for ini, fin, factor_ocupacion, nombre in temporadas:
        if ini <= fecha <= fin:
            factor = float(factor_ocupacion)
            es_temporada_alta = (nombre == 'Alta')
            break

    if es_festivo:
        factor = round(factor + BOOST_FESTIVO, 2)

    return {
        'mes': fecha.month,
        'dia_semana': dia_semana,
        'es_festivo': es_festivo,
        'es_temporada_alta': es_temporada_alta,
        'semana_ano': semana_ano,
        'factor_estacional': factor
    }


def generar_variables_estacionales(fecha_inicio, fecha_fin):
    """Calcula y guarda (upsert) una fila por cada fecha del rango [fecha_inicio, fecha_fin]."""
    if fecha_fin < fecha_inicio:
        return 0

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT fecha_inicio, fecha_fin FROM feriados")
            feriados = [(r['fecha_inicio'], r['fecha_fin']) for r in cursor.fetchall()]

            cursor.execute(
                "SELECT fecha_inicio, fecha_fin, factor_ocupacion, nombre FROM temporadas WHERE activo = 1"
            )
            temporadas = [
                (r['fecha_inicio'], r['fecha_fin'], r['factor_ocupacion'], r['nombre'])
                for r in cursor.fetchall()
            ]

            filas = []
            actual = fecha_inicio
            while actual <= fecha_fin:
                f = _calcular_features(actual, feriados, temporadas)
                filas.append((
                    actual, f['mes'], f['dia_semana'], f['es_festivo'],
                    f['es_temporada_alta'], f['semana_ano'], f['factor_estacional']
                ))
                actual += timedelta(days=1)

            sql = """
            INSERT INTO variables_estacionales
            (fecha, mes, dia_semana, es_festivo, es_temporada_alta, semana_ano, factor_estacional, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                mes = VALUES(mes),
                dia_semana = VALUES(dia_semana),
                es_festivo = VALUES(es_festivo),
                es_temporada_alta = VALUES(es_temporada_alta),
                semana_ano = VALUES(semana_ano),
                factor_estacional = VALUES(factor_estacional),
                updated_at = NOW()
            """
            cursor.executemany(sql, filas)
        conn.commit()
        return len(filas)
    finally:
        conn.close()


def _a_numeros(fila):
    """pymysql devuelve las columnas DECIMAL (como factor_estacional) como decimal.Decimal, no
    float - eso llega a los modelos de ML con dtype 'object' en vez de numérico y XGBoost lo
    rechaza directamente ("must be int, float, bool or category"). Se convierte aquí, en el único
    lugar de donde salen estos datos, así ningún llamador tiene que acordarse de hacerlo."""
    if fila is None:
        return fila
    return {k: (float(v) if isinstance(v, Decimal) else v) for k, v in fila.items()}


def obtener_o_generar(fecha):
    """Devuelve las variables estacionales de una fecha puntual, generándola primero si falta."""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT mes, dia_semana, es_festivo, es_temporada_alta, semana_ano, factor_estacional "
                "FROM variables_estacionales WHERE fecha = %s",
                (fecha,)
            )
            fila = cursor.fetchone()
    finally:
        conn.close()

    if fila:
        return _a_numeros(fila)

    generar_variables_estacionales(fecha, fecha)

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT mes, dia_semana, es_festivo, es_temporada_alta, semana_ano, factor_estacional "
                "FROM variables_estacionales WHERE fecha = %s",
                (fecha,)
            )
            return _a_numeros(cursor.fetchone())
    finally:
        conn.close()

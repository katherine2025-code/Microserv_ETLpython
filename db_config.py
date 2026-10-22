"""
Configuración de la base de datos, en un solo lugar para todo el microservicio.

Se lee de variables de entorno (o del .env local, ver .env.example). La contraseña nunca va en
el código: si falta DB_PASSWORD se usa vacía (útil para un MariaDB/MySQL local con root sin
contraseña) y se avisa en el log.
"""
import logging
import os

import pymysql
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('DB_PASSWORD'):
    logging.getLogger(__name__).warning(
        "DB_PASSWORD no está configurada: se intentará conectar a la BD sin contraseña."
    )

# Parámetros de conexión (cursor por defecto: cada fila es una tupla)
DB_PARAMS = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ots'),
    'charset': 'utf8mb4',
}

# Igual, pero cada fila es un dict {columna: valor} (lo que usan app.py y los parsers)
DB_CONFIG = {**DB_PARAMS, 'cursorclass': pymysql.cursors.DictCursor}

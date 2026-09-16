"""
Parser dedicado para el formulario Kobo "Establecimientos de Alojamiento"
exportado en modo 'labels' (o con nombres de campo, según la versión).

Este formulario ha cambiado de estructura entre versiones (comprobado con
archivos reales de febrero y agosto 2026: mismo formulario, columnas en
posiciones distintas). Por eso las columnas clave NO se ubican por posición
fija sino por CONTENIDO:
  - la parroquia se detecta porque sus valores caen dentro de un conjunto
    conocido (Manglaralto, Colonche, Santa Elena, Salinas);
  - las habitaciones disponibles son la columna que le sigue inmediatamente
    (así viene diseñado el formulario en ambas versiones vistas);
  - la dirección se detecta porque la mayoría de sus valores contienen
    "calle" o "comuna";
  - el nombre del establecimiento es la columna que la precede inmediatamente;
  - los indicadores de turistas nacionales/extranjeros y el grupo repetido de
    ocupación (columnas ocultas con 'display:none') se detectan por nombre,
    que sí se mantuvo estable entre versiones.

Si no se puede anclar alguno de estos puntos, se reporta como advertencia de
"versión no reconocida" en vez de adivinar con una posición fija - así una
tercera versión futura del formulario no produce datos silenciosamente
incorrectos.

Cada fila del archivo = un envío de un establecimiento reportando la
ocupación de UN día (la fecha de la columna 'start'), con hasta 5
sub-registros (repeticiones 'fila', 'fila_1' .. 'fila_4') que se SUMAN para
obtener el total de pernoctaciones/habitaciones ocupadas de ese día, y se
PROMEDIA la tarifa cobrada entre los sub-registros que la reportaron.

El archivo no trae un id_hotel: trae el nombre real del establecimiento. El
hotel debe existir previamente en la tabla `hoteles` (creado desde el panel);
si no hay una coincidencia exacta por nombre normalizado, la fila se reporta
como error en vez de crear un hotel nuevo automáticamente.
"""
import re
import pandas as pd

MARCADOR_FORMULARIO = 'display:none'
REPETICIONES = ['fila', 'fila_1', 'fila_2', 'fila_3', 'fila_4']

# 'santa elena' y 'salinas' son también nombres de cantón (ambiguos); en
# cambio 'manglaralto' y 'colonche' SOLO existen como nombre de parroquia en
# esta zona, así que son la señal confiable para no confundir la columna de
# parroquia con la de cantón (que trae valores parecidos).
PARROQUIAS_DISTINTIVAS = {'manglaralto', 'colonche'}
PARROQUIAS_CONOCIDAS = PARROQUIAS_DISTINTIVAS | {'santa elena', 'salinas'}


def es_formulario_establecimientos(columnas) -> bool:
    """Detecta este formulario por la presencia de las columnas ocultas que
    Kobo genera al aplanar el grupo repetido de fechas de ocupación."""
    return any(MARCADOR_FORMULARIO in str(c) for c in columnas)


def _norm(texto) -> str:
    return re.sub(r'\s+', ' ', str(texto)).strip()


def _norm_nombre(texto) -> str:
    return _norm(texto).lower()


def _a_entero(valor):
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto == '' or texto.lower() == 'nan':
        return None
    try:
        return int(round(float(texto)))
    except (ValueError, TypeError):
        return None


def _parsear_fecha_start(valor):
    """Fechas ISO (YYYY-MM-DD..., como en el export .xlsx) nunca son
    ambiguas y NO deben pasar por dayfirst=True: pandas puede
    intercambiar mes y día en ese modo aunque el formato ya sea inequívoco
    (p.ej. '2026-08-09' -> 09 de agosto, mal interpretado como 08 de
    septiembre). Solo se usa dayfirst para formatos tipo D/M/YYYY (como en
    el export .csv de versiones anteriores del formulario)."""
    texto = str(valor).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}', texto):
        return pd.to_datetime(texto, errors='coerce', dayfirst=False)
    return pd.to_datetime(texto, errors='coerce', dayfirst=True)


def _buscar_columna(columnas, prefijo, sufijo):
    """Busca la columna oculta 'prefijo-sufijo' generada por Kobo, tolerando
    espacios extra al final del nombre real de la columna."""
    objetivo = f'{prefijo}-{sufijo}'
    for c in columnas:
        if objetivo in str(c):
            return c
    return None


def _detectar_columna_parroquia(df):
    """La columna de parroquia. Se prioriza la que contenga algún valor
    exclusivo de parroquia (nunca es nombre de cantón); si ninguna columna
    los trae, se usa como respaldo la que más coincidencias tenga con el
    conjunto ampliado (que sí puede confundirse con la columna de cantón)."""
    for col in df.columns:
        valores = df[col].dropna().astype(str).str.strip().str.lower()
        if valores.isin(PARROQUIAS_DISTINTIVAS).any():
            return col

    mejor_col, mejor_score = None, 0
    for col in df.columns:
        valores = df[col].dropna().astype(str).str.strip().str.lower()
        if valores.empty:
            continue
        coincidencias = valores.isin(PARROQUIAS_CONOCIDAS).sum()
        if coincidencias > mejor_score:
            mejor_score, mejor_col = coincidencias, col
    return mejor_col


def _detectar_columna_direccion(df, candidatas):
    """La columna de texto libre donde la mayoría de valores contienen
    'calle' o 'comuna' (patrón típico de una dirección en esta zona)."""
    mejor_col, mejor_score = None, 0.0
    for col in candidatas:
        valores = df[col].dropna().astype(str).str.lower()
        if len(valores) == 0:
            continue
        coincidencias = valores.str.contains('calle|comuna', regex=True).sum()
        proporcion = coincidencias / len(valores)
        if proporcion >= 0.3 and proporcion > mejor_score:
            mejor_score, mejor_col = proporcion, col
    return mejor_col


def detectar_columnas(df):
    """Ubica las columnas clave por contenido, tolerando que su posición
    cambie entre versiones del formulario. Devuelve un dict con los nombres
    de columna encontrados (None si no se pudo anclar alguno) y una lista de
    advertencias de versión no reconocida."""
    columnas = list(df.columns)
    advertencias = []

    col_parroquia = _detectar_columna_parroquia(df)
    col_habitaciones = None
    if col_parroquia:
        idx = columnas.index(col_parroquia)
        if idx + 1 < len(columnas):
            col_habitaciones = columnas[idx + 1]
    else:
        advertencias.append(
            "No se pudo identificar la columna de parroquia en este archivo "
            "(versión de formulario no reconocida); las habitaciones disponibles "
            "tampoco se pudieron ubicar."
        )

    candidatas_direccion = [c for c in columnas if 'pregunta' in str(c).lower()]
    col_direccion = _detectar_columna_direccion(df, candidatas_direccion)
    col_nombre = None
    if col_direccion:
        idx = columnas.index(col_direccion)
        if idx > 0:
            col_nombre = columnas[idx - 1]
    else:
        advertencias.append(
            "No se pudo identificar la columna de dirección en este archivo "
            "(versión de formulario no reconocida); el nombre del establecimiento "
            "tampoco se pudo ubicar."
        )

    col_nacionales = next(
        (c for c in columnas if 'turistas' in str(c).lower() and 'nacional' in str(c).lower()), None
    )
    col_extranjeros = next(
        (c for c in columnas if 'turistas' in str(c).lower() and 'extranjer' in str(c).lower()), None
    )
    if not col_nacionales or not col_extranjeros:
        advertencias.append(
            "No se encontraron las columnas de turistas nacionales/extranjeros; "
            "se guardarán como 0."
        )

    return {
        'nombre': col_nombre,
        'habitaciones_disponibles': col_habitaciones,
        'nacionales': col_nacionales,
        'extranjeros': col_extranjeros,
    }, advertencias


def extraer_registro_dia(row, columnas_todas, columnas_detectadas):
    """A partir de una fila cruda del archivo, arma el registro de ocupación
    de ESE día (sumando las hasta 5 repeticiones internas del envío)."""
    col_nombre = columnas_detectadas['nombre']
    col_hab = columnas_detectadas['habitaciones_disponibles']
    col_nac = columnas_detectadas['nacionales']
    col_ext = columnas_detectadas['extranjeros']

    nombre_establecimiento = _norm(row[col_nombre]) if col_nombre else ''
    habitaciones_disponibles = _a_entero(row[col_hab]) if col_hab else None
    nacionales = _a_entero(row[col_nac]) if col_nac else 0
    extranjeros = _a_entero(row[col_ext]) if col_ext else 0

    total_pernoctaciones = 0
    total_habitaciones_ocupadas = 0
    tarifas = []

    for rep in REPETICIONES:
        col_p = _buscar_columna(columnas_todas, rep, 'Pernoctaciones')
        col_h = _buscar_columna(columnas_todas, rep, 'Habitaciones ocupadas')
        col_t = _buscar_columna(columnas_todas, rep, 'Tarifa cobrada')

        pernoc = _a_entero(row[col_p]) if col_p else None
        hab = _a_entero(row[col_h]) if col_h else None
        tarifa = _a_entero(row[col_t]) if col_t else None

        if pernoc is not None:
            total_pernoctaciones += pernoc
        if hab is not None:
            total_habitaciones_ocupadas += hab
        if tarifa is not None:
            tarifas.append(tarifa)

    tarifa_promedio = round(sum(tarifas) / len(tarifas), 2) if tarifas else 0.0

    if habitaciones_disponibles and habitaciones_disponibles > 0:
        ocupacion_porcentaje = round(
            min(100.0, (total_habitaciones_ocupadas / habitaciones_disponibles) * 100), 2
        )
    else:
        ocupacion_porcentaje = 0.0

    return {
        'nombre_establecimiento': nombre_establecimiento,
        'checkin_nacionales': nacionales or 0,
        'checkin_extranjeros': extranjeros or 0,
        'pernoctaciones': total_pernoctaciones,
        'habitaciones_ocupadas': total_habitaciones_ocupadas,
        'habitaciones_disponibles': habitaciones_disponibles or 0,
        'tarifa_cobrada': tarifa_promedio,
        'ocupacion_porcentaje': ocupacion_porcentaje
    }


def procesar_establecimientos(df, get_db_connection):
    """Procesa el DataFrame crudo (tal cual lo entrega leer_archivo, sin pasar
    por el mapeo posicional genérico) e inserta en ocupacion_hotelera."""
    columnas_todas = list(df.columns)
    columnas_detectadas, advertencias = detectar_columnas(df)

    if not columnas_detectadas['nombre'] or not columnas_detectadas['habitaciones_disponibles']:
        return {
            'insertados': 0,
            'errores': len(df),
            'detalles': [
                "No se pudo procesar: esta versión del formulario no coincide con "
                "ninguna conocida (no se identificó el nombre del establecimiento "
                "o las habitaciones disponibles). Revisa el archivo o avisa para "
                "agregar soporte a esta versión."
            ],
            'advertencias': advertencias
        }

    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []

    try:
        cursor.execute("SELECT id_hotel, LOWER(TRIM(nombre)) AS nombre_norm FROM hoteles")
        hoteles_por_nombre = {}
        for fila in cursor.fetchall():
            hoteles_por_nombre.setdefault(fila['nombre_norm'], []).append(fila['id_hotel'])

        sql = """
        INSERT INTO ocupacion_hotelera
        (id_hotel, fecha, checkin_nacionales, checkin_extranjeros, pernoctaciones,
         habitaciones_ocupadas, habitaciones_disponibles, tarifa_cobrada,
         ocupacion_porcentaje, fuente_dato, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'Encuesta', NOW())
        """

        for index, row in df.iterrows():
            fila_num = int(index) + 2  # +1 por índice 0, +1 por la fila de encabezado
            try:
                fecha_dt = _parsear_fecha_start(row.get('start'))
                if pd.isna(fecha_dt):
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(f"Fila {fila_num}: no se pudo leer la fecha de 'start'")
                    continue
                fecha = fecha_dt.strftime('%Y-%m-%d')

                registro = extraer_registro_dia(row, columnas_todas, columnas_detectadas)
                nombre_norm = _norm_nombre(registro['nombre_establecimiento'])

                if not nombre_norm:
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(f"Fila {fila_num}: falta el nombre del establecimiento")
                    continue

                candidatos = hoteles_por_nombre.get(nombre_norm)
                if not candidatos:
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(
                            f"Fila {fila_num}: hotel '{registro['nombre_establecimiento']}' no encontrado. "
                            f"Regístralo primero en Hoteles."
                        )
                    continue
                if len(candidatos) > 1:
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(
                            f"Fila {fila_num}: hay {len(candidatos)} hoteles con el nombre "
                            f"'{registro['nombre_establecimiento']}' - no se puede saber cuál usar."
                        )
                    continue

                id_hotel = candidatos[0]
                cursor.execute(sql, (
                    id_hotel, fecha,
                    registro['checkin_nacionales'], registro['checkin_extranjeros'],
                    registro['pernoctaciones'], registro['habitaciones_ocupadas'],
                    registro['habitaciones_disponibles'], registro['tarifa_cobrada'],
                    registro['ocupacion_porcentaje']
                ))
                insertados += 1
            except Exception as e:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {fila_num}: {str(e)[:120]}")

        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()

    return {'insertados': insertados, 'errores': errores, 'detalles': detalles, 'advertencias': advertencias}

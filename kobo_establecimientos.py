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

Cada fila del archivo = un envío de un establecimiento reportando los días
del feriado (Fecha 1..5 = repeticiones 'fila', 'fila_1' .. 'fila_4'). Se
guarda UNA fila por envío, fechada con 'start', con los totales del período
y la cantidad de días reportados (ver extraer_registro_dia).

El archivo no trae un id_hotel: trae el nombre real del establecimiento. Si
ya existe un hotel con ese nombre (normalizado) en `hoteles`, se usa su
id_hotel; si no existe, se crea automáticamente con los datos disponibles
del propio formulario (nombre, parroquia, habitaciones) - así una carga real
no se bloquea por hoteles no registrados de antemano. Los hoteles creados así
quedan marcados en la advertencia de la respuesta para que el administrador
los revise/complete después desde el panel.
"""
import re
import pandas as pd

from etl_utils import _reparar_mojibake
from cantones import canton_de_parroquia, columnas_de

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
    # Reparar mojibake ANTES de usarlo como clave de búsqueda/creación de hotel: sin esto, el
    # mismo hotel podía terminar duplicado con un nombre distinto (con caracteres corruptos) cada
    # vez que el archivo llegaba con una codificación distinta a la de la carga anterior.
    return _reparar_mojibake(_norm(texto)).lower()


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


def _totales_ocupadas_por_fila(df, columnas_todas):
    """Suma, por fila, las 'Habitaciones ocupadas' de todas las repeticiones
    (días) presentes en el archivo, vectorizado, para validar la columna de
    habitaciones_disponibles antes de procesar fila por fila (la suma se compara
    contra capacidad × días). Devuelve (serie_totales, cantidad_de_repeticiones)."""
    columnas_rep = [_buscar_columna(columnas_todas, rep, 'Habitaciones ocupadas') for rep in REPETICIONES]
    columnas_rep = [c for c in columnas_rep if c is not None]
    if not columnas_rep:
        return pd.Series(0, index=df.index), 0
    total = sum(df[c].map(_a_entero).fillna(0) for c in columnas_rep)
    return total, len(columnas_rep)


CAPACIDAD_MAXIMA_PLAUSIBLE = 500  # ningún hotel de la zona tiene más habitaciones que esto


def _validar_columna_habitaciones(df, col, ocupadas_totales, max_repeticiones):
    """Comprueba si `col` es coherente como capacidad del hotel: en ningún día
    puede ocuparse más del 100% de las habitaciones, así que la suma de
    ocupadas de todas las repeticiones no puede superar capacidad × repeticiones
    (con 15% de margen por redondeos). Exige además un valor plausible como
    número de habitaciones (1-500): sin este límite, una columna de teléfonos
    u otro número grande "pasaría" la prueba de forma trivial (nunca la supera
    la ocupación). Devuelve (validables, correctas)."""
    tope = max(max_repeticiones, 1) * 1.15
    validables = correctas = 0
    for valor_col, ocupadas in zip(df[col], ocupadas_totales):
        capacidad = _a_entero(valor_col)
        if capacidad is None or not (0 < capacidad <= CAPACIDAD_MAXIMA_PLAUSIBLE) or ocupadas <= 0:
            continue
        validables += 1
        if ocupadas <= capacidad * tope:
            correctas += 1
    return validables, correctas


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
        candidata_posicional = columnas[idx + 1] if idx + 1 < len(columnas) else None

        # La columna de habitaciones disponibles no siempre queda justo después de la
        # parroquia (visto en el export de Salinas de agosto/2026, donde ahí cae el
        # nombre del establecimiento). Se valida el supuesto por CONTENIDO: ningún día
        # puede tener más habitaciones ocupadas que la capacidad declarada; si la
        # columna vecina no lo cumple, se busca entre las demás columnas numéricas
        # del bloque de identificación la que sí lo cumpla.
        ocupadas_totales, n_repeticiones = _totales_ocupadas_por_fila(df, columnas)
        col_habitaciones = candidata_posicional
        if candidata_posicional is not None and n_repeticiones > 0:
            validables, correctas = _validar_columna_habitaciones(
                df, candidata_posicional, ocupadas_totales, n_repeticiones)
            # validables < 3 también es sospechoso: si la columna fuera realmente la capacidad,
            # casi todas las filas con datos de ocupación deberían poder validarse contra ella
            # (valor numérico positivo). Pocas filas validables suele significar que la columna
            # ni siquiera es numérica (p. ej. cayó sobre el nombre del establecimiento).
            if validables < 3 or correctas / validables < 0.8:
                usadas = {col_parroquia, candidata_posicional}
                mejor_col, mejor_score, mejor_validables = None, 0.0, 0
                for c in columnas[idx:idx + 15]:  # bloque de identificación, no todo el archivo
                    if c in usadas or str(c).startswith(('_', 'meta/', 'start', 'end')):
                        continue
                    v, k = _validar_columna_habitaciones(df, c, ocupadas_totales, n_repeticiones)
                    if v >= 3 and k / v >= 0.8 and k / v > mejor_score:
                        mejor_col, mejor_score, mejor_validables = c, k / v, v
                if mejor_col is not None:
                    advertencias.append(
                        f"La columna de habitaciones disponibles no estaba donde se esperaba "
                        f"(versión de formulario distinta); se ubicó por contenido y se usaron "
                        f"esos valores en su lugar."
                    )
                    col_habitaciones = mejor_col
                else:
                    advertencias.append(
                        "No se pudo confirmar con certeza la columna de habitaciones disponibles "
                        "en este archivo (versión de formulario no reconocida); los porcentajes de "
                        "ocupación de este archivo pueden no ser confiables - revísalos."
                    )
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

    nacionales_dia, extranjeros_dia = _detectar_columnas_turistas_por_dia(columnas)
    if not nacionales_dia:
        advertencias.append(
            "No se encontraron las columnas de turistas por día (nacionales/extranjeros); "
            "se guardarán como 0."
        )

    return {
        'nombre': col_nombre,
        'parroquia': col_parroquia,
        'habitaciones_disponibles': col_habitaciones,
        'nacionales_dia': nacionales_dia,
        'extranjeros_dia': extranjeros_dia,
    }, advertencias


def _detectar_columnas_turistas_por_dia(columnas):
    """Las columnas con la CANTIDAD de turistas de cada día del feriado.

    'Pregunta de turistas /Nacionales' y '/Extranjeros' NO son cantidades: son las casillas
    de selección múltiple (1 = sí recibió, 0 = no), y guardarlas como turistas mostraba
    siempre "1" en la tabla. Las cantidades vienen justo después: un bloque 'Fecha 1..N' de
    nacionales seguido de otro 'Fecha 1..N' de extranjeros (Kobo repite la etiqueta, así que
    pandas las renombra a 'Fecha 1.1' en .xlsx o 'Fecha 1_10' en .csv). Se toman las columnas
    'Fecha ...' entre la última casilla de turistas y el grupo repetido de ocupación (que
    empieza con '**'), y se parten en dos mitades iguales. Devuelve ([], []) si no calza."""
    idx_casillas = [i for i, c in enumerate(columnas) if str(c).lower().startswith('pregunta de turistas')]
    if not idx_casillas:
        return [], []
    bloque = []
    for c in columnas[idx_casillas[-1] + 1:]:
        texto = str(c).strip()
        if texto.startswith('**') or MARCADOR_FORMULARIO in texto:
            break
        if texto.startswith('Fecha'):
            bloque.append(c)
    if not bloque or len(bloque) % 2:
        return [], []
    mitad = len(bloque) // 2
    return bloque[:mitad], bloque[mitad:]



def extraer_registro_dia(row, columnas_todas, columnas_detectadas):
    """A partir de una fila cruda del archivo, arma el registro de ocupación del envío.

    Cada repetición ('fila', 'fila_1'..'fila_4') es UN DÍA del feriado (Fecha 1..5 en el
    formulario), no un sub-registro del mismo día. Como el formulario no trae la fecha real
    de cada día, se guarda una fila por envío con los TOTALES del período (habitaciones
    ocupadas, pernoctaciones y turistas sumados sobre los días reportados) junto con
    dias_reportados. El % de ocupación es la fórmula MINTUR sobre el período:
    ocupadas / (capacidad × días). La tabla del frontend muestra el promedio por día
    (total / dias_reportados), que es lo comparable con la capacidad del hotel."""
    col_nombre = columnas_detectadas['nombre']
    col_parroquia = columnas_detectadas.get('parroquia')
    col_hab = columnas_detectadas['habitaciones_disponibles']

    nombre_establecimiento = _reparar_mojibake(_norm(row[col_nombre])) if col_nombre else ''
    parroquia = _reparar_mojibake(_norm(row[col_parroquia])).title() if col_parroquia and pd.notna(row[col_parroquia]) else None
    habitaciones_disponibles = _a_entero(row[col_hab]) if col_hab else None
    nacionales = sum(_a_entero(row[c]) or 0 for c in columnas_detectadas.get('nacionales_dia') or [])
    extranjeros = sum(_a_entero(row[c]) or 0 for c in columnas_detectadas.get('extranjeros_dia') or [])

    total_pernoctaciones = 0
    total_habitaciones_ocupadas = 0
    dias_reportados = 0
    dias_sobre_capacidad = 0
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
            dias_reportados += 1
            if habitaciones_disponibles and hab > habitaciones_disponibles:
                dias_sobre_capacidad += 1  # incoherente: más ocupadas que habitaciones (dato dudoso)
        if tarifa is not None:
            tarifas.append(tarifa)

    tarifa_promedio = round(sum(tarifas) / len(tarifas), 2) if tarifas else 0.0

    if habitaciones_disponibles and habitaciones_disponibles > 0 and dias_reportados > 0:
        # %OCC (MINTUR) del período reportado = habitaciones ocupadas / habitaciones disponibles, donde
        # las disponibles son capacidad x días. Antes se dividía la SUMA de varios días entre la
        # capacidad de UN solo día, y casi todo hotel quedaba topado en 100%. El tope de 100 solo
        # protege ante un dato incoherente; esos casos se cuentan en 'dias_sobre_capacidad'.
        ocupacion_porcentaje = round(
            min(100.0, total_habitaciones_ocupadas / (habitaciones_disponibles * dias_reportados) * 100), 2
        )
    else:
        ocupacion_porcentaje = 0.0

    return {
        'nombre_establecimiento': nombre_establecimiento,
        'parroquia': parroquia,
        'checkin_nacionales': nacionales,
        'checkin_extranjeros': extranjeros,
        # total_turistas: el backend lo calcula solo cuando se inserta por Sequelize (hooks
        # beforeCreate/beforeUpdate), pero el ETL inserta con SQL directo y esos hooks no corren.
        # Se calcula aquí para no dejarlo en 0.
        'total_turistas': nacionales + extranjeros,
        'pernoctaciones': total_pernoctaciones,
        'habitaciones_ocupadas': total_habitaciones_ocupadas,
        'habitaciones_disponibles': habitaciones_disponibles or 0,
        # habitaciones_totales: no es un dato que traiga el formulario aparte; se usa el mismo
        # valor de habitaciones_disponibles (la capacidad que el hotel reportó ese día).
        'habitaciones_totales': habitaciones_disponibles or 0,
        'tarifa_cobrada': tarifa_promedio,
        'ocupacion_porcentaje': ocupacion_porcentaje,
        'dias_reportados': dias_reportados,
        'dias_sobre_capacidad': dias_sobre_capacidad
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
    insertados, errores, omitidos, detalles = 0, 0, 0, []
    hoteles_creados = []

    try:
        cursor.execute("SELECT id_hotel, LOWER(TRIM(nombre)) AS nombre_norm FROM hoteles")
        hoteles_por_nombre = {}
        for fila in cursor.fetchall():
            hoteles_por_nombre.setdefault(fila['nombre_norm'], []).append(fila['id_hotel'])

        # Cantón (Santa Elena / Salinas) deducido de la parroquia, si la BD ya tiene la columna
        con_canton = 'canton' in columnas_de(cursor, 'hoteles')
        sql_crear_hotel = (
            "INSERT INTO hoteles (nombre, parroquia, habitaciones_totales, canton, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, NOW(), NOW())"
            if con_canton else
            "INSERT INTO hoteles (nombre, parroquia, habitaciones_totales, created_at, updated_at) "
            "VALUES (%s, %s, %s, NOW(), NOW())"
        )

        # uuid_kobo identifica el envío de forma única (columna "_uuid" del export de Kobo). Con
        # esto, subir dos veces el mismo archivo - o dos exportaciones que se superponen, como
        # pasó con Salinas en agosto - ya no duplica: la fila se omite si su _uuid ya está.
        con_uuid = 'uuid_kobo' in columnas_de(cursor, 'ocupacion_hotelera')
        uuids_existentes = set()
        if con_uuid:
            cursor.execute("SELECT uuid_kobo FROM ocupacion_hotelera WHERE uuid_kobo IS NOT NULL")
            uuids_existentes = {f['uuid_kobo'] for f in cursor.fetchall()}
        else:
            advertencias.append(
                "Reinicia el backend para activar la protección contra cargas duplicadas de este "
                "archivo (falta la columna ocupacion_hotelera.uuid_kobo)."
            )

        con_dias = 'dias_reportados' in columnas_de(cursor, 'ocupacion_hotelera')
        if not con_dias:
            advertencias.append(
                "Reinicia el backend para crear la columna ocupacion_hotelera.dias_reportados "
                "(sin ella no se guarda cuántos días reportó cada hotel)."
            )
        sql = """
        INSERT INTO ocupacion_hotelera
        (id_hotel, fecha, checkin_nacionales, checkin_extranjeros, total_turistas, pernoctaciones,
         habitaciones_ocupadas, habitaciones_disponibles, habitaciones_totales, tarifa_cobrada,
         ocupacion_porcentaje, fuente_dato, uuid_kobo, dias_reportados, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Encuesta', %s, %s, NOW())
        """ if con_dias else """
        INSERT INTO ocupacion_hotelera
        (id_hotel, fecha, checkin_nacionales, checkin_extranjeros, total_turistas, pernoctaciones,
         habitaciones_ocupadas, habitaciones_disponibles, habitaciones_totales, tarifa_cobrada,
         ocupacion_porcentaje, fuente_dato, uuid_kobo, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Encuesta', %s, NOW())
        """
        filas_sobre_capacidad = 0

        for index, row in df.iterrows():
            fila_num = int(index) + 2  # +1 por índice 0, +1 por la fila de encabezado
            try:
                uuid_fila = _norm(row.get('_uuid')) if con_uuid and pd.notna(row.get('_uuid')) else None
                if uuid_fila and uuid_fila in uuids_existentes:
                    omitidos += 1
                    continue

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
                    # No existe: se crea automáticamente con lo que trae el
                    # formulario, para no bloquear la carga por hoteles no
                    # registrados de antemano. El admin puede completar los
                    # demás datos (categoría, contacto, etc.) después.
                    valores_hotel = [
                        registro['nombre_establecimiento'],
                        registro['parroquia'],
                        registro['habitaciones_disponibles'] or 0
                    ]
                    if con_canton:
                        valores_hotel.append(canton_de_parroquia(registro['parroquia']))
                    cursor.execute(sql_crear_hotel, valores_hotel)
                    nuevo_id_hotel = cursor.lastrowid
                    hoteles_por_nombre[nombre_norm] = [nuevo_id_hotel]
                    candidatos = [nuevo_id_hotel]
                    hoteles_creados.append(registro['nombre_establecimiento'])
                if len(candidatos) > 1:
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(
                            f"Fila {fila_num}: hay {len(candidatos)} hoteles con el nombre "
                            f"'{registro['nombre_establecimiento']}' - no se puede saber cuál usar."
                        )
                    continue

                id_hotel = candidatos[0]
                valores = [
                    id_hotel, fecha,
                    registro['checkin_nacionales'], registro['checkin_extranjeros'], registro['total_turistas'],
                    registro['pernoctaciones'], registro['habitaciones_ocupadas'],
                    registro['habitaciones_disponibles'], registro['habitaciones_totales'],
                    registro['tarifa_cobrada'], registro['ocupacion_porcentaje'], uuid_fila
                ]
                if con_dias:
                    valores.append(registro['dias_reportados'])
                cursor.execute(sql, tuple(valores))
                if registro['dias_sobre_capacidad']:
                    filas_sobre_capacidad += 1
                if uuid_fila:
                    uuids_existentes.add(uuid_fila)  # protege también contra duplicados DENTRO del mismo archivo
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

    if hoteles_creados:
        advertencias.append(
            f"Se crearon {len(hoteles_creados)} hotel(es) nuevo(s) automáticamente a partir del archivo: "
            + ", ".join(hoteles_creados[:10])
            + ("..." if len(hoteles_creados) > 10 else "")
            + ". Revisa/completa sus datos en Hoteles (categoría, contacto, etc.)."
        )
    if filas_sobre_capacidad:
        advertencias.append(
            f"Consistencia: {filas_sobre_capacidad} hotel(es) reportaron algún día con más habitaciones "
            "ocupadas que su capacidad declarada (posible error de digitación). Se cargaron tal cual, "
            "con el porcentaje topado en 100%; conviene revisarlos."
        )
    if omitidos:
        advertencias.append(
            f"{omitidos} fila(s) ya estaban cargadas (mismo envío de Kobo) y se omitieron - no se duplicaron."
        )

    return {'insertados': insertados, 'errores': errores, 'omitidos': omitidos, 'detalles': detalles, 'advertencias': advertencias}

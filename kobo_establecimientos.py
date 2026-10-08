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
    presentes en el archivo (mismo cálculo que extraer_registro_dia, pero
    vectorizado para validar la columna de habitaciones_disponibles antes de
    procesar fila por fila). Devuelve (serie_totales, cantidad_de_repeticiones)."""
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


def _columnas_turistas_por_dia(columnas):
    """Columnas 'Fecha 1'..'Fecha N' de la pregunta de turistas: el formulario pregunta cuántos
    turistas nacionales y cuántos extranjeros se alojaron CADA día. Kobo las exporta como dos
    grupos seguidos con los mismos rótulos (el segundo con sufijo: 'Fecha 1.1' en xlsx, 'Fecha 1_10'
    en csv): primero el de nacionales, luego el de extranjeros. Los encabezados '##### Fecha N' del
    grupo de ocupación NO son de este bloque (empiezan con '#'), y puede haber otra 'Fecha N' suelta
    más adelante: por eso se toman solo tramos de columnas CONTIGUAS numeradas 1, 2, 3...
    Devuelve (nacionales, extranjeros), cada uno una lista de columnas en orden de día; listas
    vacías si el archivo no trae el bloque."""
    tramos, actual, anterior = [], [], None
    for pos, c in enumerate(columnas):
        m = re.match(r'^fecha\s*(\d)', str(c).strip().lower())
        n = int(m.group(1)) if m else None
        if n is not None and actual and anterior == (pos - 1, n - 1):
            actual.append(c)
        else:
            if actual:
                tramos.append(actual)
            actual = [c] if n == 1 else []
        anterior = (pos, n)
    if actual:
        tramos.append(actual)
    if len(tramos) < 2 or len(tramos[0]) != len(tramos[1]):
        return [], []
    return tramos[0], tramos[1]


def _personas_por_dia(row, cols_nac, cols_ext):
    """Turistas alojados (nacionales + extranjeros) en cada día reportado, o [] si no hay datos."""
    dias = []
    for c_nac, c_ext in zip(cols_nac, cols_ext):
        nac, ext = _a_entero(row[c_nac]), _a_entero(row[c_ext])
        if nac is None and ext is None:
            continue
        dias.append((nac or 0) + (ext or 0))
    return dias


PERSONAS_POR_HABITACION_MAX = 10  # dormitorios compartidos de hostal: hasta ~10 camas por habitación


def _detectar_columna_plazas(df, columnas, col_hab, cols_nac, cols_ext):
    """La capacidad en PLAZAS (personas que el establecimiento puede alojar) es una de las preguntas
    numéricas que siguen a la de habitaciones; su posición cambió entre versiones del formulario
    (Santa Elena: la inmediata siguiente; Salinas: una más allá). Se elige por CONTENIDO:
      - una habitación aloja al menos a 1 persona y como mucho a ~10 (habitaciones <= plazas <= 10 x);
      - en ningún día puede haber más turistas alojados que plazas (15% de margen).
    Devuelve None si ninguna candidata es coherente: mejor sin tasa por plazas que con una inventada."""
    if col_hab is None:
        return None
    idx = columnas.index(col_hab)
    mejor_col, mejor_score = None, 0.0
    for c in columnas[idx + 1:idx + 4]:
        if str(c).startswith(('_', 'meta/')) or c in cols_nac or c in cols_ext:
            continue
        validables = coherentes = caben = con_personas = 0
        for _, row in df.iterrows():
            hab, plazas = _a_entero(row[col_hab]), _a_entero(row[c])
            if not hab or not plazas or hab <= 0 or plazas <= 0:
                continue
            validables += 1
            if hab <= plazas <= hab * PERSONAS_POR_HABITACION_MAX:
                coherentes += 1
            personas = _personas_por_dia(row, cols_nac, cols_ext)
            if personas:
                con_personas += 1
                if max(personas) <= plazas * 1.15:
                    caben += 1
        if validables < 3 or coherentes / validables < 0.6:
            continue
        score = coherentes / validables + (caben / con_personas if con_personas else 0)
        if score > mejor_score:
            mejor_col, mejor_score = c, score
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

    nacionales_dia, extranjeros_dia = _columnas_turistas_por_dia(columnas)
    col_plazas = _detectar_columna_plazas(df, columnas, col_habitaciones, nacionales_dia, extranjeros_dia)
    if col_habitaciones and not col_plazas:
        advertencias.append(
            "No se identificó con certeza la capacidad en plazas (personas) de los establecimientos; "
            "la tasa de ocupación por plazas de este archivo queda sin calcular."
        )

    return {
        'nombre': col_nombre,
        'parroquia': col_parroquia,
        'habitaciones_disponibles': col_habitaciones,
        'plazas_disponibles': col_plazas,
        'nacionales': col_nacionales,
        'extranjeros': col_extranjeros,
        'nacionales_dia': nacionales_dia,
        'extranjeros_dia': extranjeros_dia,
    }, advertencias


def extraer_registro_dia(row, columnas_todas, columnas_detectadas):
    """A partir de una fila cruda del archivo, arma el registro de ocupación
    de ESE día (sumando las hasta 5 repeticiones internas del envío)."""
    col_nombre = columnas_detectadas['nombre']
    col_parroquia = columnas_detectadas.get('parroquia')
    col_hab = columnas_detectadas['habitaciones_disponibles']
    col_nac = columnas_detectadas['nacionales']
    col_ext = columnas_detectadas['extranjeros']

    nombre_establecimiento = _reparar_mojibake(_norm(row[col_nombre])) if col_nombre else ''
    parroquia = _reparar_mojibake(_norm(row[col_parroquia])).title() if col_parroquia and pd.notna(row[col_parroquia]) else None
    col_plazas = columnas_detectadas.get('plazas_disponibles')
    cols_nac_dia = columnas_detectadas.get('nacionales_dia') or []
    cols_ext_dia = columnas_detectadas.get('extranjeros_dia') or []
    habitaciones_disponibles = _a_entero(row[col_hab]) if col_hab else None
    plazas_disponibles = _a_entero(row[col_plazas]) if col_plazas else None
    if plazas_disponibles and habitaciones_disponibles and not (
            habitaciones_disponibles <= plazas_disponibles <= habitaciones_disponibles * PERSONAS_POR_HABITACION_MAX):
        plazas_disponibles = None  # menos plazas que habitaciones (o absurdo): error de digitación
    if cols_nac_dia:
        # Las columnas 'nacionales'/'extranjeros' son las casillas de "¿qué tipo de turistas
        # recibió?" (valen 1/0), no cantidades: la cantidad real está en el bloque por día.
        nacionales = sum(_a_entero(row[c]) or 0 for c in cols_nac_dia)
        extranjeros = sum(_a_entero(row[c]) or 0 for c in cols_ext_dia)
    else:
        nacionales = _a_entero(row[col_nac]) if col_nac else 0
        extranjeros = _a_entero(row[col_ext]) if col_ext else 0
    personas_por_dia = _personas_por_dia(row, cols_nac_dia, cols_ext_dia)

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

    # Tasa de ocupación por PLAZAS = personas alojadas / (plazas x días): qué tan lleno está el
    # establecimiento respecto a cuántas personas puede alojar (no es lo mismo que la tasa por
    # habitaciones: una habitación de 4 camas con 1 huésped cuenta como ocupada, pero solo usa 1
    # de 4 plazas). Personas alojadas = turistas nacionales + extranjeros de cada día; si el archivo
    # no trae ese bloque, las pernoctaciones del día (cada persona que duerme esa noche = 1).
    if personas_por_dia:
        personas_alojadas, dias_personas = sum(personas_por_dia), len(personas_por_dia)
    else:
        personas_alojadas, dias_personas = total_pernoctaciones, dias_reportados
    if plazas_disponibles and plazas_disponibles > 0 and dias_personas > 0:
        ocupacion_plazas_porcentaje = round(
            min(100.0, personas_alojadas / (plazas_disponibles * dias_personas) * 100), 2
        )
    else:
        ocupacion_plazas_porcentaje = None

    return {
        'nombre_establecimiento': nombre_establecimiento,
        'parroquia': parroquia,
        'checkin_nacionales': nacionales or 0,
        'checkin_extranjeros': extranjeros or 0,
        # total_turistas: el backend lo calcula solo cuando se inserta por Sequelize (hooks
        # beforeCreate/beforeUpdate), pero el ETL inserta con SQL directo y esos hooks no corren.
        # Se calcula aquí para no dejarlo en 0.
        'total_turistas': (nacionales or 0) + (extranjeros or 0),
        'pernoctaciones': total_pernoctaciones,
        'habitaciones_ocupadas': total_habitaciones_ocupadas,
        'habitaciones_disponibles': habitaciones_disponibles or 0,
        # habitaciones_totales: no es un dato que traiga el formulario aparte; se usa el mismo
        # valor de habitaciones_disponibles (la capacidad que el hotel reportó ese día).
        'habitaciones_totales': habitaciones_disponibles or 0,
        'tarifa_cobrada': tarifa_promedio,
        'ocupacion_porcentaje': ocupacion_porcentaje,
        'plazas_disponibles': plazas_disponibles,
        'ocupacion_plazas_porcentaje': ocupacion_plazas_porcentaje,
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
        con_plazas = 'ocupacion_plazas_porcentaje' in columnas_de(cursor, 'ocupacion_hotelera')
        if not con_plazas:
            advertencias.append(
                "Reinicia el backend para crear las columnas de ocupación por plazas "
                "(sin ellas solo se guarda la tasa de ocupación por habitaciones)."
            )
        campos = ['id_hotel', 'fecha', 'checkin_nacionales', 'checkin_extranjeros', 'total_turistas',
                  'pernoctaciones', 'habitaciones_ocupadas', 'habitaciones_disponibles',
                  'habitaciones_totales', 'tarifa_cobrada', 'ocupacion_porcentaje', 'uuid_kobo']
        if con_dias:
            campos.append('dias_reportados')
        if con_plazas:
            campos += ['plazas_disponibles', 'ocupacion_plazas_porcentaje']
        sql = (f"INSERT INTO ocupacion_hotelera ({', '.join(campos)}, fuente_dato, created_at) "
               f"VALUES ({', '.join(['%s'] * len(campos))}, 'Encuesta', NOW())")
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
                if con_plazas:
                    valores += [registro['plazas_disponibles'], registro['ocupacion_plazas_porcentaje']]
                cursor.execute(sql, tuple(valores))
                if registro['plazas_disponibles']:
                    # La capacidad en plazas es un dato del establecimiento: se completa en su ficha
                    # si todavía no la tiene (no pisa un valor que el administrador haya cargado).
                    cursor.execute(
                        "UPDATE hoteles SET plazas_disponibles = %s WHERE id_hotel = %s AND plazas_disponibles IS NULL",
                        (registro['plazas_disponibles'], id_hotel)
                    )
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

"""
Parser dedicado para el formulario Kobo "Recolección de datos Turismo"
(encuestas a turistas).

Mucho más irregular que el de Establecimientos (ver kobo_establecimientos.py):
coexisten al menos DOS versiones del formulario (columna `__version__` en el
export), y una misma pregunta puede caer en una posición de columna distinta
según la versión. Además, casi todas las preguntas quedaron exportadas con
el nombre genérico de Kobo "Pregunta", "Pregunta.1"... (pandas las distingue
solo por el sufijo, sin texto real).

Por eso aquí NO se usa "ancla + posición relativa fija" para la mayoría de
los campos: cada uno se identifica por CONTENIDO, recorriendo todas las
columnas de respuesta y quedándose con la primera que calce con el conjunto
de valores conocido de esa pregunta. Así funciona sin importar en qué
versión del formulario ni en qué columna haya caído la respuesta.

Excepciones que si usan posición:
  - nivel_satisfaccion: en todas las exportaciones vistas tiene un nombre de
    columna literal (no genérico), así que se ubica por ese texto.
  - probabilidad_retorno: NINGUNA exportación disponible conservó el texto
    real de esta pregunta. Se usa la columna INMEDIATAMENTE SIGUIENTE a
    nivel_satisfaccion (mismo orden en ambas versiones del formulario).
    *** ESTA SUPOSICIÓN QUEDÓ PENDIENTE DE CONFIRMAR contra el diseño
    original del formulario en KoboToolbox (ver PROBABILIDAD_RETORNO_NOTA). ***

fecha_encuesta viene de la columna 'start' (fecha de inicio del envío).
ciudad_residencia no existe en este formulario - se guarda como
"Prefiero no responder" siempre (misma convención usada para nulos de texto).
"""
import re
import pandas as pd

MARCADOR_FORMULARIO = 'nivel de satisfaccion'

PROBABILIDAD_RETORNO_NOTA = (
    "El mapeo de 'probabilidad_retorno' es una suposición (columna siguiente a "
    "nivel_satisfaccion) pendiente de confirmar contra el diseño original del "
    "formulario en KoboToolbox - ver docstring de kobo_encuestas_turismo.py."
)

TOKENS_NULOS = {'', 'nan', 'null', 'none', 'n/a', 'na', '-'}

# Prefijos/sub-strings de columnas que NO son respuestas de opción (metadatos
# de Kobo, GPS, texto libre que podría dar falsos positivos por substring).
_EXCLUIR_PREFIJOS = ('_', 'meta/', 'start', 'end', 'Unnamed', 'Ubicacion')
_EXCLUIR_SUBSTRINGS = ('encuestador', 'especifique')


def es_formulario_turismo(columnas) -> bool:
    """Detecta este formulario por el texto literal de la pregunta de
    satisfacción general, presente en todas las exportaciones vistas."""
    return any(MARCADOR_FORMULARIO in _norm(c).lower() for c in columnas)


def _norm(texto) -> str:
    return re.sub(r'\s+', ' ', str(texto)).strip()


def _valor_valido(valor) -> bool:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return False
    return _norm(valor).lower() not in TOKENS_NULOS


def _parsear_fecha_start(valor):
    """Mismo criterio que kobo_establecimientos.py: las fechas ISO
    (YYYY-MM-DD..., como en 'start') no deben pasar por dayfirst=True o
    pandas puede intercambiar día y mes aunque el formato ya sea inequívoco."""
    texto = str(valor).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}', texto):
        return pd.to_datetime(texto, errors='coerce', dayfirst=False, utc=True)
    return pd.to_datetime(texto, errors='coerce', dayfirst=True, utc=True)


def _columnas_pregunta(columnas):
    """Columnas candidatas a contener una respuesta de opción: excluye
    metadatos de Kobo, GPS, columnas de sub-opciones de selección múltiple
    (tienen '/' en el nombre, ej. 'Pregunta/Ciudad...') y texto libre que
    podría producir falsos positivos (nombre del encuestador, campos
    "especifique")."""
    resultado = []
    for c in columnas:
        texto = str(c)
        if texto.startswith(_EXCLUIR_PREFIJOS):
            continue
        if '/' in texto:
            continue
        if any(s in texto.lower() for s in _EXCLUIR_SUBSTRINGS):
            continue
        resultado.append(c)
    return resultado


def _clasificar(row, columnas, mapeo, ambiguo_si_multiple=False, valor_ambiguo='Otro'):
    """Recorre las columnas dadas y devuelve el valor mapeado de la primera
    celda que calce con alguna clave de `mapeo` (substring, insensible a
    mayúsculas/acentos simples). Si `ambiguo_si_multiple` y una misma celda
    calza con más de una clave distinta (respuesta corrupta/combinada, visto
    en los datos reales para género), devuelve `valor_ambiguo` en vez de
    escoger una al azar."""
    for col in columnas:
        valor = row.get(col)
        if not _valor_valido(valor):
            continue
        texto = _norm(valor).lower()
        coincidencias = {mapeado for clave, mapeado in mapeo.items() if clave in texto}
        if not coincidencias:
            continue
        if ambiguo_si_multiple and len(coincidencias) > 1:
            return valor_ambiguo
        return next(iter(coincidencias))
    return None


GENERO_MAPEO = {
    'hombre': 'Masculino',
    'mujer': 'Femenino',
    'persona no binaria': 'Otro',
    'prefiero no responder': 'Prefiero no responder',
}

EDAD_MAPEO = {
    'menor de 18': 16,
    '18 a 24': 21,
    '25 a 34': 29,
    '35 a 44': 39,
    '45 a 54': 49,
    '55 a 64': 59,
    '65 años o más': 70,
    '65 anos o mas': 70,
}

PAIS_MAPEO = {
    'ecuador': 'Ecuador',
    'colombia': 'Colombia',
    'estados unidos': 'Estados Unidos',
    'perú': 'Perú',
    'peru': 'Perú',
    'españa': 'España',
    'espana': 'España',
    'canadá': 'Canadá',
    'canada': 'Canadá',
    'prefiero no responder': 'Prefiero no responder',
}

MOTIVO_MAPEO = {
    'ocio': 'Ocio / Vacaciones',
    'vacaciones': 'Ocio / Vacaciones',
    'visita a familiares': 'Visita a familiares o amigos',
    'negocios': 'Negocios, trabajo o eventos corporativos',
    'asistencia a eventos': 'Asistencia a eventos',
    'estudios': 'Estudios o formación',
    'formación': 'Estudios o formación',
    'salud': 'Salud o tratamiento médico',
    'trámite': 'Trámite personal o gestiones',
    'tramite': 'Trámite personal o gestiones',
}

NOCHES_MAPEO = {
    '0 noches': 0,
    '1 noche': 1,
    '2 noches': 2,
    '3 noches': 3,
    '4 a 6 noches': 5,
    '7 noches o más': 7,
    '7 noches o mas': 7,
}

GASTO_MAPEO = {
    '0 (sin gasto': 0.0,
    'menos de $20': 10.0,
    'menos de 20': 10.0,
    'de $21 a $50': 35.0,
    'de $51 a $100': 75.0,
    'más de $200': 250.0,
    'mas de $200': 250.0,
    'más de $100': 150.0,
    'mas de $100': 150.0,
}

NIVEL_EDUCATIVO_MAPEO = {
    'ninguno': 'Ninguno',
    'educación primaria': 'Educación Primaria',
    'educacion primaria': 'Educación Primaria',
    'educación secundaria': 'Educación Secundaria',
    'educacion secundaria': 'Educación Secundaria',
    'formación técnica': 'Formación técnica o tecnológica',
    'formacion tecnica': 'Formación técnica o tecnológica',
    'educación universitaria': 'Educación universitaria',
    'educacion universitaria': 'Educación universitaria',
    'posgrado': 'Posgrado',
    'prefiero no responder': 'Prefiero no responder',
}

OCUPACION_MAPEO = {
    'sector privado': 'Empleado/a sector privado',
    'sector público': 'Empleado/a sector público',
    'sector publico': 'Empleado/a sector público',
    'estudiante': 'Estudiante',
    'independiente': 'Trabajador/a independiente',
    'emprendedor': 'Emprendedor/a',
    'ama/o de casa': 'Ama/o de casa',
    'ama de casa': 'Ama/o de casa',
    'jubilado': 'Jubilado/a',
    'búsqueda de empleo': 'Desempleado/a en búsqueda de empleo',
    'busqueda de empleo': 'Desempleado/a en búsqueda de empleo',
    'no busca empleo': 'Desempleado/a no busca empleo',
    'prefiero no responder': 'Prefiero no responder',
}

FRECUENCIA_MAPEO = {
    'nunca': 'Nunca',
    '1 vez': '1 vez',
    '2-3 veces': '2-3 veces',
    '4-5 veces': '4-5 veces',
    'más de 5 veces': 'Más de 5 veces',
    'mas de 5 veces': 'Más de 5 veces',
}


def _extraer_tamano_grupo(row, columnas):
    for col in columnas:
        valor = row.get(col)
        if not _valor_valido(valor):
            continue
        texto = _norm(valor).lower()
        if 'solo yo' in texto or 'viajé solo' in texto or 'viaje solo' in texto:
            return 1
        m = re.search(r'(\d+)\s*personas?', texto)
        if m:
            return int(m.group(1))
    return None


def _extraer_pais(row, columnas):
    directo = _clasificar(row, columnas, PAIS_MAPEO)
    if directo:
        return directo
    # "Otra (especifique)": el país real queda en la columna de texto libre
    # 'Por favor especifique el pais'.
    for col in columnas:
        valor = row.get(col)
        if _valor_valido(valor) and 'otra' in _norm(valor).lower():
            col_especifique = next(
                (c for c in row.index if 'especifique el pais' in str(c).lower()), None
            )
            if col_especifique is not None and _valor_valido(row.get(col_especifique)):
                return _norm(row[col_especifique]).title()
            return 'Otro'
    return None


def extraer_registro(row, columnas_pregunta, idx_nivel_satisfaccion):
    """A partir de una fila cruda, clasifica cada campo por contenido."""
    genero = _clasificar(row, columnas_pregunta, GENERO_MAPEO, ambiguo_si_multiple=True)
    edad = _clasificar(row, columnas_pregunta, EDAD_MAPEO)
    pais = _extraer_pais(row, columnas_pregunta)
    motivo = _clasificar(row, columnas_pregunta, MOTIVO_MAPEO)
    noches = _clasificar(row, columnas_pregunta, NOCHES_MAPEO)
    gasto = _clasificar(row, columnas_pregunta, GASTO_MAPEO)
    nivel_educativo = _clasificar(row, columnas_pregunta, NIVEL_EDUCATIVO_MAPEO)
    ocupacion = _clasificar(row, columnas_pregunta, OCUPACION_MAPEO)
    frecuencia = _clasificar(row, columnas_pregunta, FRECUENCIA_MAPEO)
    tamano_grupo = _extraer_tamano_grupo(row, columnas_pregunta)

    nivel_satisfaccion = None
    probabilidad_retorno = None
    if idx_nivel_satisfaccion is not None:
        valor_sat = row.get(columnas_pregunta[idx_nivel_satisfaccion])
        if _valor_valido(valor_sat):
            m = re.search(r'\d', _norm(valor_sat))
            if m:
                nivel_satisfaccion = int(m.group())

        idx_retorno = idx_nivel_satisfaccion + 1
        if idx_retorno < len(columnas_pregunta):
            valor_ret = row.get(columnas_pregunta[idx_retorno])
            if _valor_valido(valor_ret):
                m = re.search(r'\d', _norm(valor_ret))
                if m:
                    probabilidad_retorno = int(m.group())

    return {
        'genero': genero or 'Prefiero no responder',
        'edad': edad if edad is not None else 0,
        'pais_residencia': pais or 'Prefiero no responder',
        'nivel_educativo': nivel_educativo or 'Prefiero no responder',
        'ocupacion': ocupacion or 'Prefiero no responder',
        'tamano_grupo': tamano_grupo if tamano_grupo is not None else 0,
        'frecuencia_visitas': frecuencia or 'Prefiero no responder',
        'motivo_visita': motivo or 'Prefiero no responder',
        'noches_estadia': noches if noches is not None else 0,
        'gasto_total': gasto if gasto is not None else 0.0,
        'nivel_satisfaccion': nivel_satisfaccion if nivel_satisfaccion is not None else 0,
        'probabilidad_retorno': probabilidad_retorno if probabilidad_retorno is not None else 0,
    }


def procesar_encuestas_turismo(df, get_db_connection):
    """Procesa el DataFrame crudo (tal cual lo entrega leer_archivo) e
    inserta en encuestas_turisticas."""
    columnas_pregunta = _columnas_pregunta(df.columns)
    idx_nivel_satisfaccion = next(
        (i for i, c in enumerate(columnas_pregunta) if MARCADOR_FORMULARIO in _norm(c).lower()),
        None
    )

    advertencias = []
    if idx_nivel_satisfaccion is None:
        advertencias.append(
            "No se pudo ubicar la pregunta de satisfacción general; "
            "nivel_satisfaccion y probabilidad_retorno quedarán en 0."
        )
    else:
        advertencias.append(PROBABILIDAD_RETORNO_NOTA)

    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, detalles = 0, 0, []

    try:
        sql = """
        INSERT INTO encuestas_turisticas
        (fecha_encuesta, genero, edad, pais_residencia, ciudad_residencia,
         nivel_educativo, ocupacion, tamano_grupo, frecuencia_visitas,
         motivo_visita, noches_estadia, gasto_total, nivel_satisfaccion,
         probabilidad_retorno, fecha_registro)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
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

                registro = extraer_registro(row, columnas_pregunta, idx_nivel_satisfaccion)

                cursor.execute(sql, (
                    fecha, registro['genero'], registro['edad'], registro['pais_residencia'],
                    'Prefiero no responder',  # ciudad_residencia: no existe en este formulario
                    registro['nivel_educativo'], registro['ocupacion'], registro['tamano_grupo'],
                    registro['frecuencia_visitas'], registro['motivo_visita'],
                    registro['noches_estadia'], registro['gasto_total'],
                    registro['nivel_satisfaccion'], registro['probabilidad_retorno']
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

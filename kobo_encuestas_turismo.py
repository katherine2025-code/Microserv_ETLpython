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
import json
import re
import pandas as pd

from etl_utils import _reparar_mojibake
from cantones import columnas_de

MARCADOR_FORMULARIO = 'nivel de satisfaccion'

# Los 15 lugares que ofrece la pregunta "¿Qué lugares piensa visitar...?" del formulario de Kobo
# (selección múltiple). MISMA lista que ots_front/.../lugares.service.ts y el mapa del dashboard
# (backend_ots/src/seeds/lugaresSeed.js) - si se agrega un lugar nuevo a la pregunta, agregarlo
# también ahí para que el mapa lo reconozca.
LUGARES_CONOCIDOS = [
    'Montañita', 'Olón', 'Manglaralto', 'Ayampe', 'Ayangue', 'Salinas', 'Chipipe',
    'La Chocolatera', 'Canoa', 'San Pablo', 'Playa Ballenita', 'La Lobería',
    'Cerro El Morro', 'San Lorenzo', 'San Pedro'
]

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


_SECCION_RE = re.compile(r'^\s*secc', re.IGNORECASE)


def _sin_grupo_seccion(columna) -> str:
    """Quita el prefijo de GRUPO de sección que Kobo antepone a CADA pregunta en
    las exportaciones con etiquetas (p. ej. 'SECCIÓN A. PERFIL.../1. ¿Qué edad...').
    Ese primer segmento es el nombre de la sección del formulario, NO una sub-opción
    de selección múltiple. Sin quitarlo, la regla de '/' de _columnas_pregunta
    descartaría TODAS las columnas (todas empiezan con 'SECCIÓN .../') y la encuesta
    entera entraría vacía a la base - que es justo lo que pasaba con los exports
    reales de Kobo (abril/mayo/agosto quedaron con todos los campos en su valor por
    defecto). Lo que queda después del prefijo se evalúa normalmente: si todavía tiene
    '/', es una verdadera sub-opción/matriz (Pregunta_018/..., satisfacción).
    'SECCIÓN' puede venir con mojibake ('SECCI�N'), por eso solo se exige que el primer
    segmento empiece por 'secc'."""
    s = _norm(columna)
    partes = s.split('/')
    if len(partes) > 1 and _SECCION_RE.match(partes[0]):
        return '/'.join(partes[1:]).strip()
    return s


def _columnas_pregunta(columnas):
    """Columnas candidatas a contener una respuesta de opción: excluye
    metadatos de Kobo, GPS, columnas de sub-opciones de selección múltiple
    (tienen '/' en el nombre, ej. 'Pregunta/Ciudad...') y texto libre que
    podría producir falsos positivos (nombre del encuestador, campos
    "especifique"). Se evalúa sobre el nombre SIN el prefijo de grupo de sección
    (ver _sin_grupo_seccion), pero se devuelve la columna ORIGINAL para poder
    leer su valor con row.get()."""
    resultado = []
    for c in columnas:
        texto = _sin_grupo_seccion(c)
        if texto.startswith(_EXCLUIR_PREFIJOS):
            continue
        if '/' in texto:
            continue
        if any(s in texto.lower() for s in _EXCLUIR_SUBSTRINGS):
            continue
        resultado.append(c)
    return resultado


def _detectar_col_satisfaccion(columnas):
    """La pregunta de satisfacción general conserva su texto literal ('nivel de
    satisfaccion') en todas las exportaciones, pero en el formato con secciones
    cae bajo un sub-grupo (…/Pregunta_019/…) con dos '/', así que _columnas_pregunta
    la descarta. Se ubica aquí sobre TODAS las columnas, por su texto."""
    for c in columnas:
        if MARCADOR_FORMULARIO in _norm(c).lower():
            return c
    return None


def _detectar_col_probabilidad(columnas, columnas_pregunta, col_satisfaccion):
    """probabilidad_retorno no conserva su texto real en ninguna exportación: se
    asume la PRIMERA columna-pregunta candidata que aparece después de la de
    satisfacción (mismo criterio que la versión anterior, ver PROBABILIDAD_RETORNO_NOTA)."""
    if col_satisfaccion is None:
        return None
    cols = list(columnas)
    try:
        pos_sat = cols.index(col_satisfaccion)
    except ValueError:
        return None
    for c in columnas_pregunta:
        if cols.index(c) > pos_sat:
            return c
    return None


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

# ============================================================================
# Preguntas "nuevas" de opción única que SÍ se pueden identificar con certeza
# (añadidas 2026-10-12). A diferencia de las de arriba (que se clasifican con
# _clasificar recorriendo TODAS las columnas), estas se extraen por COLUMNA:
# primero se localiza la columna de cada pregunta (por `firmas`, valores que solo
# aparecen en ESA pregunta), y luego se lee el valor de esa columna fila por fila.
# Así se evita el choque entre preguntas que comparten una opción (p. ej.
# "Vehículo propio" está tanto en transporte de llegada como en el de destino):
# un recorrido global por contenido le asignaría a las dos la misma columna.
#
# `mapeo` es una lista ORDENADA de (substring, etiqueta): se devuelve la etiqueta
# del PRIMER substring contenido en el valor, por eso van de lo más específico a lo
# más general (ej. 'muy fácil' antes que 'fácil').
# NO se incluyen aquí (quedaron pendientes por no ser identificables con certeza):
# los 5 gastos por rubro (columnas B-F sin etiqueta) y las escalas de "acuerdo".
PREGUNTAS_COLUMNA = [
    {
        'campo': 'con_quien_viaja',
        'firmas': ['en pareja', 'con amigos', 'con familia', 'tour organizado'],
        'mapeo': [('tour organizado', 'Tour organizado'), ('con amigos', 'Con amigos/as'),
                  ('con familia', 'Con familia (incluye niños/as)'), ('en pareja', 'En pareja'),
                  ('solo/a', 'Solo/a'), ('otro', 'Otro')],
    },
    {
        'campo': 'facilidad_informacion',
        'firmas': ['muy fácil', 'ni fácil ni difícil', 'muy difícil', 'no busqué información'],
        'mapeo': [('no busqué', 'No busqué información'), ('muy fácil', 'Muy fácil'),
                  ('ni fácil ni difícil', 'Ni fácil ni difícil'), ('muy difícil', 'Muy difícil'),
                  ('difícil', 'Difícil'), ('fácil', 'Fácil')],
    },
    {
        'campo': 'como_organizo',
        'firmas': ['cuenta propia', 'paquete turístico', 'ayuda de familiares', 'agencia de viajes'],
        'mapeo': [('cuenta propia', 'Por cuenta propia'), ('paquete turístico', 'Por medio de un paquete turístico'),
                  ('redes sociales o grupos', 'A través de redes sociales o grupos de viaje'),
                  ('ayuda de familiares', 'Con ayuda de familiares o amigos en el destino'),
                  ('agencia de viajes en línea', 'A través de una agencia de viajes en línea (OTA)'),
                  ('agencia de viajes presencial', 'A través de una agencia de viajes presencial'),
                  ('otro', 'Otro')],
    },
    {
        'campo': 'anticipacion',
        'firmas': ['1 a 3 días', '4 a 7 días', '1 a 2 semanas', '3 a 4 semanas', '1 a 2 meses', 'más de 2 meses', 'mismo día'],
        'mapeo': [('mismo día', 'El mismo día del viaje'), ('1 a 3 días', '1 a 3 días antes'),
                  ('4 a 7 días', '4 a 7 días antes'), ('1 a 2 semanas', '1 a 2 semanas antes'),
                  ('3 a 4 semanas', '3 a 4 semanas antes'), ('1 a 2 meses', '1 a 2 meses antes'),
                  ('más de 2 meses', 'Más de 2 meses antes')],
    },
    {
        'campo': 'transporte_llegada',
        # Firmas SIN "vehículo propio" (lo comparte con transporte de destino): la columna se
        # identifica por las opciones exclusivas de llegada, pero el valor "Vehículo propio" sí
        # se mapea abajo porque ya se lee de la columna correcta.
        'firmas': ['autobús interprovincial', 'alquiler sin conductor', 'de un familiar',
                   'turístico contratado', 'avión', 'transporte por app'],
        'mapeo': [('autobús interprovincial', 'Autobús interprovincial'),
                  ('alquiler sin conductor', 'Vehículo de alquiler sin conductor'),
                  ('de un familiar', 'Vehículo de un familiar o amigo'),
                  ('turístico contratado', 'Transporte turístico contratado'),
                  ('transporte por app', 'Servicio de transporte por app'),
                  ('avión', 'Avión'), ('vehículo propio', 'Vehículo propio'), ('otro', 'Otro')],
    },
    {
        'campo': 'tipo_viajero',
        'firmas': ['gama media', 'gama alta', 'bajo presupuesto', 'mochilero'],
        'mapeo': [('bajo presupuesto', 'Bajo Presupuesto/Mochilero'), ('mochilero', 'Bajo Presupuesto/Mochilero'),
                  ('gama media', 'Gama Media'), ('gama alta', 'Gama Alta'), ('lujo', 'Lujo')],
    },
    {
        'campo': 'expectativas',
        'firmas': ['por debajo de lo esperado', 'igual a lo esperado', 'por encima de lo esperado'],
        'mapeo': [('muy por debajo', 'Muy por debajo de lo esperado'), ('algo por debajo', 'Algo por debajo de lo esperado'),
                  ('igual a lo esperado', 'Igual a lo esperado'), ('algo por encima', 'Algo por encima de lo esperado'),
                  ('muy por encima', 'Muy por encima de lo esperado')],
    },
    {
        # P20 del cuestionario ajustado: rubro en el que está dispuesto a gastar MÁS (opción única).
        'campo': 'rubro_mayor_gasto',
        'firmas': ['hoteles de lujo', 'transporte y vuelos', 'experiencias y actividades',
                   'actividades recreativas', 'compras personales', 'artesan'],
        'mapeo': [('hoteles de lujo', 'Alojamiento'), ('transporte y vuelos', 'Transporte y vuelos'),
                  ('experiencias y actividades', 'Experiencias y actividades'),
                  ('actividades recreativas', 'Actividades recreativas/culturales'),
                  ('compras personales', 'Compras personales'), ('artesan', 'Recuerdos y artesanías'),
                  ('alojamiento', 'Alojamiento'), ('gastronomía', 'Gastronomía')],
    },
    # ---- Preguntas de SELECCIÓN MÚLTIPLE (tipo='multi'): el valor que guarda Kobo trae varias
    # opciones pegadas; se extraen todas las reconocidas y se guardan separadas por "; ". ----
    {
        'campo': 'actividades_esperadas', 'tipo': 'multi',  # P10
        'firmas': ['cultura y patrimonio', 'naturaleza y aventura', 'turismo rural', 'turismo deportivo',
                   'entretenimiento y vida nocturna'],
        'mapeo': [('ciudad (parques', 'Ciudad'), ('cultura y patrimonio', 'Cultura y Patrimonio'),
                  ('naturaleza y aventura', 'Naturaleza y Aventura'), ('turismo rural', 'Turismo rural'),
                  ('sol y playa', 'Sol y playa'), ('gastronomía', 'Gastronomía'),
                  ('entretenimiento y vida nocturna', 'Entretenimiento y vida nocturna'),
                  ('ferias', 'Ferias y convenciones'), ('turismo deportivo', 'Turismo deportivo')],
    },
    {
        'campo': 'busco_informacion', 'tipo': 'multi',  # P13
        'firmas': ['buscadores de internet', 'agencias en línea', 'sitio web del destino',
                   'opiniones online', 'agentes de viaje'],
        'mapeo': [('buscadores de internet', 'Buscadores de Internet'), ('agencias en línea', 'Agencias en línea'),
                  ('agentes de viaje', 'Agentes de viaje tradicionales'), ('opiniones online', 'Opiniones online de turistas'),
                  ('recomendaciones personales', 'Recomendaciones personales'), ('sitio web del destino', 'Sitio web del destino'),
                  ('redes sociales', 'Redes sociales')],
    },
    {
        'campo': 'transporte_destino', 'tipo': 'multi',  # P18
        'firmas': ['transporte público', 'taxi amarillo', 'motorizado alquilado', 'bicicleta'],
        'mapeo': [('transporte público', 'Transporte público (bus)'), ('taxi amarillo', 'Taxi amarillo'),
                  ('motorizado alquilado', 'Vehículo motorizado alquilado'),
                  ('motorizado familiar', 'Vehículo motorizado familiar o amigo'),
                  ('uber', 'Uber u otras apps'), ('bicicleta', 'Bicicleta'), ('caminando', 'Caminando'),
                  ('vehículo propio', 'Vehículo propio')],
    },
    {
        'campo': 'motivos_recomendacion', 'tipo': 'multi',  # P26
        'firmas': ['cumplieron con mis necesidades', 'fácil encontrar servicios', 'diversificados',
                   'facilidades y equipamientos', 'seguridad durante'],
        'mapeo': [('cumplieron con mis necesidades', 'Servicios cumplieron necesidades'),
                  ('fácil encontrar servicios', 'Fácil encontrar servicios'),
                  ('diversificados', 'Atractivos diversificados'),
                  ('seguridad durante', 'Seguridad durante la estadía'),
                  ('facilidades y equipamientos', 'Facilidades y equipamientos')],
    },
]

# Bloque de gasto por rubro (P19): 6 columnas contiguas A-F. A = la columna con "presupuesto
# total" en el encabezado; B-F son las 5 siguientes, en el orden del cuestionario ajustado
# (11-09-2026): A.Alojamiento, B.Alimentación, C.Transporte interno, D.Entretenimiento,
# E.Compras, F.Recuerdos. El `gasto_total` se recalcula como la SUMA de los 6 (punto medio de
# cada rango), que es el gasto diario real por persona — antes `gasto_total` guardaba solo la
# columna A (alojamiento), no el total.
GASTO_RUBROS = ['gasto_alojamiento', 'gasto_alimentacion', 'gasto_transporte',
                'gasto_entretenimiento', 'gasto_compras', 'gasto_recuerdos']
GASTO_MAPEO_ORD = [('0 (sin gasto', 0.0), ('menos de $20', 10.0), ('de $21 a $50', 35.0),
                   ('de $51 a $100', 75.0), ('más de $200', 250.0), ('mas de $200', 250.0),
                   ('más de $100', 150.0), ('mas de $100', 150.0)]


def _detectar_bloque_gasto(columnas):
    """Devuelve {campo_rubro: columna} para las 6 columnas A-F del gasto, en orden."""
    cols = list(columnas)
    idxA = next((i for i, c in enumerate(cols)
                 if 'presupuesto total' in _reparar_mojibake(_norm(c)).lower()), None)
    if idxA is None or idxA + 6 > len(cols):
        return {}
    return dict(zip(GASTO_RUBROS, cols[idxA:idxA + 6]))


def _mapear_multi(valor, mapeo):
    """Para preguntas de selección múltiple: devuelve todas las opciones reconocidas dentro del
    valor (que trae varias pegadas), sin repetir y separadas por '; '. None si nada calza."""
    if not _valor_valido(valor):
        return None
    texto = _reparar_mojibake(_norm(valor)).lower()
    seleccionadas = []
    for substr, etiqueta in mapeo:
        if substr in texto and etiqueta not in seleccionadas:
            seleccionadas.append(etiqueta)
    return '; '.join(seleccionadas) if seleccionadas else None

# Matriz de satisfacción por aspecto (escala 1-5). Se identifican por el TEXTO del
# encabezado (único y estable entre versiones del formulario), no por posición.
MATRIZ_SATISFACCION = {
    'satisfaccion_informacion': 'información turística',
    'satisfaccion_atencion': 'atención y amabilidad',
    'satisfaccion_seguridad': 'seguridad percibida',
    'satisfaccion_limpieza': 'limpieza y mantenimiento',
    'satisfaccion_calidad_precio': 'calidad-precio',
}

CAMPOS_NUEVOS_TEXTO = [q['campo'] for q in PREGUNTAS_COLUMNA]
CAMPOS_NUEVOS_ESCALA = list(MATRIZ_SATISFACCION.keys())


def _mapear_valor(valor, mapeo):
    """Devuelve la etiqueta del primer (substring, etiqueta) de `mapeo` contenido en
    el valor; None si la celda no es válida o nada calza."""
    if not _valor_valido(valor):
        return None
    texto = _norm(valor).lower()
    for substr, etiqueta in mapeo:
        if substr in texto:
            return etiqueta
    return None


def _asignar_columnas_nuevas(df, columnas_pregunta):
    """Para cada pregunta de PREGUNTAS_COLUMNA busca la columna cuyos valores calzan
    más veces con sus `firmas`, asignando cada columna a una sola pregunta (evita que
    dos preguntas con una opción compartida terminen en la misma columna). Devuelve
    {campo: nombre_columna}."""
    series = {}
    for c in columnas_pregunta:
        try:
            series[c] = df[c].dropna().astype(str).str.lower()
        except Exception:
            series[c] = None
    umbral = max(3, int(0.02 * len(df)))  # evita falsos positivos por un par de celdas sueltas
    usadas = set()
    asignacion = {}
    for q in PREGUNTAS_COLUMNA:
        mejor, mejor_score = None, 0
        for c in columnas_pregunta:
            if c in usadas or series.get(c) is None or len(series[c]) == 0:
                continue
            s = series[c]
            score = int(s.apply(lambda t: any(f in t for f in q['firmas'])).sum())
            if score > mejor_score:
                mejor, mejor_score = c, score
        if mejor is not None and mejor_score >= umbral:
            asignacion[q['campo']] = mejor
            usadas.add(mejor)
    return asignacion


def _detectar_matriz_satisfaccion(columnas):
    """{campo: columna} para los aspectos de MATRIZ_SATISFACCION, por texto de encabezado."""
    res = {}
    for campo, texto in MATRIZ_SATISFACCION.items():
        for c in columnas:
            if texto in _reparar_mojibake(_norm(c)).lower():
                res[campo] = c
                break
    return res


def _detectar_columna_lugares(columnas):
    """La pregunta de lugares visitados se identifica por CONTENIDO del encabezado (aparece con
    y sin el prefijo de sección según la versión del formulario), no por posición."""
    for c in columnas:
        texto = _norm(c).lower()
        if 'lugares' in texto and 'visit' in texto:
            return c
    return None


def extraer_lugares(valor):
    """La respuesta llega como los nombres de los lugares elegidos, PEGADOS uno tras otro sin
    separador (p. ej. 'La Chocolatera La Lobería Cerro El Morro Chipipe'), porque Kobo exporta
    así esta pregunta de selección múltiple en este formulario. Como ningún lugar de
    LUGARES_CONOCIDOS es substring de otro, buscar cada nombre completo como frase dentro del
    texto es suficiente y no requiere adivinar dónde corta cada palabra."""
    if not _valor_valido(valor):
        return []
    texto = _reparar_mojibake(_norm(valor)).lower()
    return [lugar for lugar in LUGARES_CONOCIDOS if lugar.lower() in texto]


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


def _primer_digito(valor):
    """Las respuestas de escala vienen como '4' o '1 - Muy improbable'; se toma el
    primer dígito. Devuelve None si la celda no es válida o no tiene dígitos."""
    if not _valor_valido(valor):
        return None
    m = re.search(r'\d', _norm(valor))
    return int(m.group()) if m else None


def extraer_registro(row, columnas_pregunta, col_satisfaccion, col_probabilidad,
                     cols_nuevas=None, cols_matriz=None, cols_gasto=None):
    """A partir de una fila cruda, clasifica cada campo por contenido.
    `cols_nuevas` ({campo: columna}), `cols_matriz` ({campo: columna}) y `cols_gasto`
    ({campo_rubro: columna}) vienen precalculados por archivo."""
    cols_nuevas = cols_nuevas or {}
    cols_matriz = cols_matriz or {}
    cols_gasto = cols_gasto or {}
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

    nivel_satisfaccion = _primer_digito(row.get(col_satisfaccion)) if col_satisfaccion else None
    probabilidad_retorno = _primer_digito(row.get(col_probabilidad)) if col_probabilidad else None

    registro = {
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

    # Preguntas nuevas que se leen de su columna asignada. Las de opción única (tipo 'texto')
    # devuelven una etiqueta; las de selección múltiple (tipo 'multi') devuelven las opciones
    # elegidas separadas por "; ".
    mapeos = {q['campo']: q['mapeo'] for q in PREGUNTAS_COLUMNA}
    tipos = {q['campo']: q.get('tipo', 'texto') for q in PREGUNTAS_COLUMNA}
    for campo in CAMPOS_NUEVOS_TEXTO:
        col = cols_nuevas.get(campo)
        if tipos[campo] == 'multi':
            valor = _mapear_multi(row.get(col), mapeos[campo]) if col else None
        else:
            valor = _mapear_valor(row.get(col), mapeos[campo]) if col else None
        registro[campo] = valor or 'Prefiero no responder'
    # Matriz de satisfacción por aspecto (escala 1-5; 0 = sin dato, como nivel_satisfaccion).
    for campo in CAMPOS_NUEVOS_ESCALA:
        col = cols_matriz.get(campo)
        valor = _primer_digito(row.get(col)) if col else None
        registro[campo] = valor if valor is not None else 0

    # Gasto por rubro (P19, 6 columnas) + recálculo de gasto_total = suma de los 6 (punto medio
    # de cada rango). Si no se detectó el bloque, se conserva el gasto_total de arriba (columna A).
    suma, hay_gasto = 0.0, False
    for campo in GASTO_RUBROS:
        col = cols_gasto.get(campo)
        val = _mapear_valor(row.get(col), GASTO_MAPEO_ORD) if col else None
        registro[campo] = val if val is not None else 0.0
        if val is not None:
            suma += val
            hay_gasto = True
    if hay_gasto:
        registro['gasto_total'] = suma

    return registro


def procesar_encuestas_turismo(df, get_db_connection):
    """Procesa el DataFrame crudo (tal cual lo entrega leer_archivo) e
    inserta en encuestas_turisticas."""
    columnas_pregunta = _columnas_pregunta(df.columns)
    col_satisfaccion = _detectar_col_satisfaccion(df.columns)
    col_probabilidad = _detectar_col_probabilidad(df.columns, columnas_pregunta, col_satisfaccion)
    col_lugares = _detectar_columna_lugares(df.columns)
    cols_nuevas = _asignar_columnas_nuevas(df, columnas_pregunta)
    cols_matriz = _detectar_matriz_satisfaccion(df.columns)
    cols_gasto = _detectar_bloque_gasto(df.columns)

    advertencias = []
    if col_satisfaccion is None:
        advertencias.append(
            "No se pudo ubicar la pregunta de satisfacción general; "
            "nivel_satisfaccion y probabilidad_retorno quedarán en 0."
        )
    else:
        advertencias.append(PROBABILIDAD_RETORNO_NOTA)
    if not col_lugares:
        advertencias.append(
            "Este archivo no trae la pregunta de lugares visitados (o no se pudo identificar); "
            "estos registros no aparecerán en el mapa del dashboard."
        )

    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, omitidos, detalles = 0, 0, 0, []
    columnas_bd = columnas_de(cursor, 'encuestas_turisticas')
    con_lugares = col_lugares is not None and 'lugares_visitados' in columnas_bd
    if col_lugares and not con_lugares:
        advertencias.append(
            "Reinicia el backend para aplicar la actualización de la base de datos: sin ella no "
            "se guardan los lugares visitados de este archivo."
        )

    # uuid_kobo identifica el envío de forma única (columna "_uuid" del export de Kobo). Con esto,
    # subir dos veces el mismo archivo ya no duplica encuestas: la fila se omite si su _uuid ya está.
    con_uuid = 'uuid_kobo' in columnas_bd
    uuids_existentes = set()
    if con_uuid:
        cursor.execute("SELECT uuid_kobo FROM encuestas_turisticas WHERE uuid_kobo IS NOT NULL")
        uuids_existentes = {f['uuid_kobo'] for f in cursor.fetchall()}
    else:
        advertencias.append(
            "Reinicia el backend para activar la protección contra cargas duplicadas de este "
            "archivo (falta la columna encuestas_turisticas.uuid_kobo)."
        )

    try:
        campos = ['fecha_encuesta', 'genero', 'edad', 'pais_residencia', 'ciudad_residencia',
                  'nivel_educativo', 'ocupacion', 'tamano_grupo', 'frecuencia_visitas',
                  'motivo_visita', 'noches_estadia', 'gasto_total', 'nivel_satisfaccion',
                  'probabilidad_retorno']
        # Campos nuevos (2026-10): solo si ya existen en la BD (requiere la migración de columnas).
        campos_nuevos_presentes = [c for c in (CAMPOS_NUEVOS_TEXTO + CAMPOS_NUEVOS_ESCALA + GASTO_RUBROS) if c in columnas_bd]
        campos.extend(campos_nuevos_presentes)
        if con_lugares:
            campos.append('lugares_visitados')
        if con_uuid:
            campos.append('uuid_kobo')
        marcas = ', '.join(['%s'] * len(campos))
        sql = f"INSERT INTO encuestas_turisticas ({', '.join(campos)}, fecha_registro) VALUES ({marcas}, NOW())"

        for index, row in df.iterrows():
            fila_num = int(index) + 2  # +1 por índice 0, +1 por la fila de encabezado
            try:
                uuid_fila = _norm(row.get('_uuid')) if con_uuid and _valor_valido(row.get('_uuid')) else None
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

                registro = extraer_registro(row, columnas_pregunta, col_satisfaccion,
                                            col_probabilidad, cols_nuevas, cols_matriz, cols_gasto)

                valores = [
                    fecha, registro['genero'], registro['edad'], registro['pais_residencia'],
                    'Prefiero no responder',  # ciudad_residencia: no existe en este formulario
                    registro['nivel_educativo'], registro['ocupacion'], registro['tamano_grupo'],
                    registro['frecuencia_visitas'], registro['motivo_visita'],
                    registro['noches_estadia'], registro['gasto_total'],
                    registro['nivel_satisfaccion'], registro['probabilidad_retorno']
                ]
                valores.extend(registro[c] for c in campos_nuevos_presentes)
                if con_lugares:
                    lugares = extraer_lugares(row.get(col_lugares)) if col_lugares else []
                    valores.append(json.dumps(lugares, ensure_ascii=False))
                if con_uuid:
                    valores.append(uuid_fila)

                cursor.execute(sql, valores)
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

    if omitidos:
        advertencias.append(
            f"{omitidos} fila(s) ya estaban cargadas (mismo envío de Kobo) y se omitieron - no se duplicaron."
        )

    return {'insertados': insertados, 'errores': errores, 'omitidos': omitidos, 'detalles': detalles, 'advertencias': advertencias}

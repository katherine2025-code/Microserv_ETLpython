"""
Parser dedicado para el CSV que genera la propia app OTS con las encuestas
que llenan los encuestadores (Gestión de Encuestas -> "Descargar respuestas",
o el botón "Exportar CSV" del formulario del encuestador).

A diferencia de los exports crudos de KoboToolbox (ver kobo_*.py), este
formato es PROPIO y conocido: cada columna es el `codigo` de una pregunta
(ej. 'edad', 'gasto_transporte', 'fecha1_nacionales'), así que cada campo se
lee por NOMBRE de columna, sin adivinar por contenido ni por posición.

Columnas:
  id_respuesta (opcional) | fecha_encuesta | tipo_encuesta | <codigo de cada pregunta>

  - id_respuesta: id en la tabla `respuestas_encuestas`. Si viene, la carga es
    IDEMPOTENTE: una respuesta ya cargada (fecha_etl no nula) se omite, así
    subir dos veces el mismo archivo no duplica datos.
  - tipo_encuesta: 'turista' -> encuestas_turisticas | 'hotel' -> ocupacion_hotelera.

El administrador puede editar las preguntas y respuestas. Por eso el parser es
tolerante: si falta una columna se usa el valor por defecto de siempre y se
avisa; si una respuesta ya no coincide con las opciones conocidas se conserva
el texto tal cual (o 0 en campos numéricos) y se cuenta en las advertencias.
"""
import re
from datetime import timedelta

import pandas as pd

from cantones import canton_de_parroquia, normalizar_canton, columnas_de
from kobo_encuestas_turismo import (
    EDAD_MAPEO, PAIS_MAPEO, NOCHES_MAPEO, GASTO_MAPEO,
    NIVEL_EDUCATIVO_MAPEO, OCUPACION_MAPEO, FRECUENCIA_MAPEO,
    PREGUNTAS_COLUMNA, GASTO_RUBROS, GASTO_MAPEO_ORD, MATRIZ_SATISFACCION,
    CAMPOS_NUEVOS_ESCALA, _mapear_valor, _mapear_multi,
)

COLUMNAS_MARCADOR = ('tipo_encuesta', 'fecha_encuesta')
NO_RESPONDER = 'Prefiero no responder'
TOKENS_NULOS = {'', 'nan', 'null', 'none', 'n/a', 'na', '-'}
SEPARADOR_MULTIPLE = ' | '
ECUADOR_UTC = timedelta(hours=-5)  # Ecuador continental: sin horario de verano

GENERO_APP = {'hombre': 'Masculino', 'mujer': 'Femenino', 'otro': 'Otro', 'no binaria': 'Otro',
              'prefiero no responder': NO_RESPONDER}
# Opciones de la app que equivalen a categorías ya existentes en encuestas_turisticas
MOTIVOS_APP = {
    'ocio / vacaciones': 'Ocio / Vacaciones',
    'visita a familiares o amigos': 'Visita a familiares o amigos',
    'negocios, trabajo o eventos corporativos': 'Negocios, trabajo o eventos corporativos',
    'asistencia a eventos (culturales, deportivos, religiosos, etc.)': 'Asistencia a eventos',
    'estudios o formación': 'Estudios o formación',
    'salud o tratamiento médico': 'Salud o tratamiento médico',
}
# Cuestionario ajustado (11-09-2026): 6 rubros de gasto A-F. El código de la pregunta en la app
# coincide 1:1 con la columna de encuestas_turisticas (GASTO_RUBROS), por eso se leen por nombre.
COLUMNAS_GASTO = list(GASTO_RUBROS)  # gasto_alojamiento, _alimentacion, _transporte, _entretenimiento, _compras, _recuerdos

# Preguntas "nuevas" de opción única / múltiple: mapea el CÓDIGO de la pregunta en la app ->
# (columna BD, mapeo canónico, tipo). Se reutilizan los mapeos del parser de Kobo para que un
# mismo concepto quede con la MISMA etiqueta venga de donde venga (consistencia). El único
# código que difiere del nombre de columna es 'actividades' -> columna 'actividades_esperadas'.
_CODIGO_APP = {'actividades_esperadas': 'actividades'}
PREGUNTAS_APP = [
    {'codigo': _CODIGO_APP.get(p['campo'], p['campo']), 'columna': p['campo'],
     'mapeo': p['mapeo'], 'tipo': p.get('tipo', 'texto')}
    for p in PREGUNTAS_COLUMNA
]

CAMPOS_TURISTA = ['edad', 'genero', 'pais_residencia', 'educacion', 'ocupacion', 'personas_grupo',
                  'motivo_visita', 'veces_visitado', 'noches_hospedado', 'satisfaccion_atractivos',
                  'probabilidad_recomendar'] + COLUMNAS_GASTO
CAMPOS_HOTEL = ['nombre_establecimiento', 'canton', 'parroquia', 'num_habitaciones', 'feriado',
                'fecha1_fecha', 'fecha1_nacionales', 'fecha1_extranjeros',
                'fecha1_pernoctaciones', 'fecha1_habitaciones', 'fecha1_tarifa']


# ==========================================================================
# DETECCIÓN
# ==========================================================================
def es_csv_app(columnas) -> bool:
    nombres = {str(c).strip().lower() for c in columnas}
    return all(m in nombres for m in COLUMNAS_MARCADOR)


def tipo_encuesta_del_archivo(df):
    """'turista' | 'hotel' si todo el archivo es de un mismo tipo; None si
    viene mezclado, vacío o con un tipo desconocido."""
    tipos = {str(v).strip().lower() for v in df['tipo_encuesta'].dropna()}
    if len(tipos) == 1:
        tipo = next(iter(tipos))
        return tipo if tipo in ('turista', 'hotel') else None
    return None


# ==========================================================================
# UTILIDADES
# ==========================================================================
def _norm(texto) -> str:
    return re.sub(r'\s+', ' ', str(texto)).strip()


def _valido(valor) -> bool:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return False
    return _norm(valor).lower() not in TOKENS_NULOS


def _texto(row, campo):
    """Texto limpio de una celda, o None si falta la columna o está vacía."""
    if campo not in row.index:
        return None
    valor = row[campo]
    return _norm(valor) if _valido(valor) else None


def _entero(row, campo):
    texto = _texto(row, campo)
    if texto is None:
        return None
    m = re.search(r'-?\d+(?:[.,]\d+)?', texto)
    if not m:
        return None
    return max(0, int(round(float(m.group().replace(',', '.')))))


def _decimal(row, campo):
    texto = _texto(row, campo)
    if texto is None:
        return None
    m = re.search(r'-?\d+(?:[.,]\d+)?', texto.replace('$', ''))
    if not m:
        return None
    return max(0.0, float(m.group().replace(',', '.')))


def _buscar(texto, mapeo):
    """Valor asociado a la primera clave del mapeo contenida en el texto."""
    bajo = texto.lower()
    for clave, valor in mapeo.items():
        if clave in bajo:
            return valor
    return None


def _primera_opcion(texto):
    """En preguntas de selección múltiple, la primera respuesta elegida."""
    return _norm(texto.split(SEPARADOR_MULTIPLE.strip())[0]) if texto else texto


def _fecha_local(valor):
    """Fecha (YYYY-MM-DD) en hora de Ecuador. Acepta 'YYYY-MM-DD HH:MM:SS'
    (export del servidor, ya en hora local) o ISO con zona ('...Z', export del
    formulario del encuestador), que se convierte de UTC a hora de Ecuador."""
    if not _valido(valor):
        return None
    texto = _norm(valor)
    dayfirst = not re.match(r'^\d{4}-\d{2}-\d{2}', texto)
    fecha = pd.to_datetime(texto, errors='coerce', dayfirst=dayfirst, utc=texto.endswith('Z') or '+' in texto[10:])
    if pd.isna(fecha):
        return None
    if fecha.tzinfo is not None:
        fecha = (fecha + ECUADOR_UTC).tz_localize(None)
    return fecha.strftime('%Y-%m-%d')


class _Contador:
    """Cuenta respuestas que no coincidieron con las opciones conocidas
    (p. ej. porque el administrador editó el cuestionario) para advertirlo."""

    def __init__(self):
        self.no_reconocidos = {}
        self.columnas_faltantes = []

    def sin_mapeo(self, campo):
        self.no_reconocidos[campo] = self.no_reconocidos.get(campo, 0) + 1


# ==========================================================================
# TURISTAS -> encuestas_turisticas
# ==========================================================================
def _extraer_turista(row, cont):
    def categoria(campo, mapeo, defecto=NO_RESPONDER, maximo=100):
        texto = _texto(row, campo)
        if texto is None:
            return defecto
        encontrado = _buscar(texto, mapeo)
        if encontrado is not None:
            return encontrado
        cont.sin_mapeo(campo)
        return texto[:maximo]  # se conserva la respuesta tal cual en vez de perderla

    genero = _texto(row, 'genero')
    if genero is None:
        genero = NO_RESPONDER
    else:
        mapeado = _buscar(genero, GENERO_APP)
        if mapeado is None:
            cont.sin_mapeo('genero')
        genero = mapeado or 'Otro'

    edad_txt = _texto(row, 'edad')
    edad = _buscar(edad_txt, EDAD_MAPEO) if edad_txt else None
    if edad_txt and edad is None:
        cont.sin_mapeo('edad')

    pais_txt = _texto(row, 'pais_residencia')
    pais = (_buscar(pais_txt, PAIS_MAPEO) or pais_txt.title()) if pais_txt else NO_RESPONDER

    grupo_txt = _texto(row, 'personas_grupo')
    tamano_grupo = 0
    if grupo_txt:
        if re.search(r'solo yo|viajé solo|viaje solo', grupo_txt.lower()):
            tamano_grupo = 1
        else:
            m = re.search(r'(\d+)', grupo_txt)
            tamano_grupo = int(m.group(1)) if m else 0
            if not m:
                cont.sin_mapeo('personas_grupo')

    motivo_txt = _texto(row, 'motivo_visita')
    if motivo_txt:
        # Coincidencia EXACTA (no por subcadena: 'Negocios' contiene 'ocio'). Los motivos
        # que el ETL ya conocía conservan su categoría; los demás se guardan tal cual.
        primero = _primera_opcion(motivo_txt)
        motivo = MOTIVOS_APP.get(primero.lower(), primero)[:150]
    else:
        motivo = NO_RESPONDER

    noches_txt = _texto(row, 'noches_hospedado')
    noches = _buscar(noches_txt, NOCHES_MAPEO) if noches_txt else None
    if noches_txt and noches is None:
        cont.sin_mapeo('noches_hospedado')

    # Gasto por rubro (6) + gasto_total = suma de los 6 (valor medio de cada rango). Cada rubro
    # se guarda además en su propia columna.
    gasto_total, hay_gasto = 0.0, False
    rubros = {}
    for campo in COLUMNAS_GASTO:
        texto = _texto(row, campo)
        monto = _mapear_valor(texto, GASTO_MAPEO_ORD) if texto else None
        rubros[campo] = monto if monto is not None else 0.0
        if texto and monto is None and 'prefiero no responder' not in texto.lower():
            cont.sin_mapeo(campo)
        if monto is not None:
            gasto_total += monto
            hay_gasto = True

    def escala(campo):
        texto = _texto(row, campo)
        m = re.search(r'\d', texto) if texto else None
        return int(m.group()) if m else 0  # 'N/A' o vacío -> 0 (convención del ETL)

    registro = {
        'genero': genero,
        'edad': edad if edad is not None else 0,
        'pais_residencia': pais,
        'ciudad_residencia': NO_RESPONDER,  # esta encuesta no pregunta ciudad
        'nivel_educativo': categoria('educacion', NIVEL_EDUCATIVO_MAPEO),
        'ocupacion': categoria('ocupacion', OCUPACION_MAPEO),
        'tamano_grupo': tamano_grupo,
        'frecuencia_visitas': categoria('veces_visitado', FRECUENCIA_MAPEO, maximo=50),
        'motivo_visita': motivo,
        'noches_estadia': noches if noches is not None else 0,
        'gasto_total': round(gasto_total, 2) if hay_gasto else 0.0,
        'nivel_satisfaccion': escala('satisfaccion_atractivos'),
        # La app pregunta "probabilidad de recomendar/volver" (escala 1-5); es el
        # equivalente del indicador de retorno/lealtad que guarda la tabla.
        'probabilidad_retorno': escala('probabilidad_recomendar'),
    }
    registro.update(rubros)

    # Preguntas nuevas de opción única / múltiple (cuestionario ajustado). Se leen por código y
    # se normalizan a la etiqueta canónica con el mismo mapeo que el parser de Kobo.
    for p in PREGUNTAS_APP:
        texto = _texto(row, p['codigo'])
        if p['tipo'] == 'multi':
            valor = _mapear_multi(texto, p['mapeo']) if texto else None
        else:
            valor = _mapear_valor(texto, p['mapeo']) if texto else None
        if texto and valor is None and 'prefiero no responder' not in texto.lower():
            cont.sin_mapeo(p['codigo'])
        registro[p['columna']] = valor or NO_RESPONDER

    # Matriz de satisfacción por aspecto (escala 1-5; código = columna).
    for campo in CAMPOS_NUEVOS_ESCALA:
        registro[campo] = escala(campo)

    return registro


# Orden de columnas para el INSERT de turista. Las nuevas (cuestionario ajustado) solo se
# incluyen si ya existen en encuestas_turisticas (ver _sql_turista), para no romper si falta
# la migración de columnas.
CAMPOS_TURISTA_BASE = ['genero', 'edad', 'pais_residencia', 'ciudad_residencia',
                       'nivel_educativo', 'ocupacion', 'tamano_grupo', 'frecuencia_visitas',
                       'motivo_visita', 'noches_estadia', 'gasto_total', 'nivel_satisfaccion',
                       'probabilidad_retorno']
CAMPOS_TURISTA_NUEVOS = ([p['columna'] for p in PREGUNTAS_APP]
                         + list(CAMPOS_NUEVOS_ESCALA) + list(GASTO_RUBROS))


def _sql_turista(columnas_bd):
    """Arma el INSERT de turista incluyendo las columnas nuevas que existan en la BD."""
    nuevos = [c for c in CAMPOS_TURISTA_NUEVOS if c in columnas_bd]
    campos = ['fecha_encuesta'] + CAMPOS_TURISTA_BASE + nuevos
    marcas = ', '.join(['%s'] * len(campos))
    sql = (f"INSERT INTO encuestas_turisticas ({', '.join(campos)}, fecha_registro) "
           f"VALUES ({marcas}, NOW())")
    return sql, nuevos


# ==========================================================================
# ESTABLECIMIENTOS -> ocupacion_hotelera
# ==========================================================================
def _extraer_hotel(row, fecha_captura):
    """Datos del establecimiento + un bloque por cada fecha (día del feriado) reportada.
    Devuelve (registro, avisos). Un feriado dura de 1 a 5 días: se generará un registro
    de ocupación por cada bloque con fecha."""
    avisos = []
    disponibles = _entero(row, 'num_habitaciones') or 0
    plazas = _entero(row, 'num_plazas') or 0
    categoria = _texto(row, 'categoria') or ''
    m = re.match(r'^([1-5])\s*estrella', categoria.lower())

    parroquia = (_texto(row, 'parroquia') or '').title()[:100] or None
    canton_declarado = normalizar_canton(_texto(row, 'canton'))
    canton_parroquia = canton_de_parroquia(parroquia)
    if canton_declarado and canton_parroquia and canton_declarado != canton_parroquia:
        avisos.append(
            f"la parroquia '{parroquia}' pertenece al cantón {canton_parroquia} pero la encuesta "
            f"se registró en {canton_declarado}; se usó {canton_declarado}"
        )

    bloques, fechas_vistas = [], set()
    for n in range(1, 6):
        texto_fecha = _texto(row, f'fecha{n}_fecha')
        if texto_fecha:
            fecha = _fecha_local(texto_fecha)
            if not fecha:
                avisos.append(f"Fecha {n}: '{texto_fecha}' no es una fecha válida; se omitió ese día")
                continue
        elif n == 1:
            fecha = fecha_captura  # versión anterior del formulario: solo traía "Fecha 1" sin día
        else:
            continue  # bloque sin usar (el feriado duró menos días)

        if fecha in fechas_vistas:
            avisos.append(f"Fecha {n}: el día {fecha} está repetido; se omitió")
            continue
        fechas_vistas.add(fecha)

        ocupadas = _entero(row, f'fecha{n}_habitaciones') or 0
        nacionales_dia = _entero(row, f'fecha{n}_nacionales') or 0
        extranjeros_dia = _entero(row, f'fecha{n}_extranjeros') or 0
        pernoctaciones = _entero(row, f'fecha{n}_pernoctaciones') or 0
        # Tasa por plazas = personas alojadas ese día / plazas (capacidad en personas). Sin turistas
        # desglosados se usan las pernoctaciones del día (cada persona que duerme esa noche = 1).
        personas = (nacionales_dia + extranjeros_dia) or pernoctaciones
        bloques.append({
            'fecha': fecha,
            'checkin_nacionales': nacionales_dia,
            'checkin_extranjeros': extranjeros_dia,
            'total_turistas': nacionales_dia + extranjeros_dia,
            'pernoctaciones': pernoctaciones,
            'habitaciones_ocupadas': ocupadas,
            'tarifa_cobrada': _decimal(row, f'fecha{n}_tarifa') or 0.0,
            'ocupacion_porcentaje': round(min(100.0, ocupadas / disponibles * 100), 2) if disponibles > 0 else 0.0,
            'ocupacion_plazas_porcentaje': round(min(100.0, personas / plazas * 100), 2) if plazas > 0 else None,
        })

    return {
        'nombre': _texto(row, 'nombre_establecimiento') or '',
        'direccion': (_texto(row, 'direccion') or '')[:255] or None,
        'correo': (_texto(row, 'email') or '')[:150] or None,
        'telefono': (_texto(row, 'telefono') or '')[:50] or None,
        'parroquia': parroquia,
        'canton': canton_declarado or canton_parroquia,
        'feriado': (_texto(row, 'feriado') or '')[:80] or None,
        'categoria': m.group(1) if m else '3',
        'habitaciones_disponibles': disponibles,
        'plazas_disponibles': _entero(row, 'num_plazas'),
        'bloques': bloques,
    }, avisos


def _elegir_hotel(existentes, canton):
    """Entre los hoteles con el mismo nombre, el que corresponde al cantón indicado. Dos
    establecimientos pueden llamarse igual en Salinas y en Santa Elena: no se deben mezclar.
    Devuelve (lista_de_ids_candidatos, hay_que_completar_canton)."""
    if not canton:
        return [i for i, _ in existentes], False
    del_canton = [i for i, c in existentes if c == canton]
    if del_canton:
        return del_canton, False
    sin_canton = [i for i, c in existentes if not c]
    if sin_canton:
        return sin_canton, True  # mismo nombre, cantón aún sin clasificar: es el mismo hotel
    return [], False


def _sql_crear_hotel(columnas_hoteles):
    campos = ['nombre', 'direccion', 'correo', 'telefono', 'categoria', 'parroquia',
              'habitaciones_totales', 'habitaciones_disponibles', 'plazas_disponibles']
    if 'canton' in columnas_hoteles:
        campos.append('canton')
    marcas = ', '.join(['%s'] * len(campos))
    return f"INSERT INTO hoteles ({', '.join(campos)}, created_at, updated_at) VALUES ({marcas}, NOW(), NOW())", \
           'canton' in columnas_hoteles


def _sql_ocupacion(columnas_ocupacion):
    # total_turistas y habitaciones_totales: el backend los calcula solo cuando se inserta por
    # Sequelize (hooks beforeCreate/beforeUpdate); el ETL inserta con SQL directo y esos hooks no
    # corren, así que se calculan aquí (ver _extraer_hotel) para no dejarlos en 0.
    campos = ['id_hotel', 'fecha', 'checkin_nacionales', 'checkin_extranjeros', 'total_turistas', 'pernoctaciones',
              'habitaciones_ocupadas', 'habitaciones_disponibles', 'habitaciones_totales',
              'tarifa_cobrada', 'ocupacion_porcentaje']
    if 'feriado' in columnas_ocupacion:
        campos.append('feriado')
    if 'ocupacion_plazas_porcentaje' in columnas_ocupacion:
        campos += ['plazas_disponibles', 'ocupacion_plazas_porcentaje']
    marcas = ', '.join(['%s'] * len(campos))
    return (f"INSERT INTO ocupacion_hotelera ({', '.join(campos)}, fuente_dato, created_at) "
            f"VALUES ({marcas}, 'Encuesta', NOW())"), 'feriado' in columnas_ocupacion, \
        'ocupacion_plazas_porcentaje' in columnas_ocupacion


# ==========================================================================
# PROCESO PRINCIPAL
# ==========================================================================
def procesar_encuestas_app(df, tipo_encuesta, get_db_connection):
    """Inserta las filas del CSV de la app. Devuelve el mismo formato que los
    demás procesadores: insertados/errores/detalles/advertencias."""
    df = df.rename(columns=lambda c: str(c).strip())
    campos_esperados = CAMPOS_TURISTA if tipo_encuesta == 'turista' else CAMPOS_HOTEL
    faltantes = [c for c in campos_esperados if c not in df.columns]

    advertencias = []
    if faltantes:
        advertencias.append(
            "El archivo no trae estas columnas (se usaron valores por defecto; puede ser un "
            "cuestionario editado o una versión anterior): " + ', '.join(faltantes)
        )

    connection = get_db_connection()
    cursor = connection.cursor()
    cont = _Contador()
    insertados, errores, omitidos, detalles = 0, 0, 0, []
    hoteles_creados = []
    registros_ocupacion = 0

    try:
        hoteles_por_nombre = {}  # nombre normalizado -> [(id_hotel, canton)]
        sql_crear_hotel = sql_ocupacion = None
        con_canton_hotel = con_feriado = False
        if tipo_encuesta == 'hotel':
            columnas_hoteles = columnas_de(cursor, 'hoteles')
            columnas_ocupacion = columnas_de(cursor, 'ocupacion_hotelera')
            sql_crear_hotel, con_canton_hotel = _sql_crear_hotel(columnas_hoteles)
            sql_ocupacion, con_feriado, con_plazas = _sql_ocupacion(columnas_ocupacion)
            if not (con_canton_hotel and con_feriado):
                advertencias.append(
                    "Reinicia el backend para aplicar la actualización de la base de datos: sin ella no se "
                    "guardan el cantón de los hoteles ni el feriado de cada registro de ocupación."
                )

            campo_canton = ', canton' if con_canton_hotel else ''
            cursor.execute(f"SELECT id_hotel, LOWER(TRIM(nombre)) AS nombre_norm{campo_canton} FROM hoteles")
            for fila in cursor.fetchall():
                hoteles_por_nombre.setdefault(fila['nombre_norm'], []).append(
                    (fila['id_hotel'], fila.get('canton') if con_canton_hotel else None))

            cursor.execute("SELECT COUNT(*) AS n FROM feriados")
            if cursor.fetchone()['n'] == 0:
                advertencias.append(
                    "La tabla de feriados está vacía: carga el calendario oficial de feriados (tipo 'Días "
                    "Feriados') para que el sistema y el modelo de predicción reconozcan los feriados."
                )

        sql_turista = campos_turista_nuevos = None
        if tipo_encuesta == 'turista':
            columnas_turisticas = columnas_de(cursor, 'encuestas_turisticas')
            sql_turista, campos_turista_nuevos = _sql_turista(columnas_turisticas)
            faltan_cols = [c for c in CAMPOS_TURISTA_NUEVOS if c not in columnas_turisticas]
            if faltan_cols:
                advertencias.append(
                    "Reinicia el backend para aplicar la migración de columnas: sin ella no se guardan "
                    f"{len(faltan_cols)} campos nuevos del cuestionario ajustado (gasto por rubro, "
                    "actividades, satisfacción por aspecto, etc.)."
                )

        # La marca de "ya cargada" vive en respuestas_encuestas.fecha_etl (la crea el backend
        # al iniciar). Si no existe todavía, se carga igual pero sin protección contra duplicados.
        cursor.execute(
            "SELECT COUNT(*) AS n FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME = 'respuestas_encuestas' AND COLUMN_NAME = 'fecha_etl'"
        )
        soporta_dedupe = cursor.fetchone()['n'] > 0
        if 'id_respuesta' in df.columns and not soporta_dedupe:
            advertencias.append(
                "Reinicia el backend para habilitar la protección contra cargas duplicadas "
                "(falta la columna respuestas_encuestas.fecha_etl); este archivo se cargó sin ella."
            )
        tiene_id = 'id_respuesta' in df.columns and soporta_dedupe

        for index, row in df.iterrows():
            fila_num = int(index) + 2  # +1 por índice 0, +1 por la fila de encabezado
            try:
                id_resp = _entero(row, 'id_respuesta') if tiene_id else None

                # Idempotencia: la respuesta ya se cargó en una corrida anterior
                if id_resp is not None:
                    cursor.execute("SELECT fecha_etl FROM respuestas_encuestas WHERE id_respuesta = %s", (id_resp,))
                    previa = cursor.fetchone()
                    if previa and previa['fecha_etl'] is not None:
                        omitidos += 1
                        continue

                fecha = _fecha_local(row.get('fecha_encuesta'))
                if not fecha:
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(f"Fila {fila_num}: no se pudo leer 'fecha_encuesta'")
                    continue

                if tipo_encuesta == 'turista':
                    r = _extraer_turista(row, cont)
                    valores = [fecha] + [r[c] for c in CAMPOS_TURISTA_BASE] \
                              + [r[c] for c in campos_turista_nuevos]
                    cursor.execute(sql_turista, valores)
                else:
                    r, avisos = _extraer_hotel(row, fecha)
                    for aviso in avisos:
                        if len(detalles) < 10:
                            detalles.append(f"Fila {fila_num}: {aviso}")

                    nombre_norm = r['nombre'].lower()
                    if not nombre_norm:
                        errores += 1
                        if len(detalles) < 10:
                            detalles.append(f"Fila {fila_num}: falta el nombre del establecimiento")
                        continue
                    if not r['bloques']:
                        errores += 1
                        if len(detalles) < 10:
                            detalles.append(f"Fila {fila_num}: no trae ninguna fecha válida de ocupación")
                        continue

                    candidatos, completar_canton = _elegir_hotel(
                        hoteles_por_nombre.get(nombre_norm, []), r['canton'])
                    if not candidatos:
                        # Hotel nuevo: se crea con lo que trae la encuesta (el admin
                        # completa el resto después), igual que en el formulario Kobo.
                        valores = [r['nombre'], r['direccion'], r['correo'], r['telefono'], r['categoria'],
                                   r['parroquia'], r['habitaciones_disponibles'], r['habitaciones_disponibles'],
                                   r['plazas_disponibles']]
                        if con_canton_hotel:
                            valores.append(r['canton'])
                        cursor.execute(sql_crear_hotel, valores)
                        candidatos = [cursor.lastrowid]
                        hoteles_por_nombre.setdefault(nombre_norm, []).append((candidatos[0], r['canton']))
                        hoteles_creados.append(r['nombre'])
                    if len(candidatos) > 1:
                        errores += 1
                        if len(detalles) < 10:
                            detalles.append(
                                f"Fila {fila_num}: hay {len(candidatos)} hoteles con el nombre "
                                f"'{r['nombre']}' - no se puede saber cuál usar."
                            )
                        continue

                    id_hotel = candidatos[0]
                    if completar_canton and con_canton_hotel:
                        cursor.execute("UPDATE hoteles SET canton = %s WHERE id_hotel = %s AND canton IS NULL",
                                       (r['canton'], id_hotel))
                        hoteles_por_nombre[nombre_norm] = [
                            (i, r['canton'] if i == id_hotel else c) for i, c in hoteles_por_nombre[nombre_norm]]

                    # Un registro de ocupación por cada día del feriado reportado. habitaciones_totales
                    # usa el mismo valor que habitaciones_disponibles: el formulario no pregunta un
                    # "total" aparte, solo la capacidad que el hotel reportó ese feriado.
                    for b in r['bloques']:
                        valores = [id_hotel, b['fecha'], b['checkin_nacionales'], b['checkin_extranjeros'],
                                   b['total_turistas'], b['pernoctaciones'], b['habitaciones_ocupadas'],
                                   r['habitaciones_disponibles'], r['habitaciones_disponibles'],
                                   b['tarifa_cobrada'], b['ocupacion_porcentaje']]
                        if con_feriado:
                            valores.append(r['feriado'])
                        if con_plazas:
                            valores += [r['plazas_disponibles'], b['ocupacion_plazas_porcentaje']]
                        cursor.execute(sql_ocupacion, valores)
                        registros_ocupacion += 1

                if id_resp is not None:
                    cursor.execute(
                        "UPDATE respuestas_encuestas SET fecha_etl = NOW() WHERE id_respuesta = %s", (id_resp,)
                    )
                insertados += 1
            except Exception as e:
                errores += 1
                if len(detalles) < 10:
                    detalles.append(f"Fila {fila_num}: {str(e)[:120]}")

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

    if tipo_encuesta == 'hotel' and insertados:
        advertencias.append(
            f"{insertados} encuesta(s) de hotel generaron {registros_ocupacion} registro(s) de ocupación "
            f"(uno por cada fecha del feriado reportada)."
        )
    if omitidos:
        advertencias.append(
            f"{omitidos} respuesta(s) ya habían sido cargadas antes y se omitieron (no se duplican)."
        )
    for campo, cantidad in cont.no_reconocidos.items():
        advertencias.append(
            f"Columna '{campo}': {cantidad} respuesta(s) no coinciden con las opciones que reconoce el ETL "
            f"(¿se editó el cuestionario?); se conservó el texto original o se usó 0."
        )
    if hoteles_creados:
        unicos = list(dict.fromkeys(hoteles_creados))
        advertencias.append(
            f"Se crearon {len(unicos)} hotel(es) nuevo(s) automáticamente: " + ", ".join(unicos[:10])
            + ("..." if len(unicos) > 10 else "")
            + ". Revisa/completa sus datos en Hoteles."
        )

    return {'insertados': insertados, 'errores': errores, 'detalles': detalles,
            'advertencias': advertencias, 'omitidos': omitidos}

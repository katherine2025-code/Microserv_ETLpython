"""
Parser para el formato DEFINITIVO de la Encuesta Turística (export con nombres de
variable limpios + columnas one-hot + GPS). Es el formato que se usa desde el
feriado de Carnaval 2026 en adelante.

A diferencia de los otros parsers (kobo_* y la app), aquí CADA pregunta tiene un
nombre de columna propio y estable (p. ej. 'edad', 'sexo', 'pais_origen',
'gasto_alojamiento', 'nps_volver', 'gps_lat'), así que todo se lee POR NOMBRE, sin
adivinar por posición ni por contenido. Esto evita el desastre del mapeo genérico
posicional (que metía el año en la edad).

Cada fila = una encuesta a un turista -> una fila en encuestas_turisticas.
"""
import json
import re

import pandas as pd

from etl_utils import _reparar_mojibake
from cantones import columnas_de
from kobo_encuestas_turismo import (
    EDAD_MAPEO, PAIS_MAPEO, NOCHES_MAPEO, NIVEL_EDUCATIVO_MAPEO, OCUPACION_MAPEO,
    FRECUENCIA_MAPEO, MOTIVO_MAPEO, GENERO_MAPEO, GASTO_MAPEO_ORD, GASTO_RUBROS,
    PREGUNTAS_COLUMNA, CAMPOS_NUEVOS_ESCALA, LUGARES_CONOCIDOS,
    _mapear_valor, _mapear_multi, _parsear_fecha_start, extraer_lugares,
)

NO_RESPONDER = 'Prefiero no responder'
TOKENS_NULOS = {'', 'nan', 'null', 'none', 'n/a', 'na', '-'}

# Firma del formato: nombres de variable que solo existen en este export.
_COLUMNAS_FIRMA = ('edad', 'sexo', 'pais_origen', 'perfil_viajero', 'nps_volver')

# Mapeo de los campos de opción única/múltiple nuevos (reutiliza los mapeos canónicos del
# parser de Kobo, por campo de la BD).
_MAPEO_POR_CAMPO = {q['campo']: (q['mapeo'], q.get('tipo', 'texto')) for q in PREGUNTAS_COLUMNA}


def es_encuesta_v2(columnas) -> bool:
    nombres = {str(c).strip().lower() for c in columnas}
    return all(c in nombres for c in _COLUMNAS_FIRMA)


def _norm(v):
    return re.sub(r'\s+', ' ', str(v)).strip()


def _valido(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return _norm(v).lower() not in TOKENS_NULOS


def _txt(row, col):
    if col not in row.index:
        return None
    v = row[col]
    return _reparar_mojibake(_norm(v)) if _valido(v) else None


def _map_dict(texto, mapeo, defecto=None):
    """Primer valor del dict cuya clave esté contenida en el texto (minúsculas)."""
    if not texto:
        return defecto
    bajo = texto.lower()
    for clave, val in mapeo.items():
        if clave in bajo:
            return val
    return defecto


def _digito(texto):
    if not texto:
        return 0
    m = re.search(r'[1-5]', texto)
    return int(m.group()) if m else 0


def _entero(texto):
    if not texto:
        return 0
    if re.search(r'solo yo|viaj[eé] solo', texto.lower()):
        return 1
    m = re.search(r'(\d+)', texto)
    return int(m.group(1)) if m else 0


def _decimal(texto):
    if not texto:
        return None
    m = re.search(r'-?\d+(?:[.,]\d+)?', texto.replace(',', '.'))
    return float(m.group()) if m else None


def _pais(row):
    t = _txt(row, 'pais_origen')
    if not t:
        return NO_RESPONDER
    directo = _map_dict(t, PAIS_MAPEO)
    if directo:
        return directo
    if 'otr' in t.lower():  # "Otra (especifique)"
        esp = _txt(row, 'pais_especifique')
        return esp.title() if esp else 'Otro'
    return t.title()


# Las 5 categorías de gasto de ESTE formato -> columnas de la BD. 'transporte' no se separa aquí,
# queda en NULL. gasto_total = suma de los puntos medios de los rubros respondidos.
_GASTO_V2 = {
    'gasto_alojamiento': 'gasto_alojamiento',
    'gasto_alimentacion': 'gasto_alimentacion',
    'gasto_actividades': 'gasto_entretenimiento',
    'gasto_compras': 'gasto_compras',
    'gasto_otros': 'gasto_recuerdos',
}


def extraer_registro_v2(row):
    """Arma el registro de encuestas_turisticas leyendo cada campo por su nombre de columna."""
    reg = {}
    reg['genero'] = _map_dict(_txt(row, 'sexo'), GENERO_MAPEO, NO_RESPONDER)
    edad = _map_dict(_txt(row, 'edad'), EDAD_MAPEO)
    reg['edad'] = int(edad) if edad is not None else 0
    reg['pais_residencia'] = _pais(row)
    reg['ciudad_residencia'] = NO_RESPONDER
    reg['nivel_educativo'] = _map_dict(_txt(row, 'educacion'), NIVEL_EDUCATIVO_MAPEO, NO_RESPONDER)
    reg['ocupacion'] = _map_dict(_txt(row, 'ocupacion'), OCUPACION_MAPEO, NO_RESPONDER)
    reg['tamano_grupo'] = _entero(_txt(row, 'tamanio_grupo'))
    reg['frecuencia_visitas'] = _map_dict(_txt(row, 'frecuencia_visita'), FRECUENCIA_MAPEO, NO_RESPONDER)
    reg['motivo_visita'] = _map_dict(_txt(row, 'motivo_viaje'), MOTIVO_MAPEO) or (_txt(row, 'motivo_viaje') or NO_RESPONDER)[:150]
    reg['noches_estadia'] = _map_dict(_txt(row, 'noches_estadia'), NOCHES_MAPEO) or 0

    # Opción única nuevas (col del archivo -> campo BD), con el mapeo canónico de PREGUNTAS_COLUMNA.
    simples = {
        'companero_viaje': 'con_quien_viaja',
        'facilidad_info': 'facilidad_informacion',
        'metodo_planificacion': 'como_organizo',
        'anticipacion_reserva': 'anticipacion',
        'transporte_llegada': 'transporte_llegada',
        'perfil_viajero': 'tipo_viajero',
        'expectativas_cumplidas': 'expectativas',
        'categoria_gasto_prioritario': 'rubro_mayor_gasto',
    }
    for col, campo in simples.items():
        mapeo = _MAPEO_POR_CAMPO.get(campo, ([], 'texto'))[0]
        reg[campo] = _mapear_valor(_txt(row, col), mapeo) or NO_RESPONDER

    # Selección múltiple (texto con varias opciones) -> opciones canónicas separadas por "; ".
    multis = {
        'atracciones_texto': 'actividades_esperadas',
        'fuentes_info_texto': 'busco_informacion',
        'transporte_local_texto': 'transporte_destino',
        'razon_satisfaccion': 'motivos_recomendacion',
    }
    for col, campo in multis.items():
        mapeo = _MAPEO_POR_CAMPO.get(campo, ([], 'multi'))[0]
        reg[campo] = _mapear_multi(_txt(row, col), mapeo) or NO_RESPONDER

    # Satisfacción por aspecto (escala 1-5) y generales.
    reg['satisfaccion_informacion'] = _digito(_txt(row, 'cal_info_turistica'))
    reg['satisfaccion_atencion'] = _digito(_txt(row, 'cal_atencion_local'))
    reg['satisfaccion_seguridad'] = _digito(_txt(row, 'cal_seguridad'))
    reg['satisfaccion_limpieza'] = _digito(_txt(row, 'cal_limpieza'))
    reg['satisfaccion_calidad_precio'] = _digito(_txt(row, 'cal_relacion_precio'))
    reg['nivel_satisfaccion'] = _digito(_txt(row, 'satisfaccion_atractivos'))
    reg['probabilidad_retorno'] = _digito(_txt(row, 'nps_volver'))

    # Gasto por rubro + total = suma de los 5 rubros (punto medio de cada rango).
    suma, hay = 0.0, False
    for col, campo in _GASTO_V2.items():
        val = _mapear_valor(_txt(row, col), GASTO_MAPEO_ORD)
        reg[campo] = val if val is not None else 0.0
        if val is not None:
            suma += val
            hay = True
    reg['gasto_transporte'] = 0.0  # este formato no separa transporte interno
    reg['gasto_total'] = round(suma, 2) if hay else 0.0

    # Lugares (para el mapa) desde el destino específico, usando la lista conocida.
    reg['lugares_visitados'] = json.dumps(extraer_lugares(_txt(row, 'destino_especifico')), ensure_ascii=False)

    # GPS (lat/lon) del envío, para ubicar la encuesta en el mapa.
    reg['gps_lat'] = _decimal(_txt(row, 'gps_lat'))
    reg['gps_lon'] = _decimal(_txt(row, 'gps_lon'))

    return reg


_CAMPOS_INSERT = [
    'fecha_encuesta', 'genero', 'edad', 'pais_residencia', 'ciudad_residencia', 'nivel_educativo',
    'ocupacion', 'tamano_grupo', 'frecuencia_visitas', 'motivo_visita', 'noches_estadia',
    'gasto_total', 'nivel_satisfaccion', 'probabilidad_retorno', 'lugares_visitados',
    'con_quien_viaja', 'actividades_esperadas', 'busco_informacion', 'facilidad_informacion',
    'como_organizo', 'anticipacion', 'transporte_llegada', 'transporte_destino', 'tipo_viajero',
    'expectativas', 'satisfaccion_informacion', 'satisfaccion_atencion', 'satisfaccion_seguridad',
    'satisfaccion_limpieza', 'satisfaccion_calidad_precio', 'rubro_mayor_gasto',
    'motivos_recomendacion', 'gps_lat', 'gps_lon',
] + list(GASTO_RUBROS)


def procesar_encuestas_v2(df, get_db_connection):
    """Inserta las filas del export v2 en encuestas_turisticas. Dedup por uuid_kobo (_uuid)."""
    df = df.rename(columns=lambda c: str(c).strip())
    advertencias = []

    connection = get_db_connection()
    cursor = connection.cursor()
    insertados, errores, omitidos, detalles = 0, 0, 0, []
    try:
        columnas_bd = columnas_de(cursor, 'encuestas_turisticas')
        campos = [c for c in _CAMPOS_INSERT if c in columnas_bd]
        faltan = [c for c in _CAMPOS_INSERT if c not in columnas_bd]
        if faltan:
            advertencias.append(
                "Reinicia el backend para aplicar la migración de columnas: faltan " + ', '.join(faltan))

        con_uuid = 'uuid_kobo' in columnas_bd
        if con_uuid:
            campos.append('uuid_kobo')
            cursor.execute("SELECT uuid_kobo FROM encuestas_turisticas WHERE uuid_kobo IS NOT NULL")
            existentes = {f['uuid_kobo'] for f in cursor.fetchall()}
        else:
            existentes = set()

        marcas = ', '.join(['%s'] * len(campos))
        sql = f"INSERT INTO encuestas_turisticas ({', '.join(campos)}, fecha_registro) VALUES ({marcas}, NOW())"

        for index, row in df.iterrows():
            fila_num = int(index) + 2
            try:
                uuid_fila = _norm(row.get('_uuid')) if con_uuid and _valido(row.get('_uuid')) else None
                if uuid_fila and uuid_fila in existentes:
                    omitidos += 1
                    continue

                fecha_dt = _parsear_fecha_start(row.get('start'))
                if pd.isna(fecha_dt):
                    errores += 1
                    if len(detalles) < 10:
                        detalles.append(f"Fila {fila_num}: no se pudo leer la fecha de 'start'")
                    continue

                reg = extraer_registro_v2(row)
                reg['fecha_encuesta'] = fecha_dt.strftime('%Y-%m-%d')
                valores = [reg.get(c) for c in campos if c != 'uuid_kobo']
                if con_uuid:
                    valores.append(uuid_fila)
                cursor.execute(sql, valores)
                if uuid_fila:
                    existentes.add(uuid_fila)
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
        advertencias.append(f"{omitidos} fila(s) ya estaban cargadas (mismo uuid) y se omitieron.")
    return {'insertados': insertados, 'errores': errores, 'omitidos': omitidos,
            'detalles': detalles, 'advertencias': advertencias}

"""
Utilidades ETL: lectura de archivos (CSV/XLSX), mapeo posicional de columnas,
normalización de texto y relleno de valores nulos.

El mapeo de columnas es POSICIONAL (por orden), no por nombre de encabezado:
el archivo debe traer las columnas en el orden definido en CANONICAL_SCHEMAS
para cada tipo. Esto evita que cambios de redacción en las preguntas de una
encuesta (nuevas versiones del cuestionario) rompan la carga, siempre que el
orden de las columnas se mantenga.
"""
import io
import re
import pandas as pd

# ==========================================================================
# ESQUEMAS CANÓNICOS (orden fijo de columnas esperado por tipo de archivo)
# ==========================================================================
# Cada entrada: (nombre_campo, tipo_dato, categorias)
#   tipo_dato: 'fecha' | 'int' | 'float' | 'texto' | 'texto_categorico'
#   categorias: lista de valores canónicos válidos (solo para texto_categorico)
CANONICAL_SCHEMAS = {
    'ocupacion': [
        ('fecha', 'fecha', None),
        ('id_hotel', 'int', None),
        ('checkin_nacionales', 'int', None),
        ('checkin_extranjeros', 'int', None),
        ('pernoctaciones', 'int', None),
        ('habitaciones_ocupadas', 'int', None),
        ('tarifa_cobrada', 'float', None),
        ('ocupacion_porcentaje', 'float', None),
    ],
    'encuestas': [
        ('fecha_encuesta', 'fecha', None),
        ('genero', 'texto_categorico', {
            'Masculino': ['masculino', 'hombre', 'male', 'varon', 'varón'],
            'Femenino': ['femenino', 'mujer', 'female'],
        }),
        ('edad', 'int', None),
        ('pais_residencia', 'texto', None),
        ('ciudad_residencia', 'texto', None),
        ('motivo_visita', 'texto', None),
        ('noches_estadia', 'int', None),
        ('gasto_total', 'float', None),
        ('nivel_satisfaccion', 'int', None),
        ('probabilidad_retorno', 'int', None),
    ],
    'clima': [
        ('fecha', 'fecha', None),
        ('temperatura', 'float', None),
        ('humedad', 'float', None),
        ('precipitacion', 'float', None),
        ('velocidad_viento', 'float', None),
        ('descripcion', 'texto', None),
    ],
    'feriados': [
        ('nombre', 'texto', None),
        ('fecha_inicio', 'fecha', None),
        ('fecha_fin', 'fecha', None),
        ('total_dias', 'int', None),
        ('temporada', 'texto_categorico', {
            'Alta': ['alta', 'high'],
            'Media': ['media', 'medium'],
            'Baja': ['baja', 'low'],
        }),
        ('descripcion', 'texto', None),
    ],
}

# Campos de texto libre que, al venir vacíos/nulos, deben reemplazarse por
# "Prefiero no responder" en vez de quedar en blanco. Solo aplica a encuestas
# turísticas (mayoría de respuestas de texto); en ocupación (alojamiento) el
# relleno de nulos numéricos con 0 ya cubre el caso, pues ahí casi todo es
# numérico.
CAMPOS_TEXTO_NULO_ENCUESTA = {'genero', 'pais_residencia', 'ciudad_residencia', 'motivo_visita'}

VALOR_NULO_TEXTO = 'Prefiero no responder'
TOKENS_NULOS = {'', 'nan', 'null', 'none', 'n/a', 'na', '-', 's/n', 'no aplica', 'no especificado'}


# ==========================================================================
# LECTURA DE ARCHIVO (.csv / .xlsx)
# ==========================================================================
def leer_archivo(filename: str, contents: bytes) -> pd.DataFrame:
    """Lee un CSV (separado por ';') o un XLSX y devuelve el DataFrame crudo,
    con todas las celdas como texto (la conversión de tipos ocurre después,
    columna por columna, en limpiar_y_convertir). Prueba encoding utf-8 y,
    si falla, latin-1 (archivos exportados desde Excel/Windows)."""
    nombre = (filename or '').lower()

    if nombre.endswith('.xlsx') or nombre.endswith('.xls'):
        try:
            df = pd.read_excel(io.BytesIO(contents), engine='openpyxl', dtype=str, header=0)
        except Exception as e:
            raise ValueError(f"No se pudo leer el archivo Excel: {e}")
        if df.shape[1] < 1:
            raise ValueError("El archivo Excel no tiene columnas.")
        return df

    if not (nombre.endswith('.csv')):
        raise ValueError("Formato de archivo no soportado. Usa .csv o .xlsx.")

    ultimo_error = None
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            # index_col=False es obligatorio aquí: si alguna fila de datos
            # trae más campos que el encabezado (columnas finales vacías sin
            # su ';' correspondiente, algo común en exports de Kobo), pandas
            # asume por defecto que la(s) columna(s) sobrante(s) son un
            # índice y RECORRE todo el resto de columnas una posición sin
            # avisar - corrompe el archivo entero en silencio. Con
            # index_col=False no crea ese índice implícito.
            df = pd.read_csv(io.BytesIO(contents), sep=';', encoding=encoding,
                              on_bad_lines='skip', dtype=str, index_col=False)
            if df.shape[1] == 1:
                # El separador ';' no aplicó (el archivo viene con coma);
                # se intenta como último recurso para no rechazar el archivo.
                df_coma = pd.read_csv(io.BytesIO(contents), sep=',', encoding=encoding,
                                       on_bad_lines='skip', dtype=str, index_col=False)
                if df_coma.shape[1] > 1:
                    return df_coma
            return df
        except Exception as e:
            ultimo_error = e
            continue
    raise ValueError(f"No se pudo leer el archivo CSV: {ultimo_error}")


# ==========================================================================
# NORMALIZACIÓN DE TEXTO
# ==========================================================================
def _reparar_mojibake(texto: str) -> str:
    """Corrige texto mal decodificado por un mismatch de encoding
    (p.ej. 'CafÃ©' -> 'Café'), típico de exportaciones CSV desde Windows."""
    if any(ch in texto for ch in ('Ã', 'Â', '�')):
        try:
            return texto.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return texto
    return texto


def normalizar_texto(valor, categorias=None):
    """Limpia un valor de texto: quita espacios repetidos, corrige
    codificación y normaliza mayúsculas/minúsculas.

    `categorias` acepta:
      - un dict {valor_canonico: [sinonimos...]} -> empareja el texto contra
        los sinónimos (p.ej. 'hombre'/'male' -> 'Masculino') y así tolera
        variantes de redacción entre versiones distintas del cuestionario.
      - una lista plana de valores canónicos -> empareja contra sí mismos.
    Si no hay coincidencia, devuelve 'Otro'. Devuelve None si el valor
    está vacío/nulo."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip()
    if texto.lower() in TOKENS_NULOS:
        return None

    texto = _reparar_mojibake(texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    if not texto:
        return None

    if categorias:
        texto_lower = texto.lower()
        mapeo = categorias if isinstance(categorias, dict) else {c: [c] for c in categorias}
        for canonico, sinonimos in mapeo.items():
            for sinonimo in sinonimos:
                if sinonimo.lower() == texto_lower or sinonimo.lower() in texto_lower:
                    return canonico
        return 'Otro'

    # Texto libre: Título para unificar "QUITO" / "quito " / "Quito" -> "Quito"
    return texto.title()


def _a_numero(valor, tipo_dato):
    """Convierte a int/float tolerando texto mezclado ('25 años', '$120,50')."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip().lower()
    if texto in TOKENS_NULOS:
        return None
    texto_limpio = texto.replace(',', '.')
    match = re.search(r'-?\d+(\.\d+)?', texto_limpio)
    if not match:
        return None
    try:
        numero = float(match.group())
        return int(round(numero)) if tipo_dato == 'int' else numero
    except ValueError:
        return None


def _a_fecha(valor):
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip()
    if texto.lower() in TOKENS_NULOS:
        return None
    try:
        fecha = pd.to_datetime(texto, dayfirst=True, errors='coerce')
        if pd.isna(fecha):
            return None
        return fecha.strftime('%Y-%m-%d')
    except Exception:
        return None


# ==========================================================================
# MAPEO POSICIONAL + DETECCIÓN DE VERSIÓN
# ==========================================================================
def mapear_posicional(df: pd.DataFrame, tipo: str):
    """Ignora el texto de los encabezados del archivo y asigna cada campo
    según su POSICIÓN, usando el orden fijo de CANONICAL_SCHEMAS. Si el
    archivo trae más o menos columnas de las esperadas (posible versión
    distinta del cuestionario/formulario), lo reporta como advertencia en
    vez de fallar. Devuelve (df_renombrado, advertencias)."""
    esquema = CANONICAL_SCHEMAS[tipo]
    nombres_esperados = [campo for campo, _, _ in esquema]
    advertencias = []

    n_esperadas = len(nombres_esperados)
    n_recibidas = len(df.columns)

    if n_recibidas < n_esperadas:
        faltantes = ', '.join(nombres_esperados[n_recibidas:])
        advertencias.append(
            f"El archivo trae {n_recibidas} columna(s) pero se esperaban {n_esperadas} para "
            f"'{tipo}'. Posible versión anterior del formulario: las columnas faltantes "
            f"({faltantes}) se completaron con valores por defecto."
        )
    elif n_recibidas > n_esperadas:
        advertencias.append(
            f"El archivo trae {n_recibidas} columnas pero solo se esperaban {n_esperadas} para "
            f"'{tipo}'. Posible nueva versión del cuestionario con preguntas adicionales: "
            f"se ignoraron las columnas sobrantes (desde la posición {n_esperadas + 1})."
        )

    df_recortado = df.iloc[:, :n_esperadas].copy()
    df_recortado.columns = nombres_esperados[:df_recortado.shape[1]]

    for nombre_faltante in nombres_esperados[df_recortado.shape[1]:]:
        df_recortado[nombre_faltante] = None

    return df_recortado[nombres_esperados], advertencias


# ==========================================================================
# LIMPIEZA, VALIDACIÓN DE TIPOS Y RELLENO DE NULOS
# ==========================================================================
def limpiar_y_convertir(df: pd.DataFrame, tipo: str):
    """Recorre cada columna según su tipo declarado en CANONICAL_SCHEMAS:
    normaliza texto, convierte números/fechas y detecta inconsistencias de
    formato/tipo de dato (valores presentes que no se pudieron convertir).
    Rellena nulos: numéricos -> 0; texto libre de encuestas -> 'Prefiero
    no responder'; texto categórico sin valor -> 'Otro'.
    Devuelve (df_limpio, advertencias)."""
    esquema = CANONICAL_SCHEMAS[tipo]
    inconsistencias_por_columna = {}

    for campo, tipo_dato, categorias in esquema:
        valores_limpios = []
        for valor in df[campo].tolist():
            if tipo_dato == 'fecha':
                nuevo = _a_fecha(valor)
            elif tipo_dato in ('int', 'float'):
                nuevo = _a_numero(valor, tipo_dato)
            else:
                nuevo = normalizar_texto(valor, categorias)

            valor_original_no_vacio = (
                valor is not None
                and not (isinstance(valor, float) and pd.isna(valor))
                and str(valor).strip().lower() not in TOKENS_NULOS
                and str(valor).strip() != ''
            )
            if nuevo is None and valor_original_no_vacio:
                inconsistencias_por_columna[campo] = inconsistencias_por_columna.get(campo, 0) + 1

            valores_limpios.append(nuevo)

        if tipo_dato in ('int', 'float'):
            valores_limpios = [v if v is not None else 0 for v in valores_limpios]
        elif tipo == 'encuestas' and campo in CAMPOS_TEXTO_NULO_ENCUESTA:
            valores_limpios = [v if v is not None else VALOR_NULO_TEXTO for v in valores_limpios]
        elif tipo_dato == 'texto_categorico':
            valores_limpios = [v if v is not None else 'Otro' for v in valores_limpios]

        df[campo] = valores_limpios

    advertencias = [
        f"Columna '{campo}': {cantidad} valor(es) con formato o tipo de dato inconsistente "
        f"se reemplazaron por un valor por defecto."
        for campo, cantidad in inconsistencias_por_columna.items()
    ]
    return df, advertencias

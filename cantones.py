"""
Cantón al que pertenece cada parroquia (provincia de Santa Elena), para
clasificar hoteles cuando el archivo no trae el cantón explícito.
Mantener sincronizado con backend_ots/src/utils/cantones.js.
"""
import unicodedata

CANTON_SANTA_ELENA = 'Santa Elena'
CANTON_SALINAS = 'Salinas'

_PARROQUIAS = {
    CANTON_SANTA_ELENA: ['Santa Elena', 'Atahualpa', 'Colonche', 'Chanduy', 'Manglaralto',
                         'Simón Bolívar', 'Ancón', 'San José de Ancón'],
    CANTON_SALINAS: ['Salinas', 'Anconcito', 'José Luis Tamayo'],
}


def _clave(texto) -> str:
    texto = unicodedata.normalize('NFD', str(texto or '').strip().lower())
    return ''.join(c for c in texto if unicodedata.category(c) != 'Mn')


_CANTON_POR_PARROQUIA = {_clave(p): canton for canton, ps in _PARROQUIAS.items() for p in ps}
_CANTONES = {_clave(c): c for c in _PARROQUIAS}


def canton_de_parroquia(parroquia):
    return _CANTON_POR_PARROQUIA.get(_clave(parroquia))


def normalizar_canton(texto):
    """'salinas ' / 'SANTA ELENA' -> nombre canónico; None si no es un cantón conocido."""
    return _CANTONES.get(_clave(texto))


def columnas_de(cursor, tabla):
    """Nombres de columnas existentes de una tabla (para escribir solo las que hay
    en la base, sin fallar si el backend aún no aplicó la migración)."""
    cursor.execute(
        "SELECT COLUMN_NAME AS c FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (tabla,)
    )
    return {fila['c'] for fila in cursor.fetchall()}

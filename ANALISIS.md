# ANALISIS.md — Microserv_ETLpython

> Diagnóstico de arquitectura. Solo lectura; ningún .py fue modificado.
> Fecha: oct 2026.

---

## Resumen ejecutivo

El microservicio tiene tres capas bien diferenciadas: **ML** (modelo.py, data_loader.py, variables_estacionales.py), **parsers ETL** (4 archivos por formato de origen) e **infra** (db_config.py, cantones.py). La autenticación X-Service-Token y el arranque asíncrono están correctamente implementados. Los problemas principales son: `app.py` es un monolito de ~970 líneas que mezcla routing con lógica ETL completa; cuatro parsers copian literalmente las mismas helpers (`_norm`, `TOKENS_NULOS`) en lugar de importar de `etl_utils`; dos endpoints (`/metrics`, `/predecir-rango`) no cierran la conexión a BD en `finally`; y se consulta la tabla `predicciones` (declarada en desuso) para devolver un contador que siempre es 0. No hay credenciales hardcodeadas ni imports circulares.

---

## Mapa de dependencias

```
app.py ──────────────────── db_config
        │                   data_loader ─── db_config
        │                                   variables_estacionales ─── db_config
        │
        ├── modelo ───────── data_loader (ver arriba)
        │
        ├── etl_utils        (stdlib: io, re, pandas)
        ├── kobo_encuestas_turismo ── etl_utils, cantones
        ├── kobo_establecimientos ─── etl_utils, cantones
        ├── ots_app_encuestas ──────── cantones, kobo_encuestas_turismo
        ├── parser_encuestas_v2 ────── etl_utils, cantones, kobo_encuestas_turismo
        └── variables_estacionales
```

Hubs (importados por ≥3 archivos propios): `cantones` (×4), `etl_utils` (×4), `kobo_encuestas_turismo` (×3), `db_config` (×3).
Hojas (importados solo por app.py): `modelo`, `kobo_establecimientos`, `ots_app_encuestas`, `parser_encuestas_v2`.

---

## Clasificación de archivos

| Archivo | Categoría | Rol | Hub / Hoja |
|---------|-----------|-----|------------|
| app.py | ENTRADA | Rutas FastAPI, middleware CORS+auth, startup, ETL fallback (procesar_*) | Raíz |
| modelo.py | ML | ModeloPredictor: entrenar RF/XGB/Prophet, predecir, persistir en MODEL_DIR | Hoja |
| data_loader.py | ML | Dataset de entrenamiento: JOIN BD + variables estacionales; fix REPEATABLE READ | Hub ×2 |
| variables_estacionales.py | ML | Genera/lee variables de calendario en BD | Hub ×2 |
| etl_utils.py | PARSERS | Lectura CSV/XLSX, mapeo posicional, normalización de texto | **Hub ×4** |
| kobo_encuestas_turismo.py | PARSERS | Parser formulario Kobo turismo (encuestador) | **Hub ×3** |
| kobo_establecimientos.py | PARSERS | Parser formulario Kobo establecimientos/ocupación hotelera | Hoja |
| ots_app_encuestas.py | PARSERS | Parser CSV de la app OTS (encuestadores internos) | Hoja |
| parser_encuestas_v2.py | PARSERS | Parser encuestas v2 con GPS y etiquetas largas | Hoja |
| db_config.py | INFRA | Única fuente de configuración BD (DB_PARAMS y DB_CONFIG) | **Hub ×3** |
| cantones.py | INFRA | Datos geográficos de referencia (parroquia → cantón) | **Hub ×4** |
| requirements.txt | INFRA | Dependencias Python con pins críticos (numpy==1.26.3) | — |
| Dockerfile | INFRA | Imagen python:3.12-slim + build-essential + CmdStan | — |

---

## Problemas detectados

| # | Problema | Severidad | Archivo:línea | Evidencia |
|---|----------|-----------|---------------|-----------|
| 1 | Lógica ETL completa dentro de app.py | MEDIA | app.py:323-486 | `procesar_ocupacion`, `procesar_clima`, `procesar_feriados`, `procesar_encuestas` son funciones ETL de hasta 60 líneas; no son routing |
| 2 | Conexión BD sin `finally` en `/predecir-rango` | MEDIA | app.py:765-782 | `connection = get_db_connection()` + `cursor.close(); connection.close()` están en el cuerpo del `try`, no en `finally`; si la query falla, la conexión se filtra |
| 3 | Conexión BD sin `finally` en `/metrics` + bare `except:` | MEDIA | app.py:165-170 | `conn.close()` está dentro del `try`, tapado por `except:` desnudo; en error, `conn` queda abierta y la excepción se silencia sin loguear |
| 4 | `_norm()` duplicada idénticamente en 4 archivos | MEDIA | kobo_establecimientos.py:62, kobo_encuestas_turismo.py:70, ots_app_encuestas.py:98, parser_encuestas_v2.py:44 | Mismo cuerpo: `re.sub(r'\s+', ' ', str(texto)).strip()` |
| 5 | `TOKENS_NULOS` duplicada en 3 parsers | MEDIA | kobo_encuestas_turismo.py:56, ots_app_encuestas.py:39, parser_encuestas_v2.py:29 | Conjunto más pequeño que en etl_utils:77; falta `'s/n'`, `'no aplica'`, `'no especificado'` |
| 6 | Importación de helpers privadas (`_mapear_valor`, `_mapear_multi`) entre parsers | MEDIA | ots_app_encuestas.py:32-34, parser_encuestas_v2.py:21 | Rompe encapsulamiento; un refactor de kobo_encuestas_turismo puede romper silenciosamente los otros dos |
| 7 | `/metrics` consulta tabla `predicciones` (en desuso) | BAJA | app.py:167 | `SELECT COUNT(*) FROM predicciones` devuelve siempre 0; dato engañoso en el frontend |
| 8 | `@app.on_event("startup")` deprecado en FastAPI | BAJA | app.py:608 | Deprecado desde FastAPI 0.93; debe migrarse a `lifespan` context manager antes de actualizar la versión |
| 9 | Mezcla de `print()` y `logger.*` para errores | BAJA | app.py en múltiples puntos | Algunos errores van a `logger.warning`, otros a `print(f"❌ ERROR...")`; no hay logging estructurado uniforme |
| 10 | `_parsear_fecha_start` duplicada con variación | BAJA | kobo_establecimientos.py:85 vs kobo_encuestas_turismo.py:80 | La de turismo pasa `utc=True`; la de establecimientos no; diferencia puede ser intencional pero no está documentada |

---

## Deuda técnica priorizada

1. **Extraer las funciones `procesar_*` de `app.py` a un módulo separado** (`etl_handlers.py` o similar). `app.py` tiene ~970 líneas; la mitad es lógica ETL de BD que podría vivir en su propio archivo y ser testeada de forma independiente. Impacto: reduce app.py a su propósito real (routing + startup).

2. **Unificar `_norm` y `TOKENS_NULOS` en etl_utils** y que los parsers los importen desde ahí. Hoy una corrección en uno de los cuatro archivos no se propaga a los demás. Costo: cambio de 4 líneas de import en cada parser; sin riesgo de regresión.

3. **Envolver con `finally` las conexiones BD en `/predecir-rango` y `/metrics`**. Son las únicas dos rutas sin ese patrón; todas las funciones `procesar_*` ya lo hacen bien. Impacto: elimina filtraciones silenciosas al escalar.

4. **Migrar `@app.on_event("startup")` a `lifespan`**. No rompe nada hoy con FastAPI 0.109, pero bloquea cualquier actualización futura de FastAPI sin trabajo adicional. Es un cambio de ~10 líneas.

5. **Eliminar o corregir `total_predicciones` en `/metrics`**. La tabla `predicciones` no recibe escrituras; devolver ese campo con valor 0 puede confundir al frontend o a quien lea los logs. Si el campo no tiene uso real, debe quitarse del response.

---

## Recomendación de reorganización

**No vale la pena reorganizar en carpetas ahora.** Con 11 archivos fuente el flat root es navegable, y mover a `etl/`, `ml/`, `parsers/` requiere actualizar todos los imports en cada archivo, probarlo en Docker, y potencialmente romper las rutas relativas de modelos. El ROI es bajo frente al riesgo en un proyecto sin tests. Lo que sí conviene (menor riesgo, mismo beneficio) es consolidar las duplicaciones de `_norm` y `TOKENS_NULOS` para que todos los parsers importen de `etl_utils`.

---

## Lo que NO se debe tocar

Según [CLAUDE.md](CLAUDE.md):

- **`db_config.py`**: no activar `autocommit`; los parsers dependen del rollback implícito en caso de error.
- **`requirements.txt` — numpy==1.26.3**: pin crítico; Prophet instala numpy 2.x que rompe sklearn y XGBoost.
- **`Dockerfile` — `build-essential`**: necesario para compilar CmdStan (Prophet); no eliminar aunque parezca superfluo.
- **`data_loader.py` — `connection.commit()` tras `generar_variables_estacionales()`**: fix del bug REPEATABLE READ; quitarlo hace que el primer entrenamiento falle con columnas `object` en XGBoost.
- **`modelo.py` — `MODEL_DIR`**: debe ser un volumen Docker persistente; si se cambia la lógica de rutas, verificar que la resolución relativa siga funcionando desde dentro del contenedor.

---

## Preguntas abiertas

1. ¿La importación de `_mapear_valor` y `_mapear_multi` (funciones privadas, con guion bajo) desde `kobo_encuestas_turismo` en `ots_app_encuestas` y `parser_encuestas_v2` es intencional o un acoplamiento accidental que debería formalizarse (renombrándolas sin guion bajo o moviéndolas a `etl_utils`)?
2. ¿La diferencia en `_parsear_fecha_start` — `utc=True` en `kobo_encuestas_turismo.py` vs. ausente en `kobo_establecimientos.py` — es intencional (las encuestas turísticas traen zona horaria en el timestamp y los establecimientos no)?
3. ¿Por qué `ots_app_encuestas.py` no usa `_reparar_mojibake` de `etl_utils` aunque hace normalización de texto similar a los otros parsers?
4. ¿El contador `total_predicciones` en `/metrics` tiene algún consumidor real en el frontend, o es un campo heredado que se puede retirar?
5. ¿Hay plan de actualizar FastAPI más allá de 0.109? Si sí, la migración de `@app.on_event` a `lifespan` debería hacerse antes.

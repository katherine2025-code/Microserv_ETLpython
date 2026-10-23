# CLAUDE.md — Microserv_ETLpython

## Stack real
- **Python 3.12**, **FastAPI 0.109**, **uvicorn 0.27**
- pandas 2.1, scikit-learn 1.3.2, XGBoost 2.0.3, Prophet 1.4.0
- pymysql 1.1 (conexiones explícitas, sin SQLAlchemy sessions)
- **numpy fijo en 1.26.3** — Prophet instala numpy 2.x que rompe sklearn/xgboost; pin crítico
- Sin package.json; todo es Python. Dockerfile base: `python:3.12-slim`

## Estructura clave
```
/                         # Todo en la raíz (sin carpetas src/)
  app.py                  # Entrada FastAPI: rutas, middleware CORS, auth X-Service-Token,
                          # startup (carga modelos en hilo secundario)
  modelo.py               # ModeloPredictor: entrenar, predecir, cargar/guardar desde MODEL_DIR
  data_loader.py          # Construye el dataset ML leyendo la BD; fix REPEATABLE READ incluido
  db_config.py            # DB_PARAMS y DB_CONFIG (DictCursor); fuente única de conexión
  etl_utils.py            # Schemas canónicos, leer_archivo, mapear_posicional, limpiar_y_convertir
  kobo_establecimientos.py # Parser formularios Kobo (hoteles/establecimientos)
  kobo_encuestas_turismo.py# Parser formularios Kobo (encuestas turísticas)
  ots_app_encuestas.py    # Parser CSV de la app interna OTS
  parser_encuestas_v2.py  # Parser v2 de encuestas
  variables_estacionales.py# Genera y recupera variables estacionales en la BD
  cantones.py             # Datos geográficos de referencia
  modelos/                # MODEL_DIR: .pkl + modelo_ganador.json (volumen Docker persistente)
```

## Comandos
| Tarea | Comando |
|-------|---------|
| Dev (hot-reload) | `uvicorn app:app --reload --port 5000` |
| Producción | `uvicorn app:app --host 0.0.0.0 --port 5000` (o vía Docker) |
| Test | ⚠️ CONFIRMAR: no hay pytest ni scripts de test |
| Lint | ⚠️ CONFIRMAR: no hay flake8/ruff configurado |

## Variables de entorno críticas
Ver `.env.example`. Variables obligatorias:
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` (base = `ots`)
- `SERVICE_TOKEN` — **obligatorio**; toda request salvo `/health` debe traer `X-Service-Token` igual a este valor
- `MODEL_DIR` — carpeta de modelos (dev: `./modelos`; Docker: `/app/modelos` montado como volumen)
- `CORS_ORIGINS` — opcional; orígenes extra separados por coma

## Convenciones del código
- **Auth**: middleware `@app.middleware("http")` verifica `X-Service-Token` antes de cada request (excepto `/health`)
- **Conexiones BD**: `pymysql.connect(**DB_PARAMS)` con `try/finally` para cerrar; nunca dejar conexiones abiertas
- **Sin autocommit** — los parsers dependen del rollback implícito en caso de error; no activar
- **REPEATABLE READ fix**: en `data_loader.py`, después de `generar_variables_estacionales()` se hace `connection.commit()` para que el siguiente SELECT vea las filas nuevas
- Modelos ML se persisten en `MODEL_DIR` con `joblib`; `modelo_ganador.json` guarda metadatos
- `/status` y `/health` indican si hay modelo cargado, cuál es el activo y su precisión
- Pydantic v2 para los body de requests

## Restricciones
- **No activar autocommit** en `db_config.py`: rompe los parsers que hacen rollback
- `numpy==1.26.3` **no subir**: Prophet + sklearn + xgboost son incompatibles con numpy 2.x
- `MODEL_DIR` debe ser un volumen Docker persistente; sin él el modelo se pierde en cada redeploy
- CmdStan se compila en la imagen Docker; no quitar `build-essential` del Dockerfile

## Estado actual (oct 2026)
- En `main` (`c5e0af1`)
- El endpoint `/predicciones-historicas` devuelve `{ predicciones: [...] }` para el dashboard del backend
- El endpoint `/predecir` ya no intenta guardar en la tabla `predicciones`

## Pendientes conocidos
- Sin tests (ningún archivo `test_*.py`)
- Sin README (el archivo no existe)
- `app.py` maneja rutas y lógica mezcladas — podría dividirse en routers FastAPI
- No hay validación de schema en los archivos CSV/Excel subidos antes del parseo

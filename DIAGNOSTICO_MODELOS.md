# Modelos ML: fuera Prophet, entran Ridge + Baseline

**Decisión (Katherine, 25-nov-2026):** la comparación de modelos cambia a **Ridge, Random Forest
y XGBoost** (3 modelos reales de familias distintas) más un **Baseline ingenuo** como referencia.
Se **retira Prophet**: con ~19 fechas sueltas estaba fuera de su terreno (brilla con series diarias
largas), daba los peores números y arrastraba dependencias frágiles (CmdStan/numpy) que hacían
pesado y quebradizo el microservicio.

## Por qué el Baseline no es un "modelo"
El Baseline predice SIEMPRE el promedio de ocupación (ignora las variables). Es el **piso a vencer**:
si un modelo real no le gana, no está aportando nada. Se **calcula y reporta**, pero **nunca** se
elige como ganador para predecir.

## Archivos cambiados

| Archivo | Cambio | Motivo |
|---|---|---|
| `modelo.py` | Quita Prophet (import, 3 métodos `_*_prophet`, bloque de entrenamiento, rama de `predecir`, carga). Agrega **Ridge** (`StandardScaler`+`Ridge(alpha=1.0)`) y **Baseline** (`DummyRegressor(mean)`). Ganador se elige entre Ridge/RF/XGBoost (baseline excluido). | El corazón del cambio de modelos. |
| `app.py` | Textos/comentarios: "Ridge vs RF vs XGBoost (con baseline)"; forma de `modelo.metricas` actualizada. | Cosmético, sin cambio de lógica. |
| `requirements.txt` | Quita `prophet==1.4.0`. Mantiene el pin `numpy==1.26.3` (por los wheels de sklearn/xgboost). | Menos dependencias. |
| `Dockerfile` | Quita `build-essential` y la compilación de **CmdStan** (eran solo para Prophet). | Imagen más liviana y robusta. |

> El backend Node y la BD **no se tocan**: el controlador de predicciones es un proxy transparente
> y la tabla `predicciones` (enum `modelo_utilizado`) está en desuso.

## Formato del JSON de comparación (lo consume el front)
`generar_comparacion()` ahora devuelve: `ganador`, **`baseline`**, **`ridge`**, `random_forest`,
`xgboost`, `diferencia_precision`. (Antes tenía `prophet` en vez de `ridge`/`baseline`.) El front
Angular ya quedó actualizado en paralelo (rama `feat/predicciones-modelos` en `ots_front`).

## Pendiente de despliegue
1. **Redesplegar el microservicio** con esta rama (la imagen ya no instala CmdStan → build más rápido).
2. **RE-ENTRENAR** tras desplegar (`POST /entrenar` desde la pantalla de Predicciones, Super Admin),
   para generar los nuevos `.pkl` (`modelo_ridge_ots.pkl`, `modelo_baseline_ots.pkl`) en el volumen
   persistente `/app/modelos`. Los `.pkl` viejos (incluido Prophet) quedan ignorados.
3. Verificar en la pantalla de Predicciones que se vean las 4 tarjetas (RF, XGBoost, Ridge, Baseline).

> Importante: NO se cambió la corrección de **fuga de datos** ni la **validación honesta** (GroupKFold
> por fecha). El techo de R² (~0.24) lo pone la cantidad de fechas, no el modelo.

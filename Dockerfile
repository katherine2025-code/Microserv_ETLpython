# python:3.12-slim (Debian), no alpine: numpy/scikit-learn/xgboost tienen wheels precompilados
# para glibc pero no siempre para musl - con alpine terminan compilando desde cero (lento y con
# más forma de fallar).
FROM python:3.12-slim

# Nota: ya NO se instala build-essential ni CmdStan. Eran necesarios SOLO para Prophet, que se
# retiró del proyecto (los modelos ahora son Ridge, Random Forest y XGBoost + un baseline, todos
# con wheels precompilados). La imagen queda más liviana y con menos piezas que puedan fallar.

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Modelos de ML: /app/modelos DEBE montarse como volumen persistente ──────────────────────
# Ahí se guardan los modelos entrenados (.pkl + modelo_ganador.json). Cada redeploy crea un
# contenedor nuevo desde esta imagen, que NO trae modelos (.dockerignore los excluye), así que
# sin volumen el modelo se pierde y el servicio reentrena al arrancar (minutos, y solo si la BD
# tiene al menos 30 registros). En Dokploy: Advanced -> Volumes/Mounts -> Volume Mount con
# mount path /app/modelos.
# No se usa la instrucción VOLUME a propósito: crearía un volumen ANÓNIMO distinto en cada
# contenedor, que tampoco sobrevive a un redeploy - hace falta un volumen con nombre.
ENV MODEL_DIR=/app/modelos
RUN mkdir -p /app/modelos

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

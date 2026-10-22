# python:3.12-slim (Debian), no alpine: numpy/scikit-learn/xgboost tienen wheels precompilados
# para glibc pero no siempre para musl - con alpine terminan compilando desde cero (lento y con
# más forma de fallar).
FROM python:3.12-slim

# build-essential (gcc/g++/make): Prophet corre sobre CmdStan, que se compila en C++. Se instala
# aquí y NO se quita después: si el modelo llega a recompilarse en tiempo de ejecución (primer
# uso de Prophet tras un cambio), necesita el compilador disponible en la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # Prophet instala numpy 2.x como dependencia; scikit-learn/xgboost de este proyecto NO son
    # compatibles con numpy 2.x (ver nota en requirements.txt) - se refuerza el pin aquí también,
    # por si el resolver de pip llegara a subirlo igual.
    && pip install --no-cache-dir "numpy==1.26.3"

# Compila CmdStan (el motor de Prophet) UNA SOLA VEZ, aquí en el build de la imagen. Sin esto, el
# primer entrenamiento en el contenedor tardaría varios minutos compilándolo en caliente.
RUN python -m cmdstanpy.install_cmdstan --verbose

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

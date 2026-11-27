-- Migración para que el ETL conserve campos que antes se perdían (ver ots_app_encuestas.py).
-- `lugares_visitados` YA existe (columna JSON), así que el fix de lugares aplica sin esto.
-- Estas dos columnas son NUEVAS; sin ellas el código no las puebla (es seguro, hace no-op).
-- Ejecutar una sola vez en la base de datos; o agregarlas al esquema automático del backend
-- (backend_ots/src/services/esquemaService.js -> agregarColumna).

ALTER TABLE encuestas_turisticas
  ADD COLUMN cedula_encuestador VARCHAR(20) NULL COMMENT 'Quien levanto la encuesta (trazabilidad, datos de la app)';

ALTER TABLE encuestas_turisticas
  ADD COLUMN probabilidad_recomendar TINYINT NULL COMMENT 'Prob. de recomendar 1-5 (antes solo iba mal-etiquetado en probabilidad_retorno)';

-- Nota: tras agregarlas, RE-CARGAR las encuestas de la app (o re-ejecutar el ETL) para poblarlas.
-- Y revisar que la descarga de "datos limpios" incluya estas columnas en su SELECT.

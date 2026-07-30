import pandas as pd
import numpy as np
from datetime import datetime
import re

def transformar_csv_kobo(input_csv, output_csv):
    """
    Transforma el CSV de KoboToolbox a formato de ocupación hotelera
    """
    print(f"\n Leyendo CSV: {input_csv}")
    
    # Leer CSV con separador ;
    df = pd.read_csv(input_csv, sep=';', encoding='utf-8', on_bad_lines='skip')
    
    print(f"    Registros leídos: {len(df)}")
    print(f"    Columnas: {len(df.columns)}")
    
    registros_ocupacion = []
    
    for idx, row in df.iterrows():
        try:
            # Extraer información del establecimiento
            parroquia = str(row.get('Pregunta_5', '')).lower()
            habitaciones_disponibles = int(row.get('Pregunta_6', 0) or 0)
            
            # Determinar id_hotel según parroquia
            if 'manglaralto' in parroquia:
                id_hotel = 1
            elif 'colonche' in parroquia:
                id_hotel = 2
            elif 'santa elena' in parroquia:
                id_hotel = 3
            else:
                id_hotel = 1
            
            # Procesar cada fecha (1 a 5)
            for i in range(1, 6):
                # Nombre de la columna de fecha
                fecha_col = f'Fecha {i}' if i == 1 else f'Fecha {i}'
                
                if fecha_col not in df.columns:
                    continue
                
                fecha_valor = row.get(fecha_col)
                if pd.isna(fecha_valor) or str(fecha_valor) == 'nan' or str(fecha_valor).strip() == '':
                    continue
                
                # Parsear fecha
                try:
                    fecha_str = str(fecha_valor).strip()
                    # Intentar diferentes formatos
                    for fmt in ['%d/%m/%Y %H:%M', '%d/%m/%Y', '%Y-%m-%d']:
                        try:
                            fecha = datetime.strptime(fecha_str, fmt)
                            break
                        except:
                            fecha = None
                    
                    if fecha is None:
                        print(f"    Fecha inválida en fila {idx}, fecha {i}: {fecha_str}")
                        continue
                    
                    # Validar que la fecha sea real
                    if fecha.year < 2020 or fecha.year > 2030:
                        print(f"    Fecha fuera de rango: {fecha}")
                        continue
                        
                except Exception as e:
                    print(f"    Error parseando fecha en fila {idx}, fecha {i}: {e}")
                    continue
                
                # Extraer datos de ocupación para esta fecha
                # Las columnas tienen nombres como "**Pernoctaciones **" para Fecha 1
                # y "fila_1-Pernoctaciones " para Fecha 2, etc.
                
                if i == 1:
                    pernoctaciones_col = '**Pernoctaciones **'
                    habitaciones_col = '**Habitaciones ocupadas **'
                    tarifa_col = '**Tarifa cobrada**'
                else:
                    # Para fechas 2-5, buscar columnas con patrón "fila_X-"
                    pernoctaciones_col = f'fila_{i-1}-Pernoctaciones '
                    habitaciones_col = f'fila_{i-1}-Habitaciones ocupadas '
                    tarifa_col = f'fila_{i-1}-Tarifa cobrada'
                
                # Obtener valores
                pernoctaciones = int(row.get(pernoctaciones_col, 0) or 0)
                habitaciones_ocupadas = int(row.get(habitaciones_col, 0) or 0)
                
                # Limpiar tarifa (puede tener símbolos)
                tarifa_str = str(row.get(tarifa_col, '0') or '0')
                tarifa_cobrada = float(re.sub(r'[^\d.]', '', tarifa_str) or 0)
                
                # Check-ins (nacionales y extranjeros)
                checkin_nacionales = int(row.get('Pregunta de turistas /Nacionales', 0) or 0)
                checkin_extranjeros = int(row.get('Pregunta de turistas /Extranjeros', 0) or 0)
                
                # Calcular ocupación porcentual
                if habitaciones_disponibles > 0:
                    ocupacion_porcentaje = round((habitaciones_ocupadas / habitaciones_disponibles) * 100, 2)
                else:
                    ocupacion_porcentaje = 0
                
                # Validar que los datos tengan sentido
                if habitaciones_ocupadas < 0 or pernoctaciones < 0:
                    continue
                
                # Crear registro
                registro = {
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'id_hotel': id_hotel,
                    'checkin_nacionales': checkin_nacionales,
                    'checkin_extranjeros': checkin_extranjeros,
                    'pernoctaciones': pernoctaciones,
                    'habitaciones_ocupadas': habitaciones_ocupadas,
                    'tarifa_cobrada': tarifa_cobrada,
                    'ocupacion_porcentaje': ocupacion_porcentaje
                }
                
                registros_ocupacion.append(registro)
        
        except Exception as e:
            print(f"    Error procesando fila {idx}: {e}")
            continue
    
    # Crear DataFrame final
    df_ocupacion = pd.DataFrame(registros_ocupacion)
    
    print(f"\n Registros de ocupación generados: {len(df_ocupacion)}")
    
    if len(df_ocupacion) > 0:
        print(f"    Rango de fechas: {df_ocupacion['fecha'].min()} a {df_ocupacion['fecha'].max()}")
        print(f"    Hoteles únicos: {df_ocupacion['id_hotel'].nunique()}")
        print(f"    Ocupación promedio: {df_ocupacion['ocupacion_porcentaje'].mean():.2f}%")
    
    # Guardar CSV limpio
    df_ocupacion.to_csv(output_csv, index=False, sep=';', encoding='utf-8')
    print(f"\n CSV guardado en: {output_csv}")
    
    return df_ocupacion

# Ejecutar transformación
if __name__ == "__main__":
    df_transformado = transformar_csv_kobo(
        'D:/ots/ETL/ESTABLECIMIENTOS_DE_ALOJAMIENTO_-_all_versions_-_labels_-_2026-02-19-16-22-25.csv',
        'D:/ots/ETL/ocupacion_hotelera_limpia.csv'
    )
    
    print("\n Primeros 10 registros:")
    print(df_transformado.head(10)9
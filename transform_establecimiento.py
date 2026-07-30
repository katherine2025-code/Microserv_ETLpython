import pandas as pd
import numpy as np
from datetime import datetime
import re

def transformar_establecimientos_a_ocupacion(input_csv, output_csv=None):
    """
    Transforma el CSV de establecimientos de alojamiento a formato de ocupación hotelera
    Convierte las múltiples fechas en filas separadas
    """
    print(f"\n Transformando establecimientos desde: {input_csv}")
    
    # Leer CSV con separador ;
    df = pd.read_csv(input_csv, sep=';', encoding='utf-8', on_bad_lines='skip')
    
    print(f"   Registros leídos: {len(df)}")
    
    registros_ocupacion = []
    
    for idx, row in df.iterrows():
        try:
            # Extraer información del establecimiento
            nombre_parroquia = str(row.get('parroquia', 'santa elena')).lower()
            
            # Determinar id_hotel basado en la parroquia
            if 'manglaralto' in nombre_parroquia:
                id_hotel = 1  # Montañita
            elif 'colonche' in nombre_parroquia:
                id_hotel = 2  # Ayangue
            elif 'santa elena' in nombre_parroquia:
                id_hotel = 3
            else:
                id_hotel = 1  # Por defecto
            
            # Obtener habitaciones disponibles
            habitaciones_disponibles = int(row.get('habitaciones_disponibles', 0) or 0)
            if habitaciones_disponibles == 0:
                habitaciones_disponibles = int(row.get('plazas_disponibles', 20) or 20) // 2
            
            # Procesar cada fecha (Fecha 1, Fecha 2, etc.)
            for i in range(1, 6):  # Fechas 1 a 5
                fecha_col = f'Fecha {i}' if i == 1 else f'Fecha {i}'
                
                if fecha_col not in row or pd.isna(row[fecha_col]):
                    continue
                
                # Parsear fecha
                try:
                    fecha_str = str(row[fecha_col])
                    if not fecha_str or fecha_str == 'nan':
                        continue
                    
                    # Intentar diferentes formatos de fecha
                    for fmt in ['%d/%m/%Y %H:%M', '%d/%m/%Y', '%Y-%m-%d']:
                        try:
                            fecha = datetime.strptime(fecha_str.split(';')[0] if ';' in fecha_str else fecha_str, fmt)
                            break
                        except:
                            fecha = datetime.now()
                except:
                    continue
                
                # Extraer datos de ocupación para esta fecha
                suffix = '' if i == 1 else f'_{i-1}'
                
                # Pernoctaciones
                pernoctaciones_col = f'**Pernoctaciones **{suffix}' if suffix else '**Pernoctaciones **'
                pernoctaciones = int(row.get(pernoctaciones_col, 0) or 0)
                
                # Habitaciones ocupadas
                habitaciones_col = f'**Habitaciones ocupadas **{suffix}' if suffix else '**Habitaciones ocupadas **'
                habitaciones_ocupadas = int(row.get(habitaciones_col, 0) or 0)
                
                # Tarifa cobrada
                tarifa_col = f'**Tarifa cobrada**{suffix}' if suffix else '**Tarifa cobrada**'
                tarifa_str = str(row.get(tarifa_col, '0') or '0')
                tarifa_cobrada = float(re.sub(r'[^\d.]', '', tarifa_str) or 0)
                
                # Calcular ocupación porcentual
                if habitaciones_disponibles > 0:
                    ocupacion_porcentaje = round((habitaciones_ocupadas / habitaciones_disponibles) * 100, 2)
                else:
                    ocupacion_porcentaje = 0
                
                # Extraer check-ins (nacionales + extranjeros)
                checkin_nacionales = 0
                checkin_extranjeros = 0
                
                # Buscar en las columnas de turistas
                for col in df.columns:
                    if 'nacional' in str(col).lower() and pd.notna(row[col]):
                        try:
                            val = int(row[col])
                            if i == 1:
                                checkin_nacionales = val
                            elif i == 2 and 'Fecha 2' in str(col):
                                checkin_nacionales = val
                        except:
                            pass
                
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
            print(f"    Error procesando fila {idx}: {str(e)}")
            continue
    
    # Crear DataFrame final
    df_ocupacion = pd.DataFrame(registros_ocupacion)
    
    print(f"   Registros de ocupación generados: {len(df_ocupacion)}")
    print(f"   Rango de fechas: {df_ocupacion['fecha'].min()} a {df_ocupacion['fecha'].max()}")
    
    # Guardar CSV limpio
    if output_csv:
        df_ocupacion.to_csv(output_csv, index=False, sep=';', encoding='utf-8')
        print(f"    CSV guardado en: {output_csv}")
    
    return df_ocupacion

# Ejecutar transformación
if __name__ == "__main__":
    df_transformado = transformar_establecimientos_a_ocupacion(
        'D:/ots/ETL/ESTABLECIMIENTOS_DE_ALOJAMIENTO_-_all_versions_-_labels_-_2026-02-19-16-22-25.csv',
        'D:/ots/ETL/ocupacion_hotelera_transformada.csv'
    )
    
    print("\nPrimeros 5 registros:")
    print(df_transformado.head())
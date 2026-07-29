import pandas as pd
import numpy as np
from datetime import datetime
import re

def transformar_ocupacion(input_csv, output_csv=None):
    """
    Transforma el CSV de ocupación hotelera a un formato limpio
    """
    print(f"\n Transformando ocupación desde: {input_csv}")
    
    # Leer CSV
    df = pd.read_csv(input_csv, sep=';', encoding='utf-8', on_bad_lines='skip')
    
    print(f" Registros leídos: {len(df)}")
    
    # 1. LIMPIEZA DE FECHA
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['fecha'] = df['fecha'].dt.strftime('%Y-%m-%d')
    
    # 2. LIMPIEZA DE ID_HOTEL
    if 'id_hotel' not in df.columns:
        df['id_hotel'] = 1  # Valor por defecto
    
    # 3. LIMPIEZA DE CHECK-INS
    for col in ['checkin_nacionales', 'checkin_extranjeros']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0
    
    # 4. CÁLCULO DE PERNOCTACIONES (si no existe)
    if 'pernoctaciones' not in df.columns:
        df['pernoctaciones'] = (df['checkin_nacionales'] + df['checkin_extranjeros']) * 2.5
    
    # 5. CÁLCULO DE HABITACIONES OCUPADAS (si no existe)
    if 'habitaciones_ocupadas' not in df.columns:
        df['habitaciones_ocupadas'] = (df['pernoctaciones'] / 2).astype(int)
    
    # 6. LIMPIEZA DE TARIFA
    if 'tarifa_cobrada' in df.columns:
        df['tarifa_cobrada'] = pd.to_numeric(df['tarifa_cobrada'], errors='coerce').fillna(0)
    else:
        df['tarifa_cobrada'] = 80.0  # Valor promedio
    
    # 7. CÁLCULO DE OCUPACIÓN PORCENTAJE (si no existe)
    if 'ocupacion_porcentaje' not in df.columns:
        # Asumiendo 100 habitaciones disponibles
        df['ocupacion_porcentaje'] = (df['habitaciones_ocupadas'] / 100 * 100).round(2)
    
    # 8. SELECCIONAR COLUMNAS FINALES
    columnas_finales = [
        'fecha', 'id_hotel', 'checkin_nacionales', 'checkin_extranjeros',
        'pernoctaciones', 'habitaciones_ocupadas', 'tarifa_cobrada', 'ocupacion_porcentaje'
    ]
    
    df_limpio = df[columnas_finales].copy()
    
    # Eliminar filas con fechas inválidas
    df_limpio = df_limpio.dropna(subset=['fecha'])
    
    print(f" Registros después de limpieza: {len(df_limpio)}")
    
    # Guardar CSV limpio
    if output_csv:
        df_limpio.to_csv(output_csv, index=False, sep=';')
        print(f" CSV limpio guardado en: {output_csv}")
    
    return df_limpio

# Ejemplo de uso
if __name__ == "__main__":
    transformar_ocupacion(
        'D:/ots/datos/ocupacion_raw.csv',
        'D:/ots/datos/ocupacion_limpio.csv'
    )
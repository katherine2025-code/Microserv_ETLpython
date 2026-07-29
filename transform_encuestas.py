import pandas as pd
import numpy as np
from datetime import datetime
import re

def transformar_encuestas(input_csv, output_csv=None):
    """
    Transforma el CSV de encuestas turísticas a un formato limpio
    """
    print(f"\n Transformando encuestas desde: {input_csv}")
    
    # Leer CSV
    df = pd.read_csv(input_csv, sep=';', encoding='utf-8', on_bad_lines='skip')
    
    print(f" Registros leídos: {len(df)}")
    print(f" Columnas originales: {df.columns.tolist()}")
    
    # 1. LIMPIEZA DE FECHA
    if 'fecha_encuesta' in df.columns or 'start' in df.columns:
        fecha_col = 'fecha_encuesta' if 'fecha_encuesta' in df.columns else 'start'
        df['fecha_encuesta'] = pd.to_datetime(df[fecha_col], errors='coerce', dayfirst=False)
        df['fecha_encuesta'] = df['fecha_encuesta'].dt.strftime('%Y-%m-%d')
    
    # 2. LIMPIEZA DE GÉNERO
    if 'genero' in df.columns or 'sexo' in df.columns or 'mujer' in str(df.columns).lower():
        genero_col = [c for c in df.columns if 'genero' in c.lower() or 'sexo' in c.lower() or 'mujer' in c.lower()][0]
        
        def limpiar_genero(valor):
            if pd.isna(valor):
                return 'No especificado'
            valor_str = str(valor).lower()
            if any(x in valor_str for x in ['mujer', 'femenino', 'female']):
                return 'Femenino'
            elif any(x in valor_str for x in ['hombre', 'masculino', 'male']):
                return 'Masculino'
            return 'Otro'
        
        df['genero'] = df[genero_col].apply(limpiar_genero)
    
    # 3. LIMPIEZA DE EDAD
    if 'edad' in df.columns or 'años' in str(df.columns).lower():
        edad_col = [c for c in df.columns if 'edad' in c.lower() or 'años' in c.lower()][0]
        
        def limpiar_edad(valor):
            if pd.isna(valor):
                return 30  # Valor por defecto
            # Extraer número
            numeros = re.findall(r'\d+', str(valor))
            if numeros:
                edad = int(numeros[0])
                if 5 <= edad <= 120:
                    return edad
            return 30  # Valor por defecto
        
        df['edad'] = df[edad_col].apply(limpiar_edad)
    
    # 4. LIMPIEZA DE PAÍS
    if 'pais' in str(df.columns).lower() or 'country' in str(df.columns).lower():
        pais_col = [c for c in df.columns if 'pais' in c.lower() or 'country' in c.lower()][0]
        df['pais_residencia'] = df[pais_col].fillna('Ecuador')
    
    # 5. LIMPIEZA DE SATISFACCIÓN
    if 'satisfaccion' in str(df.columns).lower() or 'satisfecho' in str(df.columns).lower():
        sat_col = [c for c in df.columns if 'satisfaccion' in c.lower() or 'satisfecho' in c.lower()][0]
        
        def limpiar_satisfaccion(valor):
            if pd.isna(valor):
                return 3
            valor_str = str(valor).lower()
            if 'muy satisfecho' in valor_str or '5' in valor_str:
                return 5
            elif 'satisfecho' in valor_str or '4' in valor_str:
                return 4
            elif 'ni satisfecho' in valor_str or '3' in valor_str:
                return 3
            elif 'insatisfecho' in valor_str or '2' in valor_str:
                return 2
            elif 'muy insatisfecho' in valor_str or '1' in valor_str:
                return 1
            return 3
        
        df['nivel_satisfaccion'] = df[sat_col].apply(limpiar_satisfaccion)
    
    # 6. LIMPIEZA DE GASTO
    if 'gasto' in str(df.columns).lower():
        gasto_col = [c for c in df.columns if 'gasto' in c.lower()][0]
        
        def limpiar_gasto(valor):
            if pd.isna(valor):
                return 0.0
            # Remover símbolos de moneda
            valor_str = re.sub(r'[^\d.,]', '', str(valor))
            valor_str = valor_str.replace(',', '.')
            try:
                return float(valor_str)
            except:
                return 0.0
        
        df['gasto_total'] = df[gasto_col].apply(limpiar_gasto)
    
    # 7. SELECCIONAR SOLO COLUMNAS NECESARIAS
    columnas_finales = [
        'fecha_encuesta', 'genero', 'edad', 'pais_residencia',
        'nivel_satisfaccion', 'gasto_total'
    ]
    
    # Agregar columnas que existan
    cols_existentes = [c for c in columnas_finales if c in df.columns]
    df_limpio = df[cols_existentes].copy()
    
    # Agregar columnas faltantes con valores por defecto
    for col in columnas_finales:
        if col not in df_limpio.columns:
            if col == 'edad':
                df_limpio[col] = 30
            elif col in ['nivel_satisfaccion', 'gasto_total']:
                df_limpio[col] = 0
            else:
                df_limpio[col] = 'No especificado'
    
    # Eliminar filas con fechas inválidas
    df_limpio = df_limpio.dropna(subset=['fecha_encuesta'])
    
    print(f" Registros después de limpieza: {len(df_limpio)}")
    print(f" Columnas finales: {df_limpio.columns.tolist()}")
    
    # Guardar CSV limpio
    if output_csv:
        df_limpio.to_csv(output_csv, index=False, sep=';')
        print(f" CSV limpio guardado en: {output_csv}")
    
    return df_limpio

# Ejemplo de uso
if __name__ == "__main__":
    transformar_encuestas(
        'D:/ots/datos/encuestas_raw.csv',
        'D:/ots/datos/encuestas_limpio.csv'
    )
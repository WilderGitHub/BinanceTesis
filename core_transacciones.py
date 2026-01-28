"""
================================================================================
CORE_TRANSACCIONES.PY - Módulo Base para Análisis de Mercado P2P Binance
================================================================================
Contiene la lógica central de:
- Detección de transacciones
- Corrección de duplicación de delta_completedOrders
- Filtros de calidad
- Configuraciones compartidas

Este módulo es importado por:
- generar_velas.py
- eda_completo.py

Autor: Para tesis de maestría en finanzas
Fecha: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import gc
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Campo para validación de órdenes completadas
CAMPO_VALIDACION_ORDEN = 'userDetailVo.userStatsRet.completedOrderNumOfLatest30day'

# Columnas a cargar del CSV (optimización de memoria)
COLUMNAS_CARGAR = [
    'time', 'userNo', 'adv.advNo', 'adv.tradeType', 'adv.asset', 'adv.fiatUnit',
    'adv.price', 'adv.surplusAmount', 'adv.tradableQuantity',
    'adv.maxSingleTransAmount', 'adv.minSingleTransAmount',
    'adv.dynamicMaxSingleTransQuantity', 'adv.commissionRate', 'adv.classify',
    'userDetailVo.userStatsRet.registerDays',
    'userDetailVo.userStatsRet.firstOrderDays',
    'userDetailVo.userStatsRet.avgReleaseTimeOfLatest30day',
    'userDetailVo.userStatsRet.avgPayTimeOfLatest30day',
    'userDetailVo.userStatsRet.finishRateLatest30day',
    CAMPO_VALIDACION_ORDEN,
    'userDetailVo.userStatsRet.completedOrderNum',
    'userDetailVo.userStatsRet.completedBuyOrderNum',
    'userDetailVo.userStatsRet.completedSellOrderNum',
    'userDetailVo.userStatsRet.counterpartyCount'
]

# Tipos de datos optimizados para memoria
DTYPE_MAP = {
    'adv.price': 'float32',
    'adv.surplusAmount': 'float32',
    'adv.tradableQuantity': 'float32',
    'adv.maxSingleTransAmount': 'float32',
    'adv.minSingleTransAmount': 'float32',
    'adv.dynamicMaxSingleTransQuantity': 'float32',
    'adv.commissionRate': 'float32',
    'adv.tradeType': 'category',
    'adv.asset': 'category',
    'adv.fiatUnit': 'category',
    'adv.classify': 'category',
    'userDetailVo.userStatsRet.registerDays': 'float32',
    'userDetailVo.userStatsRet.firstOrderDays': 'float32',
    'userDetailVo.userStatsRet.avgReleaseTimeOfLatest30day': 'float32',
    'userDetailVo.userStatsRet.avgPayTimeOfLatest30day': 'float32',
    'userDetailVo.userStatsRet.finishRateLatest30day': 'float32',
    CAMPO_VALIDACION_ORDEN: 'float32',
    'userDetailVo.userStatsRet.completedOrderNum': 'float32',
    'userDetailVo.userStatsRet.completedBuyOrderNum': 'float32',
    'userDetailVo.userStatsRet.completedSellOrderNum': 'float32',
    'userDetailVo.userStatsRet.counterpartyCount': 'float32'
}

# Parámetros de filtrado por defecto
MONTO_MINIMO_DEFAULT = 20
AJUSTE_TIMEZONE_HORAS = -4  # Bolivia GMT-4
VENTANA_MEDIANA_MOVIL = '60min'
MIN_PERIODOS_MEDIANA = 5
DESVIACION_PERMITIDA = 0.02  # 2%


# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

def cargar_csv_optimizado(ruta_csv, chunk_size=None):
    """
    Carga el CSV con optimización de memoria.
    
    Args:
        ruta_csv: Ruta al archivo CSV
        chunk_size: Si se especifica, retorna un iterador de chunks
        
    Returns:
        DataFrame o iterador de chunks
    """
    kwargs = {
        'sep': '|',
        'usecols': COLUMNAS_CARGAR,
        'dtype': DTYPE_MAP
    }
    
    if chunk_size:
        kwargs['chunksize'] = chunk_size
        return pd.read_csv(ruta_csv, **kwargs)
    else:
        return pd.read_csv(ruta_csv, **kwargs)


def preparar_dataframe(df):
    """
    Prepara el DataFrame después de cargarlo.
    
    Args:
        df: DataFrame cargado
        
    Returns:
        DataFrame preparado
    """
    df.columns = df.columns.str.strip()
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'] + pd.Timedelta(hours=AJUSTE_TIMEZONE_HORAS)
    return df


# ============================================================================
# FUNCIONES DE DETECCIÓN DE TRANSACCIONES
# ============================================================================

def detectar_transacciones(df, incluir_todos_pares=False):
    """
    Detecta transacciones usando validación cruzada:
    - Disminución en dynamicMaxSingleTransQuantity (volumen del anuncio)
    - Aumento en completedOrderNumOfLatest30day (órdenes del usuario)
    
    Args:
        df: DataFrame con datos del mercado P2P
        incluir_todos_pares: Si True, incluye todos los pares. Si False, solo USDT/BOB
        
    Returns:
        DataFrame con transacciones detectadas
    """
    # Filtrar por par si es necesario
    if not incluir_todos_pares:
        df = df[
            (df['adv.asset'] == 'USDT') & 
            (df['adv.fiatUnit'] == 'BOB')
        ].copy()
    else:
        df = df.copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Ordenar para calcular diferencias correctamente
    df = df.sort_values(
        by=['userNo', 'adv.advNo', 'adv.tradeType', 'adv.asset', 'adv.fiatUnit', 'time']
    )
    
    # 1. VALIDACIÓN DE VOLUMEN
    # Diferencia en cantidad disponible por anuncio específico
    df['diferencia'] = df.groupby([
        'userNo', 'adv.advNo', 'adv.tradeType', 'adv.asset', 'adv.fiatUnit'
    ])['adv.dynamicMaxSingleTransQuantity'].diff()
    
    # 2. VALIDACIÓN DE ORDEN
    # Delta de órdenes completadas por usuario
    df['delta_completedOrders'] = df.groupby('userNo')[
        CAMPO_VALIDACION_ORDEN
    ].diff()
    
    # 3. CALCULAR MONTO DE TRANSACCIÓN
    # Si la diferencia es negativa, hubo una transacción
    df['montoDynamic'] = df['diferencia'].apply(lambda x: -x if pd.notna(x) and x < 0 else 0)
    
    # 4. FILTRAR TRANSACCIONES VÁLIDAS
    # Ambas condiciones deben cumplirse
    df_trans = df[
        (df['montoDynamic'] > 0) &
        (df['delta_completedOrders'] > 0)
    ].copy()
    
    if len(df_trans) == 0:
        return pd.DataFrame()
    
    # 5. ASIGNAR PRECIO DE TRANSACCIÓN
    df_trans['precio_transaccion'] = df_trans['adv.price']
    
    return df_trans


def corregir_duplicacion_delta(df_trans):
    """
    Corrige la duplicación de delta_completedOrders cuando un usuario
    tiene múltiples anuncios activos en el mismo momento.
    
    El delta se distribuye proporcionalmente según el volumen de cada detección.
    
    Args:
        df_trans: DataFrame con transacciones detectadas
        
    Returns:
        DataFrame con delta corregido
    """
    if len(df_trans) == 0:
        return df_trans
    
    df = df_trans.copy()
    
    # Crear identificador único usuario-tiempo
    df['user_time_key'] = df['userNo'].astype(str) + '_' + df['time'].astype(str)
    
    # Calcular volumen total por usuario-tiempo
    df['vol_usuario_tiempo'] = df.groupby('user_time_key')['montoDynamic'].transform('sum')
    
    # Calcular peso proporcional de cada detección
    df['peso_volumen'] = df['montoDynamic'] / df['vol_usuario_tiempo'].replace(0, np.nan)
    df['peso_volumen'] = df['peso_volumen'].fillna(1)
    
    # Delta corregido = delta_original * peso
    df['delta_ordenes_corregido'] = df['delta_completedOrders'] * df['peso_volumen']
    
    return df


def aplicar_filtro_monto_minimo(df_trans, monto_minimo=MONTO_MINIMO_DEFAULT):
    """
    Filtra transacciones por monto mínimo.
    
    Args:
        df_trans: DataFrame con transacciones
        monto_minimo: Monto mínimo en USDT
        
    Returns:
        DataFrame filtrado
    """
    if len(df_trans) == 0:
        return df_trans
    
    return df_trans[df_trans['montoDynamic'] >= monto_minimo].copy()


def aplicar_filtro_mediana_movil(df_trans, ventana=VENTANA_MEDIANA_MOVIL, 
                                  min_periodos=MIN_PERIODOS_MEDIANA,
                                  desviacion=DESVIACION_PERMITIDA):
    """
    Elimina transacciones con precios atípicos usando mediana móvil.
    
    Args:
        df_trans: DataFrame con transacciones
        ventana: Tamaño de ventana para mediana móvil
        min_periodos: Mínimo de observaciones para calcular mediana
        desviacion: Desviación permitida respecto a la mediana (ej: 0.02 = 2%)
        
    Returns:
        DataFrame filtrado
    """
    if len(df_trans) == 0:
        return df_trans
    
    df = df_trans.copy()
    
    # Ordenar por tiempo
    df = df.sort_values('time')
    
    # Calcular mediana móvil
    df.set_index('time', inplace=True)
    df['rolling_median'] = df['precio_transaccion'].rolling(
        window=ventana, min_periods=min_periodos
    ).median()
    
    # Rellenar valores faltantes al inicio
    df['rolling_median'] = df['rolling_median'].bfill()
    
    # Calcular límites
    limite_superior = df['rolling_median'] * (1 + desviacion)
    limite_inferior = df['rolling_median'] * (1 - desviacion)
    
    # Filtrar
    df = df[
        (df['precio_transaccion'] >= limite_inferior) &
        (df['precio_transaccion'] <= limite_superior)
    ].copy()
    
    # Limpiar y restaurar índice
    df = df.drop(columns=['rolling_median'])
    df.reset_index(inplace=True)
    
    return df


def pipeline_deteccion_completo(df, monto_minimo=MONTO_MINIMO_DEFAULT,
                                 aplicar_filtro_atipicos=True,
                                 incluir_todos_pares=False,
                                 verbose=True):
    """
    Pipeline completo de detección de transacciones.
    
    Ejecuta todos los pasos:
    1. Detección de transacciones
    2. Corrección de duplicación de delta
    3. Filtro de monto mínimo
    4. Filtro de mediana móvil (opcional)
    
    Args:
        df: DataFrame con datos crudos
        monto_minimo: Monto mínimo en USDT
        aplicar_filtro_atipicos: Si aplicar filtro de mediana móvil
        incluir_todos_pares: Si incluir todos los pares o solo USDT/BOB
        verbose: Si imprimir mensajes de progreso
        
    Returns:
        DataFrame con transacciones procesadas
    """
    if verbose:
        print("Iniciando pipeline de detección de transacciones...")
    
    # Paso 1: Detección
    if verbose:
        print("  [1/4] Detectando transacciones...")
    df_trans = detectar_transacciones(df, incluir_todos_pares)
    
    if len(df_trans) == 0:
        if verbose:
            print("  No se detectaron transacciones.")
        return pd.DataFrame()
    
    if verbose:
        print(f"        Detectadas: {len(df_trans):,}")
    
    # Paso 2: Corrección de duplicación
    if verbose:
        print("  [2/4] Corrigiendo duplicación de delta_ordenes...")
    df_trans = corregir_duplicacion_delta(df_trans)
    
    # Paso 3: Filtro de monto mínimo
    if verbose:
        print(f"  [3/4] Aplicando filtro de monto mínimo (>= {monto_minimo} USDT)...")
    df_trans = aplicar_filtro_monto_minimo(df_trans, monto_minimo)
    
    if len(df_trans) == 0:
        if verbose:
            print("        No quedan transacciones después del filtro.")
        return pd.DataFrame()
    
    if verbose:
        print(f"        Restantes: {len(df_trans):,}")
    
    # Paso 4: Filtro de atípicos
    if aplicar_filtro_atipicos:
        if verbose:
            print("  [4/4] Aplicando filtro de mediana móvil...")
        df_trans = aplicar_filtro_mediana_movil(df_trans)
        
        if verbose:
            print(f"        Restantes: {len(df_trans):,}")
    else:
        if verbose:
            print("  [4/4] Filtro de atípicos omitido.")
    
    if verbose:
        print("Pipeline completado.")
    
    return df_trans


# ============================================================================
# FUNCIONES DE PROCESAMIENTO POR CHUNKS (para archivos grandes)
# ============================================================================

def procesar_por_chunks(ruta_csv, chunk_size=500_000, callback=None, verbose=True):
    """
    Procesa un archivo CSV grande por chunks.
    
    Args:
        ruta_csv: Ruta al archivo CSV
        chunk_size: Número de filas por chunk
        callback: Función a aplicar a cada chunk de transacciones detectadas
        verbose: Si imprimir mensajes de progreso
        
    Returns:
        Lista de resultados del callback, o lista de DataFrames si no hay callback
    """
    resultados = []
    chunk_num = 0
    total_registros = 0
    total_transacciones = 0
    
    if verbose:
        print(f"Procesando archivo en chunks de {chunk_size:,} filas...")
    
    for chunk in cargar_csv_optimizado(ruta_csv, chunk_size=chunk_size):
        chunk_num += 1
        total_registros += len(chunk)
        
        if verbose and chunk_num % 10 == 0:
            print(f"  Chunk {chunk_num}: {total_registros:,} registros procesados...")
        
        # Preparar chunk
        chunk = preparar_dataframe(chunk)
        
        # Detectar transacciones en este chunk
        trans_chunk = pipeline_deteccion_completo(
            chunk, 
            aplicar_filtro_atipicos=False,  # Se aplicará al final
            verbose=False
        )
        
        if len(trans_chunk) > 0:
            total_transacciones += len(trans_chunk)
            
            if callback:
                resultado = callback(trans_chunk)
                resultados.append(resultado)
            else:
                resultados.append(trans_chunk)
        
        # Liberar memoria
        del chunk
        gc.collect()
    
    if verbose:
        print(f"\nProcesamiento completado:")
        print(f"  Total registros: {total_registros:,}")
        print(f"  Total transacciones detectadas: {total_transacciones:,}")
    
    return resultados


def consolidar_transacciones_chunks(lista_chunks, aplicar_filtro_atipicos=True, verbose=True):
    """
    Consolida transacciones de múltiples chunks y aplica filtros finales.
    
    Args:
        lista_chunks: Lista de DataFrames con transacciones por chunk
        aplicar_filtro_atipicos: Si aplicar filtro de mediana móvil
        verbose: Si imprimir mensajes
        
    Returns:
        DataFrame consolidado y filtrado
    """
    if not lista_chunks:
        return pd.DataFrame()
    
    if verbose:
        print("Consolidando transacciones de todos los chunks...")
    
    df_trans = pd.concat(lista_chunks, ignore_index=True)
    
    if verbose:
        print(f"  Transacciones totales: {len(df_trans):,}")
    
    if aplicar_filtro_atipicos and len(df_trans) > 0:
        if verbose:
            print("  Aplicando filtro de mediana móvil...")
        df_trans = aplicar_filtro_mediana_movil(df_trans)
        if verbose:
            print(f"  Transacciones después de filtro: {len(df_trans):,}")
    
    return df_trans


# ============================================================================
# FUNCIONES DE DIAGNÓSTICO
# ============================================================================

def diagnostico_transacciones(df_trans, verbose=True):
    """
    Genera diagnóstico de las transacciones detectadas.
    
    Args:
        df_trans: DataFrame con transacciones
        verbose: Si imprimir resultados
        
    Returns:
        Diccionario con métricas de diagnóstico
    """
    if len(df_trans) == 0:
        return {'error': 'No hay transacciones'}
    
    diagnostico = {}
    
    # Métricas básicas
    diagnostico['total_detecciones'] = len(df_trans)
    diagnostico['volumen_total'] = df_trans['montoDynamic'].sum()
    
    # Métricas de delta_ordenes
    if 'delta_completedOrders' in df_trans.columns:
        diagnostico['suma_delta_ordenes_raw'] = df_trans['delta_completedOrders'].sum()
    
    if 'delta_ordenes_corregido' in df_trans.columns:
        diagnostico['suma_delta_ordenes_corregido'] = df_trans['delta_ordenes_corregido'].sum()
        
        if diagnostico.get('suma_delta_ordenes_raw', 0) > 0:
            diagnostico['pct_correccion'] = (
                1 - diagnostico['suma_delta_ordenes_corregido'] / diagnostico['suma_delta_ordenes_raw']
            ) * 100
    
    # Distribución de delta
    if 'delta_completedOrders' in df_trans.columns:
        diagnostico['delta_igual_1'] = (df_trans['delta_completedOrders'] == 1).sum()
        diagnostico['delta_igual_2'] = (df_trans['delta_completedOrders'] == 2).sum()
        diagnostico['delta_mayor_3'] = (df_trans['delta_completedOrders'] >= 3).sum()
    
    # Por tipo de transacción
    if 'adv.tradeType' in df_trans.columns:
        for tipo in df_trans['adv.tradeType'].unique():
            df_tipo = df_trans[df_trans['adv.tradeType'] == tipo]
            diagnostico[f'detecciones_{tipo}'] = len(df_tipo)
            diagnostico[f'volumen_{tipo}'] = df_tipo['montoDynamic'].sum()
    
    # Período
    diagnostico['fecha_min'] = df_trans['time'].min()
    diagnostico['fecha_max'] = df_trans['time'].max()
    diagnostico['dias_cobertura'] = (diagnostico['fecha_max'] - diagnostico['fecha_min']).days
    
    # Traders únicos
    diagnostico['traders_unicos'] = df_trans['userNo'].nunique()
    
    if verbose:
        print("\n" + "="*60)
        print("DIAGNÓSTICO DE TRANSACCIONES")
        print("="*60)
        print(f"Total detecciones:        {diagnostico['total_detecciones']:,}")
        print(f"Volumen total:            {diagnostico['volumen_total']:,.0f} USDT")
        print(f"Traders únicos:           {diagnostico['traders_unicos']:,}")
        print(f"Período:                  {diagnostico['fecha_min']} a {diagnostico['fecha_max']}")
        print(f"Días de cobertura:        {diagnostico['dias_cobertura']}")
        
        if 'suma_delta_ordenes_raw' in diagnostico:
            print(f"\nDelta órdenes (raw):      {diagnostico['suma_delta_ordenes_raw']:,.0f}")
        if 'suma_delta_ordenes_corregido' in diagnostico:
            print(f"Delta órdenes (corregido):{diagnostico['suma_delta_ordenes_corregido']:,.0f}")
        if 'pct_correccion' in diagnostico:
            print(f"Corrección aplicada:      {diagnostico['pct_correccion']:.1f}%")
        
        if 'delta_igual_1' in diagnostico:
            total = diagnostico['total_detecciones']
            print(f"\nDistribución de delta:")
            print(f"  delta = 1:  {diagnostico['delta_igual_1']:,} ({diagnostico['delta_igual_1']/total*100:.1f}%)")
            print(f"  delta = 2:  {diagnostico['delta_igual_2']:,} ({diagnostico['delta_igual_2']/total*100:.1f}%)")
            print(f"  delta >= 3: {diagnostico['delta_mayor_3']:,} ({diagnostico['delta_mayor_3']/total*100:.1f}%)")
        
        if 'volumen_compra' in diagnostico:
            print(f"\nPor tipo de transacción:")
            print(f"  Compra: {diagnostico['detecciones_compra']:,} det., {diagnostico['volumen_compra']:,.0f} USDT")
            print(f"  Venta:  {diagnostico['detecciones_venta']:,} det., {diagnostico['volumen_venta']:,.0f} USDT")
        
        print("="*60)
    
    return diagnostico


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def filtrar_por_par(df, asset='USDT', fiat='BOB'):
    """Filtra DataFrame por par de trading."""
    return df[
        (df['adv.asset'] == asset) & 
        (df['adv.fiatUnit'] == fiat)
    ].copy()


def agregar_variables_calendario(df):
    """Agrega variables de calendario al DataFrame."""
    df = df.copy()
    df['fecha'] = df['time'].dt.date
    df['dia_semana'] = df['time'].dt.dayofweek
    df['dia_semana_nombre'] = df['time'].dt.day_name()
    df['dia_mes'] = df['time'].dt.day
    df['mes'] = df['time'].dt.month
    df['hora'] = df['time'].dt.hour
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['es_fin_mes'] = (df['dia_mes'] >= 28).astype(int)
    df['es_quincena'] = df['dia_mes'].isin([14, 15, 16, 29, 30, 31]).astype(int)
    return df


# ============================================================================
# INFORMACIÓN DEL MÓDULO
# ============================================================================

def info():
    """Imprime información sobre el módulo."""
    print("""
================================================================================
CORE_TRANSACCIONES - Módulo Base para Análisis de Mercado P2P Binance
================================================================================

Funciones principales:
    - detectar_transacciones(df): Detecta transacciones con validación cruzada
    - corregir_duplicacion_delta(df): Corrige duplicación de delta_ordenes
    - aplicar_filtro_monto_minimo(df): Filtra por monto mínimo
    - aplicar_filtro_mediana_movil(df): Elimina precios atípicos
    - pipeline_deteccion_completo(df): Pipeline completo de detección

Para archivos grandes (>1GB):
    - procesar_por_chunks(ruta_csv): Procesa por chunks
    - consolidar_transacciones_chunks(lista): Consolida resultados

Diagnóstico:
    - diagnostico_transacciones(df): Genera métricas de diagnóstico

Uso básico:
    from core_transacciones import cargar_csv_optimizado, preparar_dataframe
    from core_transacciones import pipeline_deteccion_completo
    
    df = cargar_csv_optimizado('datos.csv')
    df = preparar_dataframe(df)
    transacciones = pipeline_deteccion_completo(df)

Uso con archivos grandes:
    from core_transacciones import procesar_por_chunks, consolidar_transacciones_chunks
    
    chunks = procesar_por_chunks('datos_grandes.csv', chunk_size=500_000)
    transacciones = consolidar_transacciones_chunks(chunks)
================================================================================
    """)


if __name__ == "__main__":
    info()

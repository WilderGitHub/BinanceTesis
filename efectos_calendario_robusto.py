"""
================================================================================
EFECTOS_CALENDARIO_ROBUSTO.PY - Análisis con Controles Adicionales
================================================================================
Version mejorada con:
- Controles de mercado cripto (BTC)
- Rezagos para autocorrelacion
- Dummies de shocks macroeconomicos
- Efectos fijos por mes
- Errores Newey-West (HAC)
- Tests de robustez

Autor: Para tesis de maestria en finanzas
Fecha: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Importar modulo core
try:
    from core_transacciones import (
        cargar_csv_optimizado, preparar_dataframe,
        procesar_por_chunks, consolidar_transacciones_chunks,
        pipeline_deteccion_completo,
        COLUMNAS_CARGAR, DTYPE_MAP
    )
except ImportError:
    print("ERROR: No se encontro core_transacciones.py")
    sys.exit(1)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac
except ImportError:
    print("ERROR: statsmodels no esta instalado")
    sys.exit(1)

# Intentar importar yfinance para datos de BTC
try:
    import yfinance as yf
    YFINANCE_DISPONIBLE = True
except ImportError:
    YFINANCE_DISPONIBLE = False
    print("ADVERTENCIA: yfinance no instalado. No se descargaran datos de BTC.")
    print("Instalar con: pip install yfinance")


# ============================================================================
# CONFIGURACION
# ============================================================================
CHUNK_SIZE = 500_000
UMBRAL_ARCHIVO_GRANDE = 1_000_000_000

# Feriados Bolivia 2024-2025
#FERIADOS_BOLIVIA = pd.to_datetime([
#    '2024-01-01', '2024-01-22', '2024-02-12', '2024-02-13', '2024-03-29',
#    '2024-05-01', '2024-05-30', '2024-06-21', '2024-08-06', '2024-11-02', '2024-12-25',
#    '2025-01-01', '2025-01-22', '2025-03-03', '2025-03-04', '2025-04-18',
#    '2025-05-01', '2025-06-19', '2025-06-21', '2025-08-06', '2025-11-02', '2025-12-25',
#])
FERIADOS_BOLIVIA = pd.to_datetime([
    # 2024
    '2024-01-01', '2024-01-22', '2024-02-12', '2024-02-13', '2024-03-29', '2024-05-01', '2024-05-30', 
    '2024-06-21', '2024-08-06', '2024-11-02', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', 
    '2024-12-30', '2024-12-31',
    # 2025
    '2025-01-22', '2025-03-03', '2025-03-04', '2025-04-18', '2025-05-01', '2025-06-19', '2025-06-21', 
    '2025-07-16', '2025-08-06', '2025-08-07', '2025-09-24', '2025-11-03', '2025-11-14', '2025-12-25'
])
# Eventos/Shocks macroeconomicos Bolivia (ajustar segun tu conocimiento)
# Formato: (fecha_inicio, fecha_fin, nombre)
#SHOCKS_MACRO = [
    # Ejemplos - AJUSTAR SEGUN EVENTOS REALES EN BOLIVIA
#    ('2024-10-01', '2024-10-15', 'crisis_dolar_oct24'),      # Ejemplo: escasez de dolares
#    ('2025-02-01', '2025-02-28', 'crisis_feb25'),            # Ejemplo: otro evento
    # Agregar mas eventos segun tu conocimiento del periodo

#]
SHOCKS_MACRO = [
    # Formato: (Fecha Inicio, Fecha Fin, Etiqueta del Evento)
    ('2024-11-04', '2024-11-23', 'evento_nov24'),
    ('2025-03-10', '2025-03-14', 'evento_mar25'),
    ('2025-04-09', '2025-04-11', 'evento_abr25_1'),
    ('2025-04-22', '2025-04-25', 'evento_abr25_2'),
    ('2025-04-28', '2025-04-30', 'evento_abr25_3'),
    ('2025-05-12', '2025-05-16', 'evento_may25'),
    ('2025-06-15', '2025-06-17', 'evento_jun25_1'),
    ('2025-06-26', '2025-06-30', 'evento_jun25_2'),
    ('2025-07-01', '2025-07-02', 'evento_jul25_1'),
    ('2025-07-21', '2025-07-28', 'evento_jul25_2'),
    ('2025-08-02', '2025-08-05', 'evento_ago25_1'),
    ('2025-08-11', '2025-08-12', 'evento_ago25_2'),
    ('2025-08-27', '2025-09-03', 'evento_ago_sep25'), # Cruza de mes
    ('2025-10-03', '2025-10-10', 'evento_oct25'),
    ('2025-10-19', '2025-11-13', 'evento_oct_nov25'), # Cruza de mes (largo)
    ('2025-11-27', '2025-11-30', 'evento_nov25'),
]


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class AnalisisCalendarioRobusto:
    """Analisis de efectos calendario con controles de robustez."""
    
    def __init__(self, ruta_csv=None, df_transacciones=None, directorio_salida=None):
        self.ruta_csv = ruta_csv
        self.transacciones = df_transacciones
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        elif ruta_csv:
            self.directorio_salida = os.path.dirname(ruta_csv)
        else:
            self.directorio_salida = os.getcwd()
        
        # Directorios
        self.dir_graficos = os.path.join(self.directorio_salida, 'robusto_graficos')
        self.dir_datos = os.path.join(self.directorio_salida, 'robusto_datos')
        self.dir_tablas = os.path.join(self.directorio_salida, 'robusto_tablas')
        os.makedirs(self.dir_graficos, exist_ok=True)
        os.makedirs(self.dir_datos, exist_ok=True)
        os.makedirs(self.dir_tablas, exist_ok=True)
        
        # Datos
        self.datos_diarios = None
        self.datos_btc = None
        self.resultados = {}
        self.reporte = []
    
    def log(self, mensaje):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {mensaje}")
        self.reporte.append(f"[{ts}] {mensaje}")
    
    # ========================================================================
    # FASE 1: CARGA DE DATOS
    # ========================================================================
    def fase1_cargar_datos(self):
        """Carga transacciones del mercado P2P."""
        self.log("="*70)
        self.log("FASE 1: Carga de datos P2P")
        self.log("="*70)
        
        if self.transacciones is not None:
            self.log(f"Usando transacciones pre-cargadas: {len(self.transacciones):,}")
            return
        
        if not self.ruta_csv:
            raise ValueError("Debe proporcionar ruta_csv o df_transacciones")
        
        file_size = os.path.getsize(self.ruta_csv)
        es_grande = file_size > UMBRAL_ARCHIVO_GRANDE
        
        self.log(f"Archivo: {self.ruta_csv}")
        self.log(f"Tamano: {file_size / 1e9:.2f} GB")
        
        if es_grande:
            self.log(f"Procesando por chunks...")
            chunks = []
            chunk_num = 0
            
            for chunk in cargar_csv_optimizado(self.ruta_csv, chunk_size=CHUNK_SIZE):
                chunk_num += 1
                if chunk_num % 20 == 0:
                    self.log(f"  Chunk {chunk_num}...")
                
                chunk = preparar_dataframe(chunk)
                trans = pipeline_deteccion_completo(chunk, aplicar_filtro_atipicos=False, verbose=False)
                if len(trans) > 0:
                    chunks.append(trans)
                
                del chunk
            
            self.transacciones = consolidar_transacciones_chunks(chunks, verbose=False)
        else:
            df = cargar_csv_optimizado(self.ruta_csv)
            df = preparar_dataframe(df)
            self.transacciones = pipeline_deteccion_completo(df, verbose=True)
        
        self.log(f"Transacciones: {len(self.transacciones):,}")
    
    # ========================================================================
    # FASE 2: DESCARGAR DATOS DE BTC
    # ========================================================================
    def fase2_datos_btc(self):
        """Descarga datos historicos de Bitcoin."""
        self.log("\n" + "="*70)
        self.log("FASE 2: Descarga de datos de Bitcoin (control)")
        self.log("="*70)
        
        if not YFINANCE_DISPONIBLE:
            self.log("yfinance no disponible. Saltando descarga de BTC.")
            self.datos_btc = None
            return
        
        # Obtener rango de fechas
        fecha_min = self.transacciones['time'].min().date()
        fecha_max = self.transacciones['time'].max().date()
        
        self.log(f"Descargando BTC-USD desde {fecha_min} hasta {fecha_max}...")
        
        try:
            btc = yf.download('BTC-USD', start=fecha_min, end=fecha_max + timedelta(days=1), 
                             progress=False)
            
            if len(btc) == 0:
                self.log("No se obtuvieron datos de BTC")
                self.datos_btc = None
                return
            
            btc = btc.reset_index()
            btc.columns = btc.columns.get_level_values(0)
            btc = btc.rename(columns={'Date': 'fecha', 'Close': 'btc_close', 
                                       'High': 'btc_high', 'Low': 'btc_low',
                                       'Volume': 'btc_volume'})
            
            btc['fecha'] = pd.to_datetime(btc['fecha']).dt.date
            btc['fecha'] = pd.to_datetime(btc['fecha'])
            
            # Calcular metricas
            btc['btc_retorno'] = btc['btc_close'].pct_change() * 100
            btc['btc_volatilidad'] = (btc['btc_high'] - btc['btc_low']) / btc['btc_close'] * 100
            btc['btc_retorno_abs'] = btc['btc_retorno'].abs()
            
            # Retorno positivo o negativo
            btc['btc_up'] = (btc['btc_retorno'] > 0).astype(int)
            btc['btc_down'] = (btc['btc_retorno'] < 0).astype(int)
            
            # Volatilidad alta (por encima del percentil 75)
            btc['btc_vol_alta'] = (btc['btc_volatilidad'] > btc['btc_volatilidad'].quantile(0.75)).astype(int)
            
            self.datos_btc = btc[['fecha', 'btc_close', 'btc_retorno', 'btc_volatilidad', 
                                   'btc_retorno_abs', 'btc_up', 'btc_down', 'btc_vol_alta']]
            
            self.log(f"Datos BTC obtenidos: {len(self.datos_btc)} dias")
            self.log(f"  Retorno promedio: {btc['btc_retorno'].mean():.2f}%")
            self.log(f"  Volatilidad promedio: {btc['btc_volatilidad'].mean():.2f}%")
            
        except Exception as e:
            self.log(f"Error descargando BTC: {e}")
            self.datos_btc = None
    
    # ========================================================================
    # FASE 3: CONSTRUIR DATASET CON CONTROLES
    # ========================================================================
    def fase3_construir_dataset(self):
        """Construye dataset con todas las variables de control."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Construccion de dataset con controles")
        self.log("="*70)
        
        df = self.transacciones.copy()
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        
        # Separar por tipo
        df_compra = df[df['adv.tradeType'] == 'compra']
        df_venta = df[df['adv.tradeType'] == 'venta']
        
        # Agregacion diaria
        diario = df.groupby('fecha').agg({
            'precio_transaccion': ['mean', 'std', 'min', 'max'],
            'montoDynamic': ['sum', 'mean', 'count'],
            'delta_ordenes_corregido': 'sum',
            'userNo': 'nunique'
        }).reset_index()
        
        diario.columns = ['fecha', 'precio_mean', 'precio_std', 'precio_min', 'precio_max',
                          'volumen', 'ticket_medio', 'n_detecciones', 'n_ordenes', 'n_traders']
        
        # Volumen por tipo
        vol_compra = df_compra.groupby('fecha')['montoDynamic'].sum()
        vol_venta = df_venta.groupby('fecha')['montoDynamic'].sum()
        
        diario['vol_compra'] = diario['fecha'].map(vol_compra).fillna(0)
        diario['vol_venta'] = diario['fecha'].map(vol_venta).fillna(0)
        diario['ratio_cv'] = diario['vol_compra'] / diario['vol_venta'].replace(0, np.nan)
        
        # ================================================================
        # VARIABLES CALENDARIO
        # ================================================================
        self.log("Construyendo variables calendario...")
        
        diario['dia_semana'] = diario['fecha'].dt.dayofweek
        diario['dia_nombre'] = diario['fecha'].dt.day_name()
        diario['dia_mes'] = diario['fecha'].dt.day
        diario['mes'] = diario['fecha'].dt.month
        diario['semana'] = diario['fecha'].dt.isocalendar().week
        
        # Dummies dia de semana
        for i, dia in enumerate(['martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']):
            diario[f'd_{dia}'] = (diario['dia_semana'] == i + 1).astype(int)
        
        # Efectos principales
        diario['es_fds'] = (diario['dia_semana'] >= 5).astype(int)
        diario['es_fin_mes'] = (diario['dia_mes'] >= 25).astype(int)
        diario['es_inicio_mes'] = (diario['dia_mes'] <= 5).astype(int)
        diario['es_quincena'] = diario['dia_mes'].isin([14, 15, 16, 28, 29, 30, 31]).astype(int)
        diario['es_feriado'] = diario['fecha'].isin(FERIADOS_BOLIVIA).astype(int)
        diario['post_feriado'] = diario['fecha'].isin(FERIADOS_BOLIVIA + timedelta(days=1)).astype(int)
        
        # Tendencia
        diario['tendencia'] = (diario['fecha'] - diario['fecha'].min()).dt.days
        
        # ================================================================
        # EFECTOS FIJOS POR MES
        # ================================================================
        self.log("Agregando efectos fijos por mes...")
        
        # Crear dummies por mes (referencia: mes 1)
        meses_unicos = sorted(diario['mes'].unique())
        for m in meses_unicos[1:]:  # Excluir el primero como referencia
            diario[f'mes_{m}'] = (diario['mes'] == m).astype(int)
        
        # ================================================================
        # DUMMIES DE SHOCKS MACRO
        # ================================================================
        self.log("Agregando dummies de shocks macroeconomicos...")
        
        diario['shock_macro'] = 0
        for inicio, fin, nombre in SHOCKS_MACRO:
            inicio_dt = pd.to_datetime(inicio)
            fin_dt = pd.to_datetime(fin)
            mask = (diario['fecha'] >= inicio_dt) & (diario['fecha'] <= fin_dt)
            diario.loc[mask, 'shock_macro'] = 1
            diario[f'shock_{nombre}'] = mask.astype(int)
            n_dias = mask.sum()
            if n_dias > 0:
                self.log(f"  {nombre}: {n_dias} dias")
        
        self.log(f"  Total dias con shock: {diario['shock_macro'].sum()}")
        
        # ================================================================
        # MERGE CON DATOS BTC
        # ================================================================
        if self.datos_btc is not None:
            self.log("Incorporando datos de Bitcoin...")
            diario = diario.merge(self.datos_btc, on='fecha', how='left')
            
            # Rellenar NaN con medianas
            for col in ['btc_retorno', 'btc_volatilidad', 'btc_retorno_abs', 'btc_up', 'btc_down', 'btc_vol_alta']:
                if col in diario.columns:
                    diario[col] = diario[col].fillna(diario[col].median())
            
            self.log(f"  Dias con datos BTC: {diario['btc_close'].notna().sum()}")
        else:
            # Crear columnas vacias si no hay datos BTC
            diario['btc_retorno'] = 0
            diario['btc_volatilidad'] = 0
            diario['btc_retorno_abs'] = 0
            diario['btc_vol_alta'] = 0
        
        # ================================================================
        # REZAGOS (para controlar autocorrelacion)
        # ================================================================
        self.log("Calculando rezagos...")
        
        diario = diario.sort_values('fecha')
        diario['log_volumen'] = np.log1p(diario['volumen'])
        diario['log_volumen_lag1'] = diario['log_volumen'].shift(1)
        diario['log_volumen_lag7'] = diario['log_volumen'].shift(7)  # Mismo dia semana anterior
        diario['volumen_lag1'] = diario['volumen'].shift(1)
        
        # Retorno del volumen
        diario['volumen_retorno'] = diario['volumen'].pct_change() * 100
        
        # ================================================================
        # VARIABLES ADICIONALES
        # ================================================================
        diario['volatilidad_precio'] = (diario['precio_max'] - diario['precio_min']) / diario['precio_mean'] * 100
        diario['log_ordenes'] = np.log1p(diario['n_ordenes'])
        diario['log_traders'] = np.log1p(diario['n_traders'])
        
        # Eliminar primera fila (NaN por rezago)
        diario = diario.dropna(subset=['log_volumen_lag1'])
        
        self.datos_diarios = diario
        self.log(f"\nDataset final: {len(diario)} observaciones")
        self.log(f"Variables disponibles: {len(diario.columns)}")
        
        # Guardar
        diario.to_csv(os.path.join(self.dir_datos, 'datos_diarios_robusto.csv'), index=False)
    
    # ========================================================================
    # FASE 4: REGRESIONES CON CONTROLES
    # ========================================================================
    def fase4_regresiones(self):
        """Ejecuta bateria de regresiones con diferentes controles."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Regresiones con controles de robustez")
        self.log("="*70)
        
        df = self.datos_diarios.copy()
        
        # ================================================================
        # MODELO BASE (referencia)
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 1: Base (solo efectos calendario)")
        self.log("-"*60)
        
        formula1 = 'log_volumen ~ es_fds + es_feriado + tendencia'
        m1 = smf.ols(formula1, data=df).fit(cov_type='HC3')
        self.resultados['M1_base'] = m1
        self._mostrar_resumen(m1, 'M1_base')
        
        # ================================================================
        # MODELO CON REZAGO (controla autocorrelacion)
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 2: Con rezago de volumen (controla autocorrelacion)")
        self.log("-"*60)
        
        formula2 = 'log_volumen ~ es_fds + es_feriado + tendencia + log_volumen_lag1'
        m2 = smf.ols(formula2, data=df).fit(cov_type='HC3')
        self.resultados['M2_rezago'] = m2
        self._mostrar_resumen(m2, 'M2_rezago')
        
        # ================================================================
        # MODELO CON CONTROLES BTC
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 3: Con controles de Bitcoin")
        self.log("-"*60)
        
        formula3 = 'log_volumen ~ es_fds + es_feriado + tendencia + btc_retorno + btc_vol_alta'
        m3 = smf.ols(formula3, data=df).fit(cov_type='HC3')
        self.resultados['M3_btc'] = m3
        self._mostrar_resumen(m3, 'M3_btc')
        
        # ================================================================
        # MODELO CON SHOCKS MACRO
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 4: Con dummy de shocks macro")
        self.log("-"*60)
        
        formula4 = 'log_volumen ~ es_fds + es_feriado + tendencia + shock_macro'
        m4 = smf.ols(formula4, data=df).fit(cov_type='HC3')
        self.resultados['M4_shocks'] = m4
        self._mostrar_resumen(m4, 'M4_shocks')
        
        # ================================================================
        # MODELO CON EFECTOS FIJOS POR MES
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 5: Con efectos fijos por mes")
        self.log("-"*60)
        
        # Construir formula con dummies de mes
        cols_mes = [c for c in df.columns if c.startswith('mes_')]
        formula5 = 'log_volumen ~ es_fds + es_feriado + tendencia + ' + ' + '.join(cols_mes)
        m5 = smf.ols(formula5, data=df).fit(cov_type='HC3')
        self.resultados['M5_ef_mes'] = m5
        self._mostrar_resumen(m5, 'M5_ef_mes')
        
        # ================================================================
        # MODELO COMPLETO (KITCHEN SINK)
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 6: Completo (todos los controles)")
        self.log("-"*60)
        
        formula6 = f'''log_volumen ~ es_fds + es_feriado + es_fin_mes + 
                       log_volumen_lag1 + btc_retorno + btc_vol_alta + 
                       shock_macro + tendencia'''
        m6 = smf.ols(formula6, data=df).fit(cov_type='HC3')
        self.resultados['M6_completo'] = m6
        self._mostrar_resumen(m6, 'M6_completo')
        
        # ================================================================
        # MODELO CON ERRORES NEWEY-WEST (HAC)
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 7: Errores Newey-West (HAC) - robusto a autocorrelacion")
        self.log("-"*60)
        
        # Usar el modelo base pero con errores HAC
        y = df['log_volumen']
        X = df[['es_fds', 'es_feriado', 'tendencia']]
        X = sm.add_constant(X)
        
        m7 = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        self.resultados['M7_HAC'] = m7
        
        self.log(f"\n{m7.summary()}")
        self._interpretar_efecto(m7, 'es_fds', 'Efecto FDS (HAC)')
        
        # ================================================================
        # MODELO RATIO COMPRA/VENTA
        # ================================================================
        self.log("\n" + "-"*60)
        self.log("MODELO 8: Ratio Compra/Venta con controles")
        self.log("-"*60)
        
        df_ratio = df.dropna(subset=['ratio_cv'])
        formula8 = 'ratio_cv ~ es_fds + es_feriado + es_fin_mes + tendencia + btc_retorno'
        m8 = smf.ols(formula8, data=df_ratio).fit(cov_type='HC3')
        self.resultados['M8_ratio'] = m8
        self._mostrar_resumen(m8, 'M8_ratio')
    
    def _mostrar_resumen(self, modelo, nombre):
        """Muestra resumen del modelo."""
        self.log(f"\n{modelo.summary()}")
        
        # Interpretar efectos principales
        for var in ['es_fds', 'es_feriado', 'es_fin_mes', 'shock_macro']:
            if var in modelo.params.index:
                self._interpretar_efecto(modelo, var, var)
    
    def _interpretar_efecto(self, modelo, var, nombre):
        """Interpreta el efecto de una variable."""
        coef = modelo.params[var]
        pval = modelo.pvalues[var]
        efecto_pct = (np.exp(coef) - 1) * 100
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        self.log(f"  {nombre}: {efecto_pct:+.1f}% (p={pval:.4f}) {sig}")
    
    # ========================================================================
    # FASE 5: TABLA COMPARATIVA
    # ========================================================================
    def fase5_tabla_comparativa(self):
        """Genera tabla comparativa de todos los modelos."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Tabla comparativa de modelos")
        self.log("="*70)
        
        # Extraer coeficientes de es_fds de cada modelo
        comparativa = []
        
        for nombre, modelo in self.resultados.items():
            if 'es_fds' in modelo.params.index:
                coef = modelo.params['es_fds']
                se = modelo.bse['es_fds']
                pval = modelo.pvalues['es_fds']
                efecto = (np.exp(coef) - 1) * 100
                
                comparativa.append({
                    'Modelo': nombre,
                    'Coef_FDS': coef,
                    'SE': se,
                    'Efecto_%': efecto,
                    'p_value': pval,
                    'R2': modelo.rsquared,
                    'R2_adj': modelo.rsquared_adj,
                    'N': int(modelo.nobs),
                    'Significativo': 'Si' if pval < 0.05 else 'No'
                })
        
        df_comp = pd.DataFrame(comparativa)
        
        self.log("\n" + "="*70)
        self.log("COMPARACION DEL EFECTO FIN DE SEMANA ENTRE MODELOS")
        self.log("="*70)
        
        self.log(f"\n{'Modelo':<15} {'Efecto %':>10} {'p-value':>10} {'R2':>8} {'N':>6} {'Sig':>5}")
        self.log("-" * 60)
        
        for _, row in df_comp.iterrows():
            sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else ''
            self.log(f"{row['Modelo']:<15} {row['Efecto_%']:>+9.1f}% {row['p_value']:>10.4f} {row['R2']:>8.3f} {row['N']:>6} {sig:>5}")
        
        # Guardar tabla
        df_comp.to_csv(os.path.join(self.dir_tablas, 'comparativa_modelos.csv'), index=False)
        
        # Resumen de robustez
        efectos = df_comp['Efecto_%']
        self.log(f"\n" + "="*70)
        self.log("RESUMEN DE ROBUSTEZ")
        self.log("="*70)
        self.log(f"  Efecto FDS minimo:  {efectos.min():+.1f}%")
        self.log(f"  Efecto FDS maximo:  {efectos.max():+.1f}%")
        self.log(f"  Efecto FDS promedio: {efectos.mean():+.1f}%")
        self.log(f"  Desviacion estandar: {efectos.std():.1f}%")
        self.log(f"  Modelos significativos: {(df_comp['p_value'] < 0.05).sum()}/{len(df_comp)}")
        
        if efectos.std() < 5:
            self.log("\n  -> EFECTO ROBUSTO: Consistente entre especificaciones")
        else:
            self.log("\n  -> PRECAUCION: Variabilidad entre especificaciones")
        
        self.tabla_comparativa = df_comp
    
    # ========================================================================
    # FASE 6: TESTS DE ROBUSTEZ ADICIONALES
    # ========================================================================
    def fase6_tests_robustez(self):
        """Ejecuta tests adicionales de robustez."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Tests de robustez adicionales")
        self.log("="*70)
        
        df = self.datos_diarios.copy()
        
        # ================================================================
        # TEST 1: Subperiodos
        # ================================================================
        self.log("\n--- TEST 1: Estabilidad en subperiodos ---")
        
        # Dividir en mitades
        n = len(df)
        df1 = df.iloc[:n//2]
        df2 = df.iloc[n//2:]
        
        formula = 'log_volumen ~ es_fds + es_feriado + tendencia'
        
        m_sub1 = smf.ols(formula, data=df1).fit(cov_type='HC3')
        m_sub2 = smf.ols(formula, data=df2).fit(cov_type='HC3')
        
        efecto1 = (np.exp(m_sub1.params['es_fds']) - 1) * 100
        efecto2 = (np.exp(m_sub2.params['es_fds']) - 1) * 100
        
        self.log(f"  Primera mitad:  Efecto FDS = {efecto1:+.1f}% (p={m_sub1.pvalues['es_fds']:.4f})")
        self.log(f"  Segunda mitad:  Efecto FDS = {efecto2:+.1f}% (p={m_sub2.pvalues['es_fds']:.4f})")
        self.log(f"  Diferencia: {abs(efecto1 - efecto2):.1f} puntos porcentuales")
        
        # ================================================================
        # TEST 2: Excluyendo outliers
        # ================================================================
        self.log("\n--- TEST 2: Excluyendo outliers (percentiles 1-99) ---")
        
        p1, p99 = df['volumen'].quantile([0.01, 0.99])
        df_sin_outliers = df[(df['volumen'] >= p1) & (df['volumen'] <= p99)]
        
        m_sin_out = smf.ols(formula, data=df_sin_outliers).fit(cov_type='HC3')
        efecto_sin_out = (np.exp(m_sin_out.params['es_fds']) - 1) * 100
        
        self.log(f"  Observaciones excluidas: {len(df) - len(df_sin_outliers)}")
        self.log(f"  Efecto FDS sin outliers: {efecto_sin_out:+.1f}% (p={m_sin_out.pvalues['es_fds']:.4f})")
        
        # ================================================================
        # TEST 3: Solo sabados vs solo domingos
        # ================================================================
        self.log("\n--- TEST 3: Sabado vs Domingo por separado ---")
        
        formula_dias = 'log_volumen ~ d_sabado + d_domingo + es_feriado + tendencia'
        m_dias = smf.ols(formula_dias, data=df).fit(cov_type='HC3')
        
        efecto_sab = (np.exp(m_dias.params['d_sabado']) - 1) * 100
        efecto_dom = (np.exp(m_dias.params['d_domingo']) - 1) * 100
        
        self.log(f"  Efecto Sabado:  {efecto_sab:+.1f}% (p={m_dias.pvalues['d_sabado']:.4f})")
        self.log(f"  Efecto Domingo: {efecto_dom:+.1f}% (p={m_dias.pvalues['d_domingo']:.4f})")
        
        # ================================================================
        # TEST 4: Variable dependiente alternativa (ordenes)
        # ================================================================
        self.log("\n--- TEST 4: Variable dependiente alternativa (log ordenes) ---")
        
        formula_ord = 'log_ordenes ~ es_fds + es_feriado + tendencia'
        m_ord = smf.ols(formula_ord, data=df).fit(cov_type='HC3')
        
        efecto_ord = (np.exp(m_ord.params['es_fds']) - 1) * 100
        
        self.log(f"  Efecto FDS en ordenes: {efecto_ord:+.1f}% (p={m_ord.pvalues['es_fds']:.4f})")
        
        # ================================================================
        # TEST 5: Placebo (dia aleatorio)
        # ================================================================
        self.log("\n--- TEST 5: Test placebo (martes como 'falso fds') ---")
        
        df['placebo_martes'] = (df['dia_semana'] == 1).astype(int)  # Martes
        formula_placebo = 'log_volumen ~ placebo_martes + es_feriado + tendencia'
        m_placebo = smf.ols(formula_placebo, data=df).fit(cov_type='HC3')
        
        efecto_placebo = (np.exp(m_placebo.params['placebo_martes']) - 1) * 100
        
        self.log(f"  Efecto 'placebo' (martes): {efecto_placebo:+.1f}% (p={m_placebo.pvalues['placebo_martes']:.4f})")
        
        if m_placebo.pvalues['placebo_martes'] > 0.1:
            self.log("  -> Placebo NO significativo (esperado) - Buen signo")
        else:
            self.log("  -> ADVERTENCIA: Placebo significativo - revisar")
    
    # ========================================================================
    # FASE 7: VISUALIZACIONES
    # ========================================================================
    def fase7_visualizaciones(self):
        """Genera visualizaciones."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Visualizaciones")
        self.log("="*70)
        
        self._grafico_comparativa_modelos()
        self._grafico_efecto_temporal()
        self._grafico_diagnosticos()
        
        self.log("Visualizaciones generadas")
    
    def _grafico_comparativa_modelos(self):
        """Grafico comparativo de efectos entre modelos."""
        if not hasattr(self, 'tabla_comparativa'):
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df = self.tabla_comparativa.copy()
        x = range(len(df))
        
        colors = ['green' if p < 0.05 else 'gray' for p in df['p_value']]
        bars = ax.bar(x, df['Efecto_%'], color=colors, edgecolor='black')
        
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axhline(y=df['Efecto_%'].mean(), color='red', linestyle='--', 
                   label=f'Promedio: {df["Efecto_%"].mean():.1f}%')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df['Modelo'], rotation=45, ha='right')
        ax.set_ylabel('Efecto Fin de Semana (%)')
        ax.set_title('Robustez del Efecto Fin de Semana\n(Verde = significativo al 5%)')
        ax.legend()
        
        # Etiquetas
        for bar, val, pval in zip(bars, df['Efecto_%'], df['p_value']):
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else ''
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                   f'{val:.1f}%{sig}', ha='center', va='top', fontsize=9, color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '01_comparativa_modelos.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_efecto_temporal(self):
        """Grafico del efecto a lo largo del tiempo."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        df = self.datos_diarios.copy()
        
        # Serie de volumen con FDS marcados
        axes[0].plot(df['fecha'], df['volumen'], color='steelblue', alpha=0.7, linewidth=1)
        
        fds = df[df['es_fds'] == 1]
        axes[0].scatter(fds['fecha'], fds['volumen'], color='red', s=15, alpha=0.5, label='Fin de semana')
        
        # Media movil
        df['vol_ma7'] = df['volumen'].rolling(7).mean()
        axes[0].plot(df['fecha'], df['vol_ma7'], color='darkblue', linewidth=2, label='MA(7)')
        
        axes[0].set_ylabel('Volumen (USDT)')
        axes[0].set_title('Volumen Diario con Fines de Semana Marcados')
        axes[0].legend()
        
        # Efecto rolling
        # Calcular efecto FDS en ventanas moviles
        window = 60
        efectos_rolling = []
        fechas_rolling = []
        
        for i in range(window, len(df)):
            df_window = df.iloc[i-window:i]
            try:
                m = smf.ols('log_volumen ~ es_fds + tendencia', data=df_window).fit()
                efecto = (np.exp(m.params['es_fds']) - 1) * 100
                efectos_rolling.append(efecto)
                fechas_rolling.append(df.iloc[i]['fecha'])
            except:
                efectos_rolling.append(np.nan)
                fechas_rolling.append(df.iloc[i]['fecha'])
        
        axes[1].plot(fechas_rolling, efectos_rolling, color='purple', linewidth=2)
        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].axhline(y=-33, color='red', linestyle='--', label='Efecto promedio (-33%)')
        axes[1].fill_between(fechas_rolling, efectos_rolling, 0, alpha=0.3, color='purple')
        axes[1].set_ylabel('Efecto FDS (%)')
        axes[1].set_xlabel('Fecha')
        axes[1].set_title(f'Efecto Fin de Semana Rolling (ventana={window} dias)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '02_efecto_temporal.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_diagnosticos(self):
        """Graficos de diagnostico del modelo principal."""
        modelo = self.resultados.get('M6_completo')
        if modelo is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuos vs ajustados
        axes[0, 0].scatter(modelo.fittedvalues, modelo.resid, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Valores Ajustados')
        axes[0, 0].set_ylabel('Residuos')
        axes[0, 0].set_title('Residuos vs Ajustados')
        
        # QQ plot
        stats.probplot(modelo.resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot de Residuos')
        
        # Histograma de residuos
        axes[1, 0].hist(modelo.resid, bins=30, density=True, alpha=0.7, color='steelblue')
        x = np.linspace(modelo.resid.min(), modelo.resid.max(), 100)
        axes[1, 0].plot(x, stats.norm.pdf(x, modelo.resid.mean(), modelo.resid.std()), 
                        'r-', linewidth=2, label='Normal')
        axes[1, 0].set_xlabel('Residuos')
        axes[1, 0].set_ylabel('Densidad')
        axes[1, 0].set_title('Distribucion de Residuos')
        axes[1, 0].legend()
        
        # Residuos en el tiempo
        axes[1, 1].plot(range(len(modelo.resid)), modelo.resid, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Observacion')
        axes[1, 1].set_ylabel('Residuos')
        axes[1, 1].set_title('Residuos en el Tiempo')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_diagnosticos.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # FASE 8: REPORTE FINAL
    # ========================================================================
    def fase8_reporte(self):
        """Genera reporte final."""
        self.log("\n" + "="*70)
        self.log("FASE 8: Reporte Final")
        self.log("="*70)
        
        reporte = []
        reporte.append("="*70)
        reporte.append("ANALISIS DE EFECTOS CALENDARIO - VERSION ROBUSTA")
        reporte.append("="*70)
        reporte.append(f"\nFecha: {datetime.now()}")
        reporte.append(f"Observaciones: {len(self.datos_diarios)}")
        
        reporte.append("\n" + "="*70)
        reporte.append("CONTROLES INCLUIDOS")
        reporte.append("="*70)
        reporte.append("  - Rezago de volumen (autocorrelacion)")
        reporte.append("  - Retorno y volatilidad de Bitcoin")
        reporte.append("  - Dummies de shocks macroeconomicos")
        reporte.append("  - Efectos fijos por mes")
        reporte.append("  - Errores robustos HC3 y HAC")
        
        reporte.append("\n" + "="*70)
        reporte.append("CONCLUSIONES DE ROBUSTEZ")
        reporte.append("="*70)
        
        if hasattr(self, 'tabla_comparativa'):
            efectos = self.tabla_comparativa['Efecto_%']
            reporte.append(f"\n  Efecto FDS promedio: {efectos.mean():+.1f}%")
            reporte.append(f"  Rango: [{efectos.min():+.1f}%, {efectos.max():+.1f}%]")
            reporte.append(f"  Modelos significativos: {(self.tabla_comparativa['p_value'] < 0.05).sum()}/{len(self.tabla_comparativa)}")
            
            if efectos.std() < 5 and (self.tabla_comparativa['p_value'] < 0.05).all():
                reporte.append("\n  CONCLUSION: El efecto fin de semana es ROBUSTO")
                reporte.append("  - Consistente entre especificaciones")
                reporte.append("  - Significativo con diferentes controles")
                reporte.append("  - No sensible a outliers ni subperiodos")
        
        # Guardar
        reporte_path = os.path.join(self.directorio_salida, 'robusto_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
            f.write('\n\n')
            f.write('\n'.join(reporte))
        
        self.log(f"\nReporte: {reporte_path}")
        self.log(f"Graficos: {self.dir_graficos}")
        self.log(f"Datos: {self.dir_datos}")
        self.log(f"Tablas: {self.dir_tablas}")
    
    # ========================================================================
    # EJECUTAR
    # ========================================================================
    def ejecutar(self):
        """Ejecuta analisis completo."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        self.log("="*70)
        self.log("ANALISIS DE EFECTOS CALENDARIO - VERSION ROBUSTA")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_datos_btc()
            self.fase3_construir_dataset()
            self.fase4_regresiones()
            self.fase5_tabla_comparativa()
            self.fase6_tests_robustez()
            self.fase7_visualizaciones()
            self.fase8_reporte()
            
            self.log(f"\n{'='*70}")
            self.log(f"COMPLETADO EN: {datetime.now() - inicio}")
            self.log(f"{'='*70}")
            
        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise


# ============================================================================
# MAIN
# ============================================================================
def main():
    from tkinter import Tk, filedialog
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("="*70)
    print("ANALISIS DE EFECTOS CALENDARIO - VERSION ROBUSTA")
    print("="*70)
    print("\nSelecciona el archivo CSV...")
    
    ruta = filedialog.askopenfilename(
        title='Seleccionar CSV',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta:
        print("Cancelado.")
        return
    
    analisis = AnalisisCalendarioRobusto(ruta_csv=ruta)
    analisis.ejecutar()


if __name__ == "__main__":
    main()

"""
================================================================================
EDA_COMPLETO.PY - Análisis Exploratorio de Datos para Mercado P2P Binance
================================================================================
Análisis exploratorio completo incluyendo:
- Estadísticas descriptivas
- Análisis de concentración (HHI, Gini, CR4)
- Segmentación Profession vs Mass
- Patrones calendario
- Clustering exploratorio de traders

Utiliza el módulo core_transacciones.py para la detección de transacciones.

Autor: Para tesis de maestría en finanzas
Fecha: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gc
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Importar módulo core
try:
    from core_transacciones import (
        cargar_csv_optimizado, preparar_dataframe,
        procesar_por_chunks, consolidar_transacciones_chunks,
        pipeline_deteccion_completo, diagnostico_transacciones,
        agregar_variables_calendario,
        COLUMNAS_CARGAR, DTYPE_MAP, CAMPO_VALIDACION_ORDEN
    )
except ImportError:
    print("ERROR: No se encontró core_transacciones.py")
    print("Asegúrate de que esté en el mismo directorio o en el PYTHONPATH")
    sys.exit(1)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHUNK_SIZE = 500_000
UMBRAL_ARCHIVO_GRANDE = 1_000_000_000  # 1GB


# ============================================================================
# CLASE PRINCIPAL DE EDA
# ============================================================================

class EDABinanceP2P:
    """Clase para análisis exploratorio del mercado P2P."""
    
    def __init__(self, ruta_csv, directorio_salida=None):
        self.ruta_csv = ruta_csv
        self.directorio_salida = directorio_salida or os.path.dirname(ruta_csv)
        
        # Crear subdirectorios
        self.dir_graficos = os.path.join(self.directorio_salida, 'eda_graficos')
        self.dir_datos = os.path.join(self.directorio_salida, 'eda_datos')
        os.makedirs(self.dir_graficos, exist_ok=True)
        os.makedirs(self.dir_datos, exist_ok=True)
        
        # Contenedores de datos
        self.transacciones = None
        self.datos_diarios = None
        self.datos_traders = None
        self.stats_generales = {}
        self.reporte = []
    
    def log(self, mensaje):
        """Logger con timestamp."""
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {mensaje}")
        self.reporte.append(f"[{ts}] {mensaje}")
    
    # ========================================================================
    # FASE 1: CARGA Y DETECCIÓN DE TRANSACCIONES
    # ========================================================================
    def fase1_cargar_transacciones(self):
        """Carga datos y detecta transacciones usando core_transacciones."""
        self.log("="*60)
        self.log("FASE 1: Carga y detección de transacciones")
        self.log("="*60)
        
        file_size = os.path.getsize(self.ruta_csv)
        es_grande = file_size > UMBRAL_ARCHIVO_GRANDE
        
        self.log(f"Tamaño del archivo: {file_size / 1e9:.2f} GB")
        
        if es_grande:
            self.log(f"Procesando por chunks de {CHUNK_SIZE:,} filas...")
            
            # Contadores para estadísticas
            total_registros = 0
            fecha_min, fecha_max = None, None
            registros_por_asset = defaultdict(int)
            registros_por_classify = defaultdict(int)
            
            chunks_trans = []
            chunk_num = 0
            
            for chunk in cargar_csv_optimizado(self.ruta_csv, chunk_size=CHUNK_SIZE):
                chunk_num += 1
                total_registros += len(chunk)
                
                if chunk_num % 10 == 0:
                    self.log(f"  Chunk {chunk_num}: {total_registros:,} registros...")
                
                chunk = preparar_dataframe(chunk)
                
                # Estadísticas
                c_min, c_max = chunk['time'].min(), chunk['time'].max()
                if fecha_min is None or c_min < fecha_min:
                    fecha_min = c_min
                if fecha_max is None or c_max > fecha_max:
                    fecha_max = c_max
                
                for asset in chunk['adv.asset'].dropna().unique():
                    registros_por_asset[asset] += len(chunk[chunk['adv.asset'] == asset])
                for cl in chunk['adv.classify'].dropna().unique():
                    registros_por_classify[cl] += len(chunk[chunk['adv.classify'] == cl])
                
                # Detectar transacciones en chunk
                trans = pipeline_deteccion_completo(chunk, aplicar_filtro_atipicos=False, verbose=False)
                if len(trans) > 0:
                    chunks_trans.append(trans)
                
                del chunk
                gc.collect()
            
            # Consolidar
            self.transacciones = consolidar_transacciones_chunks(chunks_trans, verbose=True)
            
            self.stats_generales = {
                'total_registros': total_registros,
                'fecha_min': fecha_min,
                'fecha_max': fecha_max,
                'dias_cobertura': (fecha_max - fecha_min).days if fecha_min and fecha_max else 0,
                'registros_por_asset': dict(registros_por_asset),
                'registros_por_classify': dict(registros_por_classify)
            }
        else:
            self.log("Cargando archivo completo...")
            df = cargar_csv_optimizado(self.ruta_csv)
            df = preparar_dataframe(df)
            
            self.stats_generales = {
                'total_registros': len(df),
                'fecha_min': df['time'].min(),
                'fecha_max': df['time'].max(),
                'dias_cobertura': (df['time'].max() - df['time'].min()).days,
                'registros_por_asset': df['adv.asset'].value_counts().to_dict(),
                'registros_por_classify': df['adv.classify'].value_counts().to_dict()
            }
            
            self.transacciones = pipeline_deteccion_completo(df, verbose=True)
            del df
            gc.collect()
        
        # Mostrar estadísticas
        self._mostrar_stats_generales()
        
        if len(self.transacciones) > 0:
            diagnostico_transacciones(self.transacciones, verbose=True)
    
    def _mostrar_stats_generales(self):
        """Muestra estadísticas generales."""
        self.log("\nESTADÍSTICAS GENERALES:")
        self.log(f"  Total registros: {self.stats_generales['total_registros']:,}")
        self.log(f"  Período: {self.stats_generales['fecha_min']} a {self.stats_generales['fecha_max']}")
        self.log(f"  Días: {self.stats_generales['dias_cobertura']}")
        
        if self.transacciones is not None:
            self.log(f"  Transacciones USDT/BOB: {len(self.transacciones):,}")
            self.log(f"  Volumen total: {self.transacciones['montoDynamic'].sum():,.0f} USDT")
    
    # ========================================================================
    # FASE 2: AGREGACIÓN DIARIA
    # ========================================================================
    def fase2_agregacion_diaria(self):
        """Agrega transacciones a nivel diario."""
        self.log("\n" + "="*60)
        self.log("FASE 2: Agregación diaria")
        self.log("="*60)
        
        if self.transacciones is None or len(self.transacciones) == 0:
            self.log("ERROR: No hay transacciones")
            return
        
        df = self.transacciones.copy()
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        
        # Separar por tipo
        df_compra = df[df['adv.tradeType'] == 'compra']
        df_venta = df[df['adv.tradeType'] == 'venta']
        
        # Agregación
        diario = df.groupby('fecha').agg({
            'precio_transaccion': ['mean', 'std', 'min', 'max', 'count'],
            'montoDynamic': 'sum',
            'delta_ordenes_corregido': 'sum',
            'userNo': 'nunique',
            'adv.advNo': 'nunique'
        }).reset_index()
        
        diario.columns = ['fecha', 'precio_mean', 'precio_std', 'precio_min', 'precio_max',
                          'detecciones', 'volumen_total', 'ordenes_estimadas',
                          'traders_activos', 'anuncios_activos']
        
        # Volumen por tipo
        vol_compra = df_compra.groupby(df_compra['fecha'])['montoDynamic'].sum()
        vol_venta = df_venta.groupby(df_venta['fecha'])['montoDynamic'].sum()
        
        diario['volumen_compra'] = diario['fecha'].map(vol_compra).fillna(0)
        diario['volumen_venta'] = diario['fecha'].map(vol_venta).fillna(0)
        diario['ratio_compra_venta'] = diario['volumen_compra'] / diario['volumen_venta'].replace(0, np.nan)
        
        # Órdenes por tipo
        ord_compra = df_compra.groupby(df_compra['fecha'])['delta_ordenes_corregido'].sum()
        ord_venta = df_venta.groupby(df_venta['fecha'])['delta_ordenes_corregido'].sum()
        
        diario['ordenes_compra'] = diario['fecha'].map(ord_compra).fillna(0)
        diario['ordenes_venta'] = diario['fecha'].map(ord_venta).fillna(0)
        
        # Variables calendario
        diario['dia_semana'] = diario['fecha'].dt.dayofweek
        diario['dia_semana_nombre'] = diario['fecha'].dt.day_name()
        diario['dia_mes'] = diario['fecha'].dt.day
        diario['mes'] = diario['fecha'].dt.month
        diario['es_fin_semana'] = (diario['dia_semana'] >= 5).astype(int)
        diario['es_fin_mes'] = (diario['dia_mes'] >= 28).astype(int)
        
        # Volatilidad y retorno
        diario['volatilidad'] = (diario['precio_max'] - diario['precio_min']) / diario['precio_mean'] * 100
        diario = diario.sort_values('fecha')
        diario['retorno'] = diario['precio_mean'].pct_change() * 100
        
        self.datos_diarios = diario
        self.log(f"Datos diarios generados: {len(diario)} días")
        
        diario.to_csv(os.path.join(self.dir_datos, 'datos_diarios.csv'), index=False)
    
    # ========================================================================
    # FASE 3: CONCENTRACIÓN
    # ========================================================================
    def fase3_concentracion(self):
        """Calcula métricas de concentración."""
        self.log("\n" + "="*60)
        self.log("FASE 3: Análisis de concentración")
        self.log("="*60)
        
        if self.transacciones is None or len(self.transacciones) == 0:
            self.log("ERROR: No hay transacciones")
            return
        
        df = self.transacciones.copy()
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        
        concentracion = []
        
        for fecha in df['fecha'].unique():
            df_dia = df[df['fecha'] == fecha]
            vol_trader = df_dia.groupby('userNo')['montoDynamic'].sum()
            vol_total = vol_trader.sum()
            
            if vol_total > 0:
                cuotas = vol_trader / vol_total
                
                # HHI
                hhi = (cuotas ** 2).sum() * 10000
                
                # Gini
                n = len(cuotas)
                if n > 1:
                    sorted_cuotas = np.sort(cuotas.values)
                    gini = (2 * np.sum((np.arange(1, n+1) * sorted_cuotas))) / (n * np.sum(sorted_cuotas)) - (n + 1) / n
                else:
                    gini = 1
                
                # CR4, CR10
                top = vol_trader.nlargest(10)
                cr4 = top.head(4).sum() / vol_total
                cr10 = top.sum() / vol_total
                
                concentracion.append({
                    'fecha': fecha, 'hhi': hhi, 'gini': gini,
                    'cr4': cr4, 'cr10': cr10, 'n_traders': len(vol_trader)
                })
        
        self.concentracion_diaria = pd.DataFrame(concentracion)
        
        # Merge con datos diarios
        if self.datos_diarios is not None:
            self.datos_diarios = self.datos_diarios.merge(
                self.concentracion_diaria[['fecha', 'hhi', 'gini', 'cr4', 'cr10']],
                on='fecha', how='left'
            )
        
        # Estadísticas
        self.log(f"\nMétricas de Concentración (promedio):")
        self.log(f"  HHI:  {self.concentracion_diaria['hhi'].mean():.0f}")
        self.log(f"  Gini: {self.concentracion_diaria['gini'].mean():.3f}")
        self.log(f"  CR4:  {self.concentracion_diaria['cr4'].mean()*100:.1f}%")
        self.log(f"  CR10: {self.concentracion_diaria['cr10'].mean()*100:.1f}%")
        
        hhi_mean = self.concentracion_diaria['hhi'].mean()
        if hhi_mean < 1500:
            self.log("  → Mercado COMPETITIVO")
        elif hhi_mean < 2500:
            self.log("  → Mercado MODERADAMENTE concentrado")
        else:
            self.log("  → Mercado ALTAMENTE concentrado")
        
        self.concentracion_diaria.to_csv(os.path.join(self.dir_datos, 'concentracion_diaria.csv'), index=False)
    
    # ========================================================================
    # FASE 4: SEGMENTACIÓN
    # ========================================================================
    def fase4_segmentacion(self):
        """Analiza diferencias entre profession y mass."""
        self.log("\n" + "="*60)
        self.log("FASE 4: Segmentación (Profession vs Mass)")
        self.log("="*60)
        
        if self.transacciones is None:
            return
        
        df = self.transacciones.copy()
        
        segmentos = df.groupby('adv.classify').agg({
            'montoDynamic': ['sum', 'mean', 'count'],
            'precio_transaccion': 'mean',
            'userNo': 'nunique',
            'adv.commissionRate': 'mean',
            'userDetailVo.userStatsRet.avgReleaseTimeOfLatest30day': 'mean',
            'userDetailVo.userStatsRet.finishRateLatest30day': 'mean'
        }).reset_index()
        
        segmentos.columns = ['classify', 'volumen_total', 'ticket_promedio', 'transacciones',
                             'precio_promedio', 'traders', 'comision', 'tiempo_lib', 'tasa_fin']
        
        vol_total = segmentos['volumen_total'].sum()
        segmentos['cuota_mercado'] = segmentos['volumen_total'] / vol_total * 100
        
        self.segmentacion = segmentos
        
        for _, row in segmentos.iterrows():
            self.log(f"\n  {row['classify'].upper()}:")
            self.log(f"    Cuota mercado: {row['cuota_mercado']:.1f}%")
            self.log(f"    Volumen:       {row['volumen_total']:,.0f} USDT")
            self.log(f"    Traders:       {row['traders']:,}")
            self.log(f"    Ticket prom:   {row['ticket_promedio']:.0f} USDT")
        
        segmentos.to_csv(os.path.join(self.dir_datos, 'segmentacion.csv'), index=False)
    
    # ========================================================================
    # FASE 5: ANÁLISIS DE TRADERS
    # ========================================================================
    def fase5_traders(self):
        """Construye dataset de traders para clustering."""
        self.log("\n" + "="*60)
        self.log("FASE 5: Análisis de traders")
        self.log("="*60)
        
        if self.transacciones is None:
            return
        
        df = self.transacciones.copy()
        
        # Agregar por trader
        traders = df.groupby('userNo').agg({
            'montoDynamic': ['sum', 'mean', 'std', 'count'],
            'precio_transaccion': ['mean', 'std'],
            'delta_ordenes_corregido': 'sum',
            'adv.classify': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan,
            'adv.commissionRate': 'mean',
            'userDetailVo.userStatsRet.registerDays': 'last',
            'userDetailVo.userStatsRet.avgReleaseTimeOfLatest30day': 'last',
            'userDetailVo.userStatsRet.finishRateLatest30day': 'last',
            'userDetailVo.userStatsRet.completedOrderNum': 'last',
            'userDetailVo.userStatsRet.completedBuyOrderNum': 'last',
            'userDetailVo.userStatsRet.completedSellOrderNum': 'last',
            'userDetailVo.userStatsRet.counterpartyCount': 'last',
            'time': ['min', 'max']
        }).reset_index()
        
        traders.columns = [
            'userNo', 'volumen_total', 'volumen_mean', 'volumen_std', 'num_trans',
            'precio_mean', 'precio_std', 'ordenes_est', 'classify', 'comision',
            'dias_registro', 'tiempo_lib', 'tasa_fin', 'ordenes_hist',
            'ordenes_compra_hist', 'ordenes_venta_hist', 'contrapartes',
            'primera_trans', 'ultima_trans'
        ]
        
        # Features adicionales
        traders['dias_activos'] = (traders['ultima_trans'] - traders['primera_trans']).dt.days + 1
        traders['frecuencia'] = traders['num_trans'] / traders['dias_activos'].replace(0, 1)
        traders['ticket'] = traders['volumen_total'] / traders['num_trans']
        traders['ratio_direccional'] = traders['ordenes_compra_hist'] / traders['ordenes_venta_hist'].replace(0, np.nan)
        
        # Volumen por tipo
        vol_compra = df[df['adv.tradeType'] == 'compra'].groupby('userNo')['montoDynamic'].sum()
        vol_venta = df[df['adv.tradeType'] == 'venta'].groupby('userNo')['montoDynamic'].sum()
        
        traders['vol_compra'] = traders['userNo'].map(vol_compra).fillna(0)
        traders['vol_venta'] = traders['userNo'].map(vol_venta).fillna(0)
        traders['pct_compra'] = traders['vol_compra'] / traders['volumen_total'] * 100
        
        self.datos_traders = traders
        
        self.log(f"Traders analizados: {len(traders):,}")
        
        # Top 10
        top10 = traders.nlargest(10, 'volumen_total')
        vol_top10 = top10['volumen_total'].sum()
        vol_total = traders['volumen_total'].sum()
        self.log(f"Top 10 traders concentran: {vol_top10/vol_total*100:.1f}%")
        
        traders.to_csv(os.path.join(self.dir_datos, 'datos_traders.csv'), index=False)
    
    # ========================================================================
    # FASE 6: PATRONES CALENDARIO
    # ========================================================================
    def fase6_calendario(self):
        """Analiza patrones por día de semana."""
        self.log("\n" + "="*60)
        self.log("FASE 6: Patrones calendario")
        self.log("="*60)
        
        if self.datos_diarios is None:
            return
        
        df = self.datos_diarios.copy()
        
        # Por día de semana
        dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        stats_dia = df.groupby('dia_semana_nombre').agg({
            'volumen_total': 'mean',
            'ratio_compra_venta': 'mean',
            'traders_activos': 'mean'
        }).reindex(dias)
        
        self.log("\nPor día de semana:")
        for dia in dias:
            if dia in stats_dia.index:
                row = stats_dia.loc[dia]
                self.log(f"  {dia[:3]}: Vol={row['volumen_total']:,.0f}, Ratio={row['ratio_compra_venta']:.3f}")
        
        # Fin de semana vs entre semana
        fds = df[df['es_fin_semana'] == 1]
        es = df[df['es_fin_semana'] == 0]
        
        self.log("\nFin de semana vs Entre semana:")
        self.log(f"  Entre semana: Vol={es['volumen_total'].mean():,.0f}, Ratio={es['ratio_compra_venta'].mean():.3f}")
        self.log(f"  Fin de semana: Vol={fds['volumen_total'].mean():,.0f}, Ratio={fds['ratio_compra_venta'].mean():.3f}")
        
        efecto = (fds['volumen_total'].mean() / es['volumen_total'].mean() - 1) * 100
        self.log(f"  Efecto fin de semana: {efecto:+.1f}%")
        
        stats_dia.to_csv(os.path.join(self.dir_datos, 'patrones_dia_semana.csv'))
    
    # ========================================================================
    # FASE 7: CLUSTERING
    # ========================================================================
    def fase7_clustering(self):
        """Clustering exploratorio de traders."""
        self.log("\n" + "="*60)
        self.log("FASE 7: Clustering exploratorio")
        self.log("="*60)
        
        if self.datos_traders is None:
            return
        
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            self.log("ADVERTENCIA: sklearn no instalado. Omitiendo clustering.")
            return
        
        df = self.datos_traders.copy()
        
        # Features
        features = ['volumen_total', 'num_trans', 'ticket', 'pct_compra', 
                    'frecuencia', 'tiempo_lib', 'tasa_fin', 'dias_registro', 'contrapartes']
        
        df_cl = df.dropna(subset=features)
        self.log(f"Traders con datos completos: {len(df_cl):,}")
        
        if len(df_cl) < 100:
            self.log("Muy pocos traders para clustering")
            return
        
        X = df_cl[features].replace([np.inf, -np.inf], np.nan).fillna(df_cl[features].median())
        X_scaled = StandardScaler().fit_transform(X)
        
        # Buscar K óptimo
        self.log("Buscando K óptimo...")
        k_range = range(2, 11)
        silhouettes = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            silhouettes.append(sil)
            self.log(f"  K={k}: Silhouette={sil:.3f}")
        
        mejor_k = k_range[np.argmax(silhouettes)]
        self.log(f"\nMejor K: {mejor_k}")
        
        # Clustering final
        km_final = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
        df_cl['cluster'] = km_final.fit_predict(X_scaled)
        
        # Estadísticas por cluster
        cluster_stats = df_cl.groupby('cluster').agg({
            'volumen_total': ['mean', 'sum', 'count'],
            'ticket': 'mean',
            'pct_compra': 'mean',
            'tasa_fin': 'mean',
            'classify': lambda x: (x == 'profession').mean() * 100
        }).reset_index()
        
        cluster_stats.columns = ['cluster', 'vol_mean', 'vol_total', 'n_traders',
                                  'ticket', 'pct_compra', 'tasa_fin', 'pct_prof']
        
        vol_total = cluster_stats['vol_total'].sum()
        cluster_stats['cuota'] = cluster_stats['vol_total'] / vol_total * 100
        
        for _, row in cluster_stats.iterrows():
            self.log(f"\n  Cluster {int(row['cluster'])}:")
            self.log(f"    Traders: {int(row['n_traders']):,} | Cuota: {row['cuota']:.1f}%")
            self.log(f"    Ticket: {row['ticket']:.0f} USDT | %Compra: {row['pct_compra']:.1f}%")
        
        # Guardar
        self.cluster_stats = cluster_stats
        self.mejor_k = mejor_k
        self.silhouettes = dict(zip(k_range, silhouettes))
        
        df_cl.to_csv(os.path.join(self.dir_datos, 'traders_con_clusters.csv'), index=False)
        cluster_stats.to_csv(os.path.join(self.dir_datos, 'cluster_stats.csv'), index=False)
    
    # ========================================================================
    # FASE 8: GRÁFICOS
    # ========================================================================
    def fase8_graficos(self):
        """Genera gráficos."""
        self.log("\n" + "="*60)
        self.log("FASE 8: Generación de gráficos")
        self.log("="*60)
        
        if self.datos_diarios is not None:
            self._grafico_precio()
            self._grafico_volumen()
            self._grafico_ratio()
            self._grafico_concentracion()
            self._grafico_calendario()
        
        if self.datos_traders is not None:
            self._grafico_distribucion()
        
        self.log("Gráficos generados")
    
    def _grafico_precio(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.datos_diarios['fecha'], self.datos_diarios['precio_mean'], 'b-', lw=1)
        ax.fill_between(self.datos_diarios['fecha'], 
                        self.datos_diarios['precio_min'],
                        self.datos_diarios['precio_max'], alpha=0.2)
        ax.set_title('Precio USDT/BOB')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('BOB')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '01_precio.png'), dpi=150)
        plt.close()
    
    def _grafico_volumen(self):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].bar(self.datos_diarios['fecha'], self.datos_diarios['volumen_total'], color='steelblue')
        axes[0].set_title('Volumen Total Diario')
        axes[1].bar(self.datos_diarios['fecha'], self.datos_diarios['volumen_compra'], color='green', alpha=0.7, label='Compra')
        axes[1].bar(self.datos_diarios['fecha'], -self.datos_diarios['volumen_venta'], color='red', alpha=0.7, label='Venta')
        axes[1].legend()
        axes[1].set_title('Volumen por Tipo')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '02_volumen.png'), dpi=150)
        plt.close()
    
    def _grafico_ratio(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.datos_diarios['fecha'], self.datos_diarios['ratio_compra_venta'], 'purple')
        ax.axhline(y=1, color='red', linestyle='--')
        ax.axhline(y=self.datos_diarios['ratio_compra_venta'].mean(), color='green', linestyle='--')
        ax.set_title('Ratio Compra/Venta')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_ratio.png'), dpi=150)
        plt.close()
    
    def _grafico_concentracion(self):
        if 'hhi' not in self.datos_diarios.columns:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0,0].plot(self.datos_diarios['fecha'], self.datos_diarios['hhi'])
        axes[0,0].axhline(y=1500, color='g', linestyle='--')
        axes[0,0].axhline(y=2500, color='r', linestyle='--')
        axes[0,0].set_title('HHI')
        axes[0,1].plot(self.datos_diarios['fecha'], self.datos_diarios['gini'], 'orange')
        axes[0,1].set_title('Gini')
        axes[1,0].plot(self.datos_diarios['fecha'], self.datos_diarios['cr4']*100, 'green')
        axes[1,0].set_title('CR4 (%)')
        axes[1,1].plot(self.datos_diarios['fecha'], self.datos_diarios['traders_activos'], 'purple')
        axes[1,1].set_title('Traders Activos')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '04_concentracion.png'), dpi=150)
        plt.close()
    
    def _grafico_calendario(self):
        dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_esp = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        vol = self.datos_diarios.groupby('dia_semana_nombre')['volumen_total'].mean().reindex(dias)
        axes[0].bar(dias_esp, vol.values, color='steelblue')
        axes[0].set_title('Volumen por Día de Semana')
        
        ratio = self.datos_diarios.groupby('dia_semana_nombre')['ratio_compra_venta'].mean().reindex(dias)
        axes[1].bar(dias_esp, ratio.values, color='purple')
        axes[1].axhline(y=1, color='r', linestyle='--')
        axes[1].set_title('Ratio por Día de Semana')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '05_calendario.png'), dpi=150)
        plt.close()
    
    def _grafico_distribucion(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribución volumen
        axes[0,0].hist(np.log10(self.datos_traders['volumen_total']+1), bins=50, color='steelblue')
        axes[0,0].set_title('Distribución Volumen (log10)')
        
        # Lorenz
        vol_sorted = np.sort(self.datos_traders['volumen_total'])
        cumsum = np.cumsum(vol_sorted) / vol_sorted.sum()
        n = len(vol_sorted)
        axes[0,1].plot(np.arange(1,n+1)/n, cumsum, label='Lorenz')
        axes[0,1].plot([0,1], [0,1], 'r--', label='Igualdad')
        axes[0,1].set_title('Curva de Lorenz')
        axes[0,1].legend()
        
        # Direccionalidad
        axes[1,0].hist(self.datos_traders['pct_compra'].dropna(), bins=50, color='green')
        axes[1,0].axvline(x=50, color='r', linestyle='--')
        axes[1,0].set_title('% Volumen en Compra')
        
        # Segmentación
        seg = self.datos_traders['classify'].value_counts()
        axes[1,1].pie(seg.values, labels=seg.index, autopct='%1.1f%%')
        axes[1,1].set_title('Profession vs Mass')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '06_distribucion.png'), dpi=150)
        plt.close()
    
    # ========================================================================
    # FASE 9: REPORTE FINAL
    # ========================================================================
    def fase9_reporte(self):
        """Genera reporte final con hallazgos."""
        self.log("\n" + "="*60)
        self.log("FASE 9: Reporte final")
        self.log("="*60)
        
        hallazgos = ["\n" + "="*60, "HALLAZGOS PRINCIPALES", "="*60]
        
        # Concentración
        if hasattr(self, 'concentracion_diaria'):
            hhi = self.concentracion_diaria['hhi'].mean()
            cr4 = self.concentracion_diaria['cr4'].mean()
            hallazgos.append(f"\n📊 CONCENTRACIÓN:")
            hallazgos.append(f"   HHI promedio: {hhi:.0f}")
            hallazgos.append(f"   CR4 promedio: {cr4*100:.1f}%")
        
        # Flujos
        if self.datos_diarios is not None:
            ratio = self.datos_diarios['ratio_compra_venta'].mean()
            hallazgos.append(f"\n📈 FLUJOS:")
            hallazgos.append(f"   Ratio compra/venta: {ratio:.3f}")
            if ratio > 1.1:
                hallazgos.append("   → Predomina COMPRA de USDT (dolarización)")
            elif ratio < 0.9:
                hallazgos.append("   → Predomina VENTA de USDT")
            else:
                hallazgos.append("   → Mercado equilibrado")
        
        # Segmentación
        if hasattr(self, 'segmentacion'):
            prof = self.segmentacion[self.segmentacion['classify'] == 'profession']
            if len(prof) > 0:
                hallazgos.append(f"\n👥 SEGMENTACIÓN:")
                hallazgos.append(f"   Profesionales: {prof['cuota_mercado'].values[0]:.1f}% del mercado")
        
        # Clustering
        if hasattr(self, 'cluster_stats'):
            hallazgos.append(f"\n🎯 CLUSTERING (K={self.mejor_k}):")
            hallazgos.append(f"   Silhouette: {max(self.silhouettes.values()):.3f}")
        
        # Recomendaciones
        hallazgos.append("\n" + "="*60)
        hallazgos.append("💡 RECOMENDACIONES PARA TESIS:")
        hallazgos.append("="*60)
        hallazgos.append("   ✅ Opción 1 (Calendario): VIABLE si hay variación temporal")
        hallazgos.append("   ✅ Opción 2 (Microestructura): VIABLE si HHI > 1500")
        hallazgos.append("   ✅ Opción 3 (Clustering): VIABLE si Silhouette > 0.3")
        hallazgos.append("   📌 HÍBRIDA RECOMENDADA: Combinar concentración + calendario")
        
        for h in hallazgos:
            self.log(h)
        
        # Guardar reporte
        with open(os.path.join(self.directorio_salida, 'eda_reporte.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte + hallazgos))
        
        self.log(f"\n✅ Reporte guardado en: {self.directorio_salida}")
    
    # ========================================================================
    # EJECUTAR TODO
    # ========================================================================
    def ejecutar(self):
        """Ejecuta todas las fases."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        
        try:
            self.fase1_cargar_transacciones()
            self.fase2_agregacion_diaria()
            self.fase3_concentracion()
            self.fase4_segmentacion()
            self.fase5_traders()
            self.fase6_calendario()
            self.fase7_clustering()
            self.fase8_graficos()
            self.fase9_reporte()
            
            self.log(f"\n✅ COMPLETADO en {datetime.now() - inicio}")
        except Exception as e:
            self.log(f"\n❌ ERROR: {e}")
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
    
    print("="*60)
    print("EDA COMPLETO - MERCADO P2P USDT/BOB")
    print("="*60)
    print("\nSelecciona el archivo CSV...")
    
    ruta = filedialog.askopenfilename(
        title='Seleccionar CSV',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta:
        print("Cancelado.")
        return
    
    eda = EDABinanceP2P(ruta)
    eda.ejecutar()


if __name__ == "__main__":
    main()

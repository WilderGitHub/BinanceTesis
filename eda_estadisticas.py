"""
================================================================================
EDA_ESTADISTICAS.PY - Estadísticas Descriptivas y Concentración de Mercado
================================================================================
Análisis exploratorio enfocado en:
- Estadísticas generales del dataset
- Métricas de concentración (HHI, Gini, CR4, CR10)
- Segmentación Binance (Profession vs Mass)
- Análisis de traders
- Curva de Lorenz

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
from scipy import stats

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Importar módulo core
try:
    from core_transacciones import (
        cargar_csv_optimizado, preparar_dataframe,
        consolidar_transacciones_chunks,
        pipeline_deteccion_completo
    )
except ImportError as e:
    print(f"ERROR: No se pudo importar core_transacciones.py")
    print(f"Detalle: {e}")
    sys.exit(1)

CHUNK_SIZE = 500_000


class EDAEstadisticas:
    """Análisis exploratorio: estadísticas y concentración."""
    
    def __init__(self, ruta_csv, directorio_salida=None):
        self.ruta_csv = ruta_csv
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_csv)
        
        # Subdirectorios
        self.dir_graficos = os.path.join(self.directorio_salida, 'eda_graficos')
        self.dir_datos = os.path.join(self.directorio_salida, 'eda_datos')
        self.dir_tablas = os.path.join(self.directorio_salida, 'eda_tablas')
        os.makedirs(self.dir_graficos, exist_ok=True)
        os.makedirs(self.dir_datos, exist_ok=True)
        os.makedirs(self.dir_tablas, exist_ok=True)
        
        # Datos
        self.transacciones = None
        self.datos_diarios = None
        self.metricas_concentracion = None
        self.stats_traders = None
        self.reporte = []
    
    def log(self, mensaje):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {mensaje}")
        self.reporte.append(f"[{ts}] {mensaje}")
    
    # ========================================================================
    # FASE 1: CARGA DE DATOS
    # ========================================================================
    def fase1_cargar_datos(self):
        """Carga transacciones del CSV."""
        self.log("="*70)
        self.log("FASE 1: Carga de datos")
        self.log("="*70)
        
        file_size = os.path.getsize(self.ruta_csv)
        self.log(f"Archivo: {self.ruta_csv}")
        self.log(f"Tamano: {file_size / 1e9:.2f} GB")
        
        if file_size > 1e9:
            self.log("Procesando por chunks...")
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
                gc.collect()
            
            self.transacciones = consolidar_transacciones_chunks(chunks, verbose=False)
        else:
            df = cargar_csv_optimizado(self.ruta_csv)
            df = preparar_dataframe(df)
            self.transacciones = pipeline_deteccion_completo(df, verbose=False)
        
        self.log(f"\nTransacciones detectadas: {len(self.transacciones):,}")
    
    # ========================================================================
    # FASE 2: ESTADÍSTICAS GENERALES
    # ========================================================================
    def fase2_estadisticas_generales(self):
        """Calcula estadísticas generales del dataset."""
        self.log("\n" + "="*70)
        self.log("FASE 2: Estadisticas Generales")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Período
        fecha_min = df['time'].min()
        fecha_max = df['time'].max()
        dias = (fecha_max - fecha_min).days + 1
        
        # Volumen
        volumen_total = df['montoDynamic'].sum()
        volumen_compra = df[df['adv.tradeType'] == 'compra']['montoDynamic'].sum()
        volumen_venta = df[df['adv.tradeType'] == 'venta']['montoDynamic'].sum()
        
        # Transacciones
        n_trans = len(df)
        n_compras = (df['adv.tradeType'] == 'compra').sum()
        n_ventas = (df['adv.tradeType'] == 'venta').sum()
        
        # Traders
        n_traders = df['userNo'].nunique()
        
        # Ticket
        ticket_promedio = df['montoDynamic'].mean()
        ticket_mediano = df['montoDynamic'].median()
        
        # Precio
        precio_promedio = df['precio_transaccion'].mean()
        precio_std = df['precio_transaccion'].std()
        
        # Guardar stats
        self.stats_generales = {
            'fecha_inicio': fecha_min,
            'fecha_fin': fecha_max,
            'dias': dias,
            'volumen_total': volumen_total,
            'volumen_compra': volumen_compra,
            'volumen_venta': volumen_venta,
            'ratio_cv': volumen_compra / volumen_venta,
            'n_transacciones': n_trans,
            'n_compras': n_compras,
            'n_ventas': n_ventas,
            'n_traders': n_traders,
            'ticket_promedio': ticket_promedio,
            'ticket_mediano': ticket_mediano,
            'precio_promedio': precio_promedio,
            'precio_std': precio_std,
            'trans_por_dia': n_trans / dias,
            'volumen_por_dia': volumen_total / dias
        }
        
        # Mostrar
        self.log(f"\n{'='*50}")
        self.log("RESUMEN DEL DATASET")
        self.log(f"{'='*50}")
        self.log(f"  Periodo: {fecha_min.date()} a {fecha_max.date()} ({dias} dias)")
        self.log(f"  Transacciones: {n_trans:,}")
        self.log(f"  Volumen total: {volumen_total:,.0f} USDT")
        self.log(f"  Traders unicos: {n_traders:,}")
        self.log(f"\n  Volumen compra: {volumen_compra:,.0f} USDT ({volumen_compra/volumen_total*100:.1f}%)")
        self.log(f"  Volumen venta:  {volumen_venta:,.0f} USDT ({volumen_venta/volumen_total*100:.1f}%)")
        self.log(f"  Ratio C/V: {volumen_compra/volumen_venta:.3f}")
        self.log(f"\n  Ticket promedio: ${ticket_promedio:,.2f}")
        self.log(f"  Ticket mediano:  ${ticket_mediano:,.2f}")
        self.log(f"  Precio promedio: {precio_promedio:,.2f} BOB/USDT")
        self.log(f"\n  Transacciones/dia: {n_trans/dias:,.0f}")
        self.log(f"  Volumen/dia: {volumen_total/dias:,.0f} USDT")
        
        # Interpretación
        if volumen_compra > volumen_venta:
            self.log(f"\n  -> Predomina COMPRA de USDT (dolarizacion)")
        else:
            self.log(f"\n  -> Predomina VENTA de USDT")
    
    # ========================================================================
    # FASE 3: AGREGACIÓN DIARIA
    # ========================================================================
    def fase3_agregacion_diaria(self):
        """Agrega datos a nivel diario."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Agregacion Diaria")
        self.log("="*70)
        
        df = self.transacciones.copy()
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        
        # Agregación
        diario = df.groupby('fecha').agg({
            'montoDynamic': ['sum', 'mean', 'median', 'count'],
            'precio_transaccion': ['mean', 'std', 'min', 'max'],
            'userNo': 'nunique'
        }).reset_index()
        
        diario.columns = ['fecha', 'volumen', 'ticket_mean', 'ticket_median', 'n_trans',
                          'precio_mean', 'precio_std', 'precio_min', 'precio_max', 'n_traders']
        
        # Volumen por tipo
        vol_compra = df[df['adv.tradeType'] == 'compra'].groupby('fecha')['montoDynamic'].sum()
        vol_venta = df[df['adv.tradeType'] == 'venta'].groupby('fecha')['montoDynamic'].sum()
        
        diario['vol_compra'] = diario['fecha'].map(vol_compra).fillna(0)
        diario['vol_venta'] = diario['fecha'].map(vol_venta).fillna(0)
        diario['ratio_cv'] = diario['vol_compra'] / diario['vol_venta'].replace(0, np.nan)
        
        # Variables calendario
        diario['dia_semana'] = diario['fecha'].dt.dayofweek
        diario['dia_nombre'] = diario['fecha'].dt.day_name()
        diario['mes'] = diario['fecha'].dt.month
        diario['es_fds'] = (diario['dia_semana'] >= 5).astype(int)
        
        self.datos_diarios = diario
        
        self.log(f"Dias con datos: {len(diario)}")
        self.log(f"Volumen diario promedio: {diario['volumen'].mean():,.0f} USDT")
        self.log(f"Transacciones diarias promedio: {diario['n_trans'].mean():,.0f}")
        
        # Guardar
        diario.to_csv(os.path.join(self.dir_datos, 'datos_diarios.csv'), index=False)
    
    # ========================================================================
    # FASE 4: CONCENTRACIÓN DE MERCADO
    # ========================================================================
    def fase4_concentracion_mercado(self):
        """Calcula métricas de concentración de mercado."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Concentracion de Mercado")
        self.log("="*70)
        
        df = self.transacciones.copy()
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        
        # Calcular concentración diaria
        concentraciones = []
        
        for fecha, grupo in df.groupby('fecha'):
            vol_por_trader = grupo.groupby('userNo')['montoDynamic'].sum()
            vol_total = vol_por_trader.sum()
            
            if vol_total == 0 or len(vol_por_trader) < 2:
                continue
            
            # Participaciones de mercado
            shares = vol_por_trader / vol_total
            shares_sorted = shares.sort_values(ascending=False)
            
            # HHI (0-10000)
            hhi = (shares ** 2).sum() * 10000
            
            # Gini
            n = len(shares)
            sorted_shares = np.sort(shares.values)
            cumsum = np.cumsum(sorted_shares)
            gini = 1 - 2 * np.sum(cumsum) / (n * vol_total / vol_total) + 1/n
            # Corrección del Gini
            gini = (np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_shares)) / (n * np.mean(sorted_shares))
            gini = max(0, min(1, gini / n if n > 0 else 0))
            # Recalcular Gini correctamente
            sorted_vals = np.sort(vol_por_trader.values)
            n = len(sorted_vals)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))
            
            # CR4 y CR10
            cr4 = shares_sorted.head(4).sum() * 100
            cr10 = shares_sorted.head(10).sum() * 100
            
            concentraciones.append({
                'fecha': fecha,
                'hhi': hhi,
                'gini': gini,
                'cr4': cr4,
                'cr10': cr10,
                'n_traders': len(vol_por_trader)
            })
        
        df_conc = pd.DataFrame(concentraciones)
        self.metricas_concentracion = df_conc
        
        # Promedios
        hhi_mean = df_conc['hhi'].mean()
        gini_mean = df_conc['gini'].mean()
        cr4_mean = df_conc['cr4'].mean()
        cr10_mean = df_conc['cr10'].mean()
        
        self.log(f"\n{'='*50}")
        self.log("METRICAS DE CONCENTRACION (promedio diario)")
        self.log(f"{'='*50}")
        self.log(f"  HHI:  {hhi_mean:,.0f}")
        self.log(f"  Gini: {gini_mean:.3f}")
        self.log(f"  CR4:  {cr4_mean:.1f}%")
        self.log(f"  CR10: {cr10_mean:.1f}%")
        
        # Interpretación HHI
        if hhi_mean < 1500:
            tipo_mercado = "COMPETITIVO"
        elif hhi_mean < 2500:
            tipo_mercado = "MODERADAMENTE CONCENTRADO"
        else:
            tipo_mercado = "ALTAMENTE CONCENTRADO"
        
        self.log(f"\n  -> Mercado {tipo_mercado} (HHI={hhi_mean:.0f})")
        
        # Guardar
        df_conc.to_csv(os.path.join(self.dir_datos, 'concentracion_diaria.csv'), index=False)
    
    # ========================================================================
    # FASE 5: SEGMENTACIÓN BINANCE
    # ========================================================================
    def fase5_segmentacion_binance(self):
        """Analiza segmentación Profession vs Mass."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Segmentacion Binance (Profession vs Mass)")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Agregar por segmento
        segmentos = df.groupby('adv.classify').agg({
            'montoDynamic': ['sum', 'mean', 'median', 'count'],
            'userNo': 'nunique',
            'precio_transaccion': 'mean'
        }).reset_index()
        
        segmentos.columns = ['segmento', 'volumen', 'ticket_mean', 'ticket_median', 
                             'n_trans', 'n_traders', 'precio_mean']
        
        vol_total = segmentos['volumen'].sum()
        segmentos['pct_volumen'] = segmentos['volumen'] / vol_total * 100
        
        self.segmentacion = segmentos
        
        self.log(f"\n{'Segmento':<15} {'Volumen':>15} {'% Vol':>8} {'Traders':>10} {'Ticket':>12}")
        self.log("-" * 65)
        
        for _, row in segmentos.iterrows():
            self.log(f"{row['segmento']:<15} {row['volumen']:>15,.0f} {row['pct_volumen']:>7.1f}% "
                    f"{int(row['n_traders']):>10,} ${row['ticket_mean']:>11,.0f}")
        
        # Guardar
        segmentos.to_csv(os.path.join(self.dir_tablas, 'segmentacion_binance.csv'), index=False)
    
    # ========================================================================
    # FASE 6: ANÁLISIS DE TRADERS
    # ========================================================================
    def fase6_analisis_traders(self):
        """Analiza distribución de volumen entre traders."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Analisis de Traders")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Volumen por trader
        vol_trader = df.groupby('userNo').agg({
            'montoDynamic': ['sum', 'mean', 'count'],
            'adv.classify': 'first'
        }).reset_index()
        
        vol_trader.columns = ['userNo', 'volumen', 'ticket_mean', 'n_trans', 'segmento']
        vol_trader = vol_trader.sort_values('volumen', ascending=False)
        
        vol_total = vol_trader['volumen'].sum()
        vol_trader['pct_volumen'] = vol_trader['volumen'] / vol_total * 100
        vol_trader['pct_acumulado'] = vol_trader['pct_volumen'].cumsum()
        
        self.stats_traders = vol_trader
        
        # Top 10
        self.log(f"\nTOP 10 TRADERS")
        self.log(f"{'#':<4} {'Volumen':>15} {'% Vol':>8} {'% Acum':>8} {'Trans':>8} {'Segmento':>12}")
        self.log("-" * 60)
        
        for i, (_, row) in enumerate(vol_trader.head(10).iterrows(), 1):
            self.log(f"{i:<4} {row['volumen']:>15,.0f} {row['pct_volumen']:>7.1f}% "
                    f"{row['pct_acumulado']:>7.1f}% {int(row['n_trans']):>8,} {row['segmento']:>12}")
        
        # Concentración por percentiles
        self.log(f"\nCONCENTRACION POR PERCENTILES")
        self.log("-" * 40)
        
        n_traders = len(vol_trader)
        for pct in [1, 5, 10, 20, 50]:
            n = max(1, int(n_traders * pct / 100))
            vol_top = vol_trader.head(n)['volumen'].sum()
            self.log(f"  Top {pct:>2}% ({n:>4} traders): {vol_top/vol_total*100:>6.1f}% del volumen")
        
        # Guardar
        vol_trader.to_csv(os.path.join(self.dir_datos, 'volumen_por_trader.csv'), index=False)
    
    # ========================================================================
    # FASE 7: CURVA DE LORENZ
    # ========================================================================
    def fase7_curva_lorenz(self):
        """Genera curva de Lorenz y calcula Gini global."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Curva de Lorenz")
        self.log("="*70)
        
        vol_trader = self.stats_traders.copy()
        
        # Ordenar de menor a mayor
        vol_sorted = vol_trader.sort_values('volumen')['volumen'].values
        n = len(vol_sorted)
        
        # Calcular Lorenz
        vol_cumsum = np.cumsum(vol_sorted)
        vol_total = vol_cumsum[-1]
        
        lorenz_x = np.arange(1, n + 1) / n
        lorenz_y = vol_cumsum / vol_total
        
        # Agregar punto (0,0)
        lorenz_x = np.insert(lorenz_x, 0, 0)
        lorenz_y = np.insert(lorenz_y, 0, 0)
        
        # Gini = 2 * área entre línea de igualdad y Lorenz
        area_bajo_lorenz = np.trapz(lorenz_y, lorenz_x)
        gini_global = 1 - 2 * area_bajo_lorenz
        
        self.log(f"  Gini global: {gini_global:.3f}")
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Igualdad perfecta')
        ax.fill_between(lorenz_x, lorenz_y, lorenz_x, alpha=0.3, color='red', 
                        label=f'Area de desigualdad (Gini={gini_global:.3f})')
        ax.plot(lorenz_x, lorenz_y, 'b-', linewidth=2, label='Curva de Lorenz')
        
        ax.set_xlabel('% Acumulado de Traders')
        ax.set_ylabel('% Acumulado de Volumen')
        ax.set_title('Curva de Lorenz - Concentracion del Mercado P2P USDT/BOB')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Añadir anotaciones
        ax.annotate(f'Top 1% = {self.stats_traders.head(int(len(self.stats_traders)*0.01)+1)["pct_volumen"].sum():.1f}%',
                    xy=(0.99, self.stats_traders.head(int(len(self.stats_traders)*0.01)+1)["pct_acumulado"].max()/100),
                    xytext=(0.7, 0.3), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '01_curva_lorenz.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.gini_global = gini_global
    
    # ========================================================================
    # FASE 8: VISUALIZACIONES
    # ========================================================================
    def fase8_visualizaciones(self):
        """Genera visualizaciones del EDA."""
        self.log("\n" + "="*70)
        self.log("FASE 8: Visualizaciones")
        self.log("="*70)
        
        self._grafico_serie_temporal()
        self._grafico_distribucion_tickets()
        self._grafico_concentracion()
        self._grafico_segmentacion()
        
        self.log("Graficos generados")
    
    def _grafico_serie_temporal(self):
        """Serie temporal de volumen diario."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        df = self.datos_diarios.copy()
        
        # Volumen
        axes[0].plot(df['fecha'], df['volumen'], color='steelblue', alpha=0.7, linewidth=1)
        axes[0].plot(df['fecha'], df['volumen'].rolling(7).mean(), color='darkblue', 
                     linewidth=2, label='MA(7)')
        axes[0].set_ylabel('Volumen (USDT)')
        axes[0].set_title('Volumen Diario del Mercado P2P USDT/BOB')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Ratio C/V
        axes[1].plot(df['fecha'], df['ratio_cv'], color='green', alpha=0.7, linewidth=1)
        axes[1].axhline(y=1, color='red', linestyle='--', linewidth=1, label='Equilibrio')
        axes[1].plot(df['fecha'], df['ratio_cv'].rolling(7).mean(), color='darkgreen', 
                     linewidth=2, label='MA(7)')
        axes[1].set_ylabel('Ratio Compra/Venta')
        axes[1].set_xlabel('Fecha')
        axes[1].set_title('Ratio Compra/Venta Diario')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '02_serie_temporal.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_distribucion_tickets(self):
        """Distribución de tickets."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        tickets = self.transacciones['montoDynamic'].values
        
        # Histograma (log scale)
        axes[0].hist(tickets, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
        axes[0].set_xlabel('Monto (USDT)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribucion de Tickets')
        axes[0].set_yscale('log')
        axes[0].axvline(x=np.mean(tickets), color='red', linestyle='--', 
                        label=f'Media: ${np.mean(tickets):,.0f}')
        axes[0].axvline(x=np.median(tickets), color='green', linestyle='--', 
                        label=f'Mediana: ${np.median(tickets):,.0f}')
        axes[0].legend()
        
        # Boxplot por segmento
        df = self.transacciones.copy()
        df_sample = df.sample(min(10000, len(df)))  # Muestra para velocidad
        
        sns.boxplot(data=df_sample, x='adv.classify', y='montoDynamic', ax=axes[1])
        axes[1].set_xlabel('Segmento')
        axes[1].set_ylabel('Monto (USDT)')
        axes[1].set_title('Distribucion de Tickets por Segmento')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_distribucion_tickets.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_concentracion(self):
        """Serie temporal de concentración."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df = self.metricas_concentracion.copy()
        
        # HHI
        axes[0, 0].plot(df['fecha'], df['hhi'], color='purple', alpha=0.7)
        axes[0, 0].axhline(y=1500, color='orange', linestyle='--', label='Moderado')
        axes[0, 0].axhline(y=2500, color='red', linestyle='--', label='Concentrado')
        axes[0, 0].set_ylabel('HHI')
        axes[0, 0].set_title('Indice Herfindahl-Hirschman')
        axes[0, 0].legend()
        
        # Gini
        axes[0, 1].plot(df['fecha'], df['gini'], color='green', alpha=0.7)
        axes[0, 1].set_ylabel('Gini')
        axes[0, 1].set_title('Coeficiente de Gini')
        
        # CR4
        axes[1, 0].plot(df['fecha'], df['cr4'], color='blue', alpha=0.7)
        axes[1, 0].set_ylabel('CR4 (%)')
        axes[1, 0].set_xlabel('Fecha')
        axes[1, 0].set_title('Concentracion Top 4 Traders')
        
        # N traders
        axes[1, 1].plot(df['fecha'], df['n_traders'], color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('N Traders')
        axes[1, 1].set_xlabel('Fecha')
        axes[1, 1].set_title('Numero de Traders Activos por Dia')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '04_concentracion_temporal.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_segmentacion(self):
        """Gráfico de segmentación."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        seg = self.segmentacion.copy()
        
        # Volumen
        colors = ['steelblue', 'darkorange']
        axes[0].bar(seg['segmento'], seg['volumen'] / 1e6, color=colors)
        axes[0].set_ylabel('Volumen (Millones USDT)')
        axes[0].set_title('Volumen por Segmento')
        
        for i, row in seg.iterrows():
            axes[0].text(i, row['volumen']/1e6, f"{row['pct_volumen']:.1f}%", 
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Traders
        axes[1].bar(seg['segmento'], seg['n_traders'], color=colors)
        axes[1].set_ylabel('Numero de Traders')
        axes[1].set_title('Traders por Segmento')
        
        n_total = seg['n_traders'].sum()
        for i, row in seg.iterrows():
            axes[1].text(i, row['n_traders'], f"{row['n_traders']/n_total*100:.1f}%", 
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '05_segmentacion.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # FASE 9: REPORTE
    # ========================================================================
    def fase9_reporte(self):
        """Genera reporte final."""
        self.log("\n" + "="*70)
        self.log("FASE 9: Reporte Final")
        self.log("="*70)
        
        reporte = []
        reporte.append("="*70)
        reporte.append("EDA - ESTADISTICAS DESCRIPTIVAS Y CONCENTRACION")
        reporte.append("="*70)
        reporte.append(f"\nFecha de analisis: {datetime.now()}")
        
        reporte.append("\n" + "="*70)
        reporte.append("RESUMEN EJECUTIVO")
        reporte.append("="*70)
        
        s = self.stats_generales
        reporte.append(f"\n  Periodo: {s['fecha_inicio'].date()} a {s['fecha_fin'].date()}")
        reporte.append(f"  Dias: {s['dias']}")
        reporte.append(f"  Transacciones: {s['n_transacciones']:,}")
        reporte.append(f"  Volumen total: {s['volumen_total']:,.0f} USDT")
        reporte.append(f"  Traders: {s['n_traders']:,}")
        
        reporte.append(f"\n  Ratio Compra/Venta: {s['ratio_cv']:.3f}")
        reporte.append(f"  Ticket promedio: ${s['ticket_promedio']:,.2f}")
        
        reporte.append("\n" + "="*70)
        reporte.append("CONCENTRACION DE MERCADO")
        reporte.append("="*70)
        
        c = self.metricas_concentracion
        reporte.append(f"\n  HHI promedio: {c['hhi'].mean():,.0f}")
        reporte.append(f"  Gini promedio: {c['gini'].mean():.3f}")
        reporte.append(f"  CR4 promedio: {c['cr4'].mean():.1f}%")
        reporte.append(f"  CR10 promedio: {c['cr10'].mean():.1f}%")
        reporte.append(f"  Gini global: {self.gini_global:.3f}")
        
        reporte.append("\n" + "="*70)
        reporte.append("ARCHIVOS GENERADOS")
        reporte.append("="*70)
        reporte.append(f"  Graficos: {self.dir_graficos}")
        reporte.append(f"  Datos: {self.dir_datos}")
        reporte.append(f"  Tablas: {self.dir_tablas}")
        
        # Guardar
        reporte_path = os.path.join(self.directorio_salida, 'eda_estadisticas_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
            f.write('\n\n')
            f.write('\n'.join(reporte))
        
        self.log(f"\nReporte guardado: {reporte_path}")
    
    # ========================================================================
    # EJECUTAR
    # ========================================================================
    def ejecutar(self):
        """Ejecuta análisis completo."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        self.log("="*70)
        self.log("EDA - ESTADISTICAS Y CONCENTRACION")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_estadisticas_generales()
            self.fase3_agregacion_diaria()
            self.fase4_concentracion_mercado()
            self.fase5_segmentacion_binance()
            self.fase6_analisis_traders()
            self.fase7_curva_lorenz()
            self.fase8_visualizaciones()
            self.fase9_reporte()
            
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
    print("EDA - ESTADISTICAS Y CONCENTRACION DE MERCADO")
    print("="*70)
    print("\nSelecciona el archivo CSV de Binance P2P...")
    
    ruta = filedialog.askopenfilename(
        title='Seleccionar CSV',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta:
        print("Cancelado.")
        return
    
    eda = EDAEstadisticas(ruta)
    eda.ejecutar()


if __name__ == "__main__":
    main()

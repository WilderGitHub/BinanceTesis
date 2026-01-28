"""
================================================================================
EDA_VISUALIZACIONES.PY - Visualizaciones Exploratorias
================================================================================
Genera gráficos exploratorios del mercado P2P:
- Heatmap hora x día de semana
- Patrones temporales (hora, día, mes)
- Distribuciones de precios y tickets
- Análisis de spread y liquidez

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


class EDAVisualizaciones:
    """Genera visualizaciones exploratorias del mercado P2P."""
    
    def __init__(self, ruta_csv, directorio_salida=None):
        self.ruta_csv = ruta_csv
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_csv)
        
        self.dir_graficos = os.path.join(self.directorio_salida, 'eda_visualizaciones')
        os.makedirs(self.dir_graficos, exist_ok=True)
        
        self.transacciones = None
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
        
        # Preparar variables temporales
        df = self.transacciones
        df['hora'] = df['time'].dt.hour
        df['dia_semana'] = df['time'].dt.dayofweek
        df['dia_nombre'] = df['time'].dt.day_name()
        df['mes'] = df['time'].dt.month
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        
        self.log(f"Transacciones cargadas: {len(df):,}")
    
    # ========================================================================
    # FASE 2: HEATMAP HORA x DÍA
    # ========================================================================
    def fase2_heatmap_hora_dia(self):
        """Genera heatmap de actividad por hora y día de semana."""
        self.log("\n" + "="*70)
        self.log("FASE 2: Heatmap Hora x Dia de Semana")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Pivot para volumen
        pivot_vol = df.pivot_table(
            values='montoDynamic', 
            index='hora', 
            columns='dia_semana',
            aggfunc='sum'
        )
        
        # Pivot para transacciones
        pivot_trans = df.pivot_table(
            values='montoDynamic', 
            index='hora', 
            columns='dia_semana',
            aggfunc='count'
        )
        
        # Nombres de días
        dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
        pivot_vol.columns = dias
        pivot_trans.columns = dias
        
        # Gráfico
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # Heatmap volumen
        sns.heatmap(pivot_vol / 1e6, annot=False, cmap='YlOrRd', ax=axes[0],
                    cbar_kws={'label': 'Volumen (M USDT)'})
        axes[0].set_title('Volumen por Hora y Dia de Semana', fontsize=14)
        axes[0].set_xlabel('Dia de Semana')
        axes[0].set_ylabel('Hora del Dia')
        
        # Heatmap transacciones
        sns.heatmap(pivot_trans, annot=False, cmap='YlGnBu', ax=axes[1],
                    cbar_kws={'label': 'N Transacciones'})
        axes[1].set_title('Transacciones por Hora y Dia de Semana', fontsize=14)
        axes[1].set_xlabel('Dia de Semana')
        axes[1].set_ylabel('Hora del Dia')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '01_heatmap_hora_dia.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("Heatmap generado")
    
    # ========================================================================
    # FASE 3: PATRONES POR HORA
    # ========================================================================
    def fase3_patrones_hora(self):
        """Analiza patrones por hora del día."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Patrones por Hora del Dia")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Agregar por hora
        por_hora = df.groupby('hora').agg({
            'montoDynamic': ['sum', 'mean', 'count'],
            'precio_transaccion': 'mean',
            'userNo': 'nunique'
        }).reset_index()
        
        por_hora.columns = ['hora', 'volumen', 'ticket_mean', 'n_trans', 'precio_mean', 'n_traders']
        
        # Gráfico
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Volumen por hora
        axes[0, 0].bar(por_hora['hora'], por_hora['volumen'] / 1e6, color='steelblue', alpha=0.8)
        axes[0, 0].set_xlabel('Hora del Dia')
        axes[0, 0].set_ylabel('Volumen (Millones USDT)')
        axes[0, 0].set_title('Volumen por Hora')
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Transacciones por hora
        axes[0, 1].bar(por_hora['hora'], por_hora['n_trans'], color='darkorange', alpha=0.8)
        axes[0, 1].set_xlabel('Hora del Dia')
        axes[0, 1].set_ylabel('N Transacciones')
        axes[0, 1].set_title('Transacciones por Hora')
        axes[0, 1].set_xticks(range(0, 24, 2))
        
        # Ticket promedio por hora
        axes[1, 0].plot(por_hora['hora'], por_hora['ticket_mean'], 'go-', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Hora del Dia')
        axes[1, 0].set_ylabel('Ticket Promedio (USDT)')
        axes[1, 0].set_title('Ticket Promedio por Hora')
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].axhline(y=por_hora['ticket_mean'].mean(), color='red', linestyle='--', 
                          label=f'Media: ${por_hora["ticket_mean"].mean():,.0f}')
        axes[1, 0].legend()
        
        # Traders activos por hora
        axes[1, 1].bar(por_hora['hora'], por_hora['n_traders'], color='purple', alpha=0.8)
        axes[1, 1].set_xlabel('Hora del Dia')
        axes[1, 1].set_ylabel('Traders Unicos')
        axes[1, 1].set_title('Traders Activos por Hora')
        axes[1, 1].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '02_patrones_hora.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Mostrar horas pico
        hora_pico_vol = por_hora.loc[por_hora['volumen'].idxmax(), 'hora']
        hora_min_vol = por_hora.loc[por_hora['volumen'].idxmin(), 'hora']
        
        self.log(f"  Hora pico volumen: {hora_pico_vol}:00")
        self.log(f"  Hora minimo volumen: {hora_min_vol}:00")
    
    # ========================================================================
    # FASE 4: PATRONES POR DÍA DE SEMANA
    # ========================================================================
    def fase4_patrones_dia_semana(self):
        """Analiza patrones por día de semana."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Patrones por Dia de Semana")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Agregar por día
        por_dia = df.groupby(['dia_semana', 'dia_nombre']).agg({
            'montoDynamic': ['sum', 'mean', 'count'],
            'precio_transaccion': 'mean'
        }).reset_index()
        
        por_dia.columns = ['dia_semana', 'dia_nombre', 'volumen', 'ticket_mean', 'n_trans', 'precio_mean']
        por_dia = por_dia.sort_values('dia_semana')
        
        # Gráfico
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_esp = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        
        colors = ['steelblue'] * 5 + ['darkorange'] * 2  # FDS en naranja
        
        # Volumen
        axes[0, 0].bar(dias_esp, por_dia['volumen'] / 1e6, color=colors)
        axes[0, 0].set_ylabel('Volumen (Millones USDT)')
        axes[0, 0].set_title('Volumen por Dia de Semana')
        axes[0, 0].axhline(y=por_dia['volumen'].mean() / 1e6, color='red', linestyle='--')
        
        # Variación vs media
        media_vol = por_dia['volumen'].mean()
        variacion = (por_dia['volumen'] / media_vol - 1) * 100
        
        colors_var = ['green' if v > 0 else 'red' for v in variacion]
        axes[0, 1].bar(dias_esp, variacion, color=colors_var)
        axes[0, 1].set_ylabel('Variacion vs Media (%)')
        axes[0, 1].set_title('Variacion del Volumen vs Promedio')
        axes[0, 1].axhline(y=0, color='black', linewidth=1)
        
        for i, v in enumerate(variacion):
            axes[0, 1].text(i, v, f'{v:+.1f}%', ha='center', 
                          va='bottom' if v > 0 else 'top', fontsize=10)
        
        # Ticket promedio
        axes[1, 0].bar(dias_esp, por_dia['ticket_mean'], color=colors)
        axes[1, 0].set_ylabel('Ticket Promedio (USDT)')
        axes[1, 0].set_title('Ticket Promedio por Dia')
        axes[1, 0].axhline(y=por_dia['ticket_mean'].mean(), color='red', linestyle='--')
        
        # Transacciones
        axes[1, 1].bar(dias_esp, por_dia['n_trans'], color=colors)
        axes[1, 1].set_ylabel('N Transacciones')
        axes[1, 1].set_title('Transacciones por Dia')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_patrones_dia_semana.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Mostrar efecto FDS
        vol_es = por_dia[por_dia['dia_semana'] < 5]['volumen'].mean()
        vol_fds = por_dia[por_dia['dia_semana'] >= 5]['volumen'].mean()
        efecto_fds = (vol_fds / vol_es - 1) * 100
        
        self.log(f"  Volumen entre semana: {vol_es/1e6:,.1f}M USDT")
        self.log(f"  Volumen fin de semana: {vol_fds/1e6:,.1f}M USDT")
        self.log(f"  Efecto FDS: {efecto_fds:+.1f}%")
    
    # ========================================================================
    # FASE 5: PATRONES POR MES
    # ========================================================================
    def fase5_patrones_mes(self):
        """Analiza patrones por mes."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Patrones por Mes")
        self.log("="*70)
        
        df = self.transacciones.copy()
        df['año_mes'] = df['time'].dt.to_period('M')
        
        # Agregar por mes
        por_mes = df.groupby('año_mes').agg({
            'montoDynamic': ['sum', 'mean', 'count'],
            'userNo': 'nunique'
        }).reset_index()
        
        por_mes.columns = ['año_mes', 'volumen', 'ticket_mean', 'n_trans', 'n_traders']
        por_mes['año_mes_str'] = por_mes['año_mes'].astype(str)
        
        # Gráfico
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = range(len(por_mes))
        
        # Volumen
        axes[0, 0].bar(x, por_mes['volumen'] / 1e6, color='steelblue')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(por_mes['año_mes_str'], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Volumen (Millones USDT)')
        axes[0, 0].set_title('Volumen Mensual')
        
        # Transacciones
        axes[0, 1].bar(x, por_mes['n_trans'], color='darkorange')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(por_mes['año_mes_str'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('N Transacciones')
        axes[0, 1].set_title('Transacciones Mensuales')
        
        # Ticket
        axes[1, 0].plot(x, por_mes['ticket_mean'], 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(por_mes['año_mes_str'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Ticket Promedio (USDT)')
        axes[1, 0].set_title('Ticket Promedio Mensual')
        
        # Traders
        axes[1, 1].bar(x, por_mes['n_traders'], color='purple')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(por_mes['año_mes_str'], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Traders Unicos')
        axes[1, 1].set_title('Traders Activos por Mes')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '04_patrones_mes.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"  Meses analizados: {len(por_mes)}")
    
    # ========================================================================
    # FASE 6: DISTRIBUCIÓN DE PRECIOS
    # ========================================================================
    def fase6_distribucion_precios(self):
        """Analiza distribución de precios."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Distribucion de Precios")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histograma de precios
        axes[0, 0].hist(df['precio_transaccion'], bins=100, color='steelblue', 
                        alpha=0.7, edgecolor='white')
        axes[0, 0].axvline(x=df['precio_transaccion'].mean(), color='red', linestyle='--',
                          label=f'Media: {df["precio_transaccion"].mean():.2f}')
        axes[0, 0].axvline(x=df['precio_transaccion'].median(), color='green', linestyle='--',
                          label=f'Mediana: {df["precio_transaccion"].median():.2f}')
        axes[0, 0].set_xlabel('Precio (BOB/USDT)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].set_title('Distribucion de Precios de Transaccion')
        axes[0, 0].legend()
        
        # Precio por tipo de operación
        df_compra = df[df['adv.tradeType'] == 'compra']['precio_transaccion']
        df_venta = df[df['adv.tradeType'] == 'venta']['precio_transaccion']
        
        axes[0, 1].hist(df_compra, bins=50, alpha=0.5, label='Compra', color='green')
        axes[0, 1].hist(df_venta, bins=50, alpha=0.5, label='Venta', color='red')
        axes[0, 1].set_xlabel('Precio (BOB/USDT)')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Precios de Compra vs Venta')
        axes[0, 1].legend()
        
        # Serie temporal de precios
        precio_diario = df.groupby('fecha')['precio_transaccion'].mean()
        axes[1, 0].plot(precio_diario.index, precio_diario.values, color='steelblue', linewidth=1)
        axes[1, 0].plot(precio_diario.index, precio_diario.rolling(7).mean(), 
                        color='red', linewidth=2, label='MA(7)')
        axes[1, 0].set_xlabel('Fecha')
        axes[1, 0].set_ylabel('Precio (BOB/USDT)')
        axes[1, 0].set_title('Evolucion del Precio Promedio Diario')
        axes[1, 0].legend()
        
        # Spread compra-venta
        precio_compra_diario = df[df['adv.tradeType'] == 'compra'].groupby('fecha')['precio_transaccion'].mean()
        precio_venta_diario = df[df['adv.tradeType'] == 'venta'].groupby('fecha')['precio_transaccion'].mean()
        
        spread = precio_compra_diario - precio_venta_diario
        spread = spread.dropna()
        
        axes[1, 1].plot(spread.index, spread.values, color='purple', linewidth=1, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].axhline(y=spread.mean(), color='red', linestyle='--', 
                          label=f'Media: {spread.mean():.2f}')
        axes[1, 1].set_xlabel('Fecha')
        axes[1, 1].set_ylabel('Spread (BOB)')
        axes[1, 1].set_title('Spread Compra - Venta')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '05_distribucion_precios.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"  Precio promedio: {df['precio_transaccion'].mean():.2f} BOB/USDT")
        self.log(f"  Spread promedio: {spread.mean():.2f} BOB")
    
    # ========================================================================
    # FASE 7: ANÁLISIS DE LIQUIDEZ
    # ========================================================================
    def fase7_analisis_liquidez(self):
        """Analiza métricas de liquidez."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Analisis de Liquidez")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Liquidez por hora
        liquidez_hora = df.groupby(['fecha', 'hora']).agg({
            'montoDynamic': ['sum', 'count'],
            'userNo': 'nunique'
        }).reset_index()
        
        liquidez_hora.columns = ['fecha', 'hora', 'volumen', 'n_trans', 'n_traders']
        
        # Métricas por hora promedio
        liq_hora_prom = liquidez_hora.groupby('hora').agg({
            'volumen': 'mean',
            'n_trans': 'mean',
            'n_traders': 'mean'
        })
        
        # Velocidad de rotación (transacciones por hora por trader)
        liq_hora_prom['velocidad'] = liq_hora_prom['n_trans'] / liq_hora_prom['n_traders']
        
        # Gráfico
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Volumen por hora
        axes[0, 0].bar(liq_hora_prom.index, liq_hora_prom['volumen'] / 1e3, color='steelblue')
        axes[0, 0].set_xlabel('Hora')
        axes[0, 0].set_ylabel('Volumen Promedio (Miles USDT)')
        axes[0, 0].set_title('Liquidez por Hora (Volumen)')
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Transacciones por hora
        axes[0, 1].bar(liq_hora_prom.index, liq_hora_prom['n_trans'], color='darkorange')
        axes[0, 1].set_xlabel('Hora')
        axes[0, 1].set_ylabel('Transacciones Promedio')
        axes[0, 1].set_title('Liquidez por Hora (Transacciones)')
        axes[0, 1].set_xticks(range(0, 24, 2))
        
        # Velocidad de rotación
        axes[1, 0].plot(liq_hora_prom.index, liq_hora_prom['velocidad'], 'go-', 
                        linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Hora')
        axes[1, 0].set_ylabel('Trans/Trader/Hora')
        axes[1, 0].set_title('Velocidad de Rotacion por Hora')
        axes[1, 0].set_xticks(range(0, 24, 2))
        
        # Distribución de tiempo entre transacciones
        df_sorted = df.sort_values('time')
        df_sorted['tiempo_entre_trans'] = df_sorted['time'].diff().dt.total_seconds() / 60  # minutos
        tiempo_entre = df_sorted['tiempo_entre_trans'].dropna()
        tiempo_entre = tiempo_entre[tiempo_entre < 60]  # Solo < 60 min
        
        axes[1, 1].hist(tiempo_entre, bins=60, color='purple', alpha=0.7, edgecolor='white')
        axes[1, 1].set_xlabel('Minutos entre Transacciones')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Tiempo entre Transacciones Consecutivas')
        axes[1, 1].axvline(x=tiempo_entre.median(), color='red', linestyle='--',
                          label=f'Mediana: {tiempo_entre.median():.1f} min')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '06_analisis_liquidez.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"  Tiempo mediano entre transacciones: {tiempo_entre.median():.1f} minutos")
    
    # ========================================================================
    # FASE 8: REPORTE
    # ========================================================================
    def fase8_reporte(self):
        """Genera reporte final."""
        self.log("\n" + "="*70)
        self.log("FASE 8: Reporte Final")
        self.log("="*70)
        
        reporte_path = os.path.join(self.directorio_salida, 'eda_visualizaciones_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
        
        self.log(f"\nReporte guardado: {reporte_path}")
        self.log(f"Graficos guardados en: {self.dir_graficos}")
    
    # ========================================================================
    # EJECUTAR
    # ========================================================================
    def ejecutar(self):
        """Ejecuta análisis completo."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        self.log("="*70)
        self.log("EDA - VISUALIZACIONES EXPLORATORIAS")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_heatmap_hora_dia()
            self.fase3_patrones_hora()
            self.fase4_patrones_dia_semana()
            self.fase5_patrones_mes()
            self.fase6_distribucion_precios()
            self.fase7_analisis_liquidez()
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
    print("EDA - VISUALIZACIONES EXPLORATORIAS")
    print("="*70)
    print("\nSelecciona el archivo CSV de Binance P2P...")
    
    ruta = filedialog.askopenfilename(
        title='Seleccionar CSV',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta:
        print("Cancelado.")
        return
    
    eda = EDAVisualizaciones(ruta)
    eda.ejecutar()


if __name__ == "__main__":
    main()

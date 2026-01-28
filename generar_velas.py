"""
================================================================================
GENERAR_VELAS.PY - Generación de Velas OHLCV
================================================================================
Genera velas OHLCV (Open, High, Low, Close, Volume) a partir de transacciones P2P.

Intervalos soportados:
- 1min, 5min, 15min, 30min
- 1h, 4h
- 1d, 1w

Incluye:
- Velas separadas por tipo (compra/venta)
- Métricas adicionales (n_trans, n_traders, spread)
- Detección de gaps
- Visualización básica

Autor: Para tesis de maestría en finanzas
Fecha: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import gc
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

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

# Mapeo de intervalos a reglas de pandas
INTERVALOS = {
    '1min': '1T',
    '5min': '5T',
    '15min': '15T',
    '30min': '30T',
    '1h': '1H',
    '4h': '4H',
    '1d': '1D',
    '1w': '1W'
}


class GeneradorVelas:
    """Genera velas OHLCV desde transacciones P2P."""
    
    def __init__(self, ruta_csv, intervalo='1h', directorio_salida=None):
        self.ruta_csv = ruta_csv
        self.intervalo = intervalo
        
        if intervalo not in INTERVALOS:
            raise ValueError(f"Intervalo '{intervalo}' no soportado. Use: {list(INTERVALOS.keys())}")
        
        self.freq = INTERVALOS[intervalo]
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_csv)
        
        self.dir_velas = os.path.join(self.directorio_salida, 'velas')
        os.makedirs(self.dir_velas, exist_ok=True)
        
        self.transacciones = None
        self.velas = None
        self.velas_compra = None
        self.velas_venta = None
        self.reporte = []
    
    def log(self, mensaje):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {mensaje}")
        self.reporte.append(f"[{ts}] {mensaje}")
    
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
        
        self.log(f"Transacciones cargadas: {len(self.transacciones):,}")
    
    def fase2_generar_velas_globales(self):
        """Genera velas OHLCV globales."""
        self.log("\n" + "="*70)
        self.log(f"FASE 2: Generando velas globales (intervalo: {self.intervalo})")
        self.log("="*70)
        
        df = self.transacciones.copy()
        df = df.set_index('time').sort_index()
        
        velas = df.resample(self.freq).agg({
            'precio_transaccion': ['first', 'max', 'min', 'last', 'mean', 'std'],
            'montoDynamic': ['sum', 'mean', 'count'],
            'userNo': 'nunique'
        })
        
        velas.columns = [
            'open', 'high', 'low', 'close', 'precio_mean', 'precio_std',
            'volume', 'ticket_mean', 'n_trans', 'n_traders'
        ]
        
        velas = velas.dropna(subset=['open'])
        
        velas['rango'] = velas['high'] - velas['low']
        velas['rango_pct'] = velas['rango'] / velas['open'] * 100
        velas['cambio'] = velas['close'] - velas['open']
        velas['cambio_pct'] = velas['cambio'] / velas['open'] * 100
        velas['es_verde'] = (velas['close'] >= velas['open']).astype(int)
        velas['volatilidad'] = velas['precio_std'] / velas['precio_mean'] * 100
        velas['volatilidad'] = velas['volatilidad'].fillna(0)
        
        self.velas = velas.reset_index()
        
        self.log(f"Velas generadas: {len(velas):,}")
    
    def fase3_generar_velas_por_tipo(self):
        """Genera velas separadas para compra y venta."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Generando velas por tipo")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Compra
        df_compra = df[df['adv.tradeType'] == 'compra'].set_index('time').sort_index()
        velas_compra = df_compra.resample(self.freq).agg({
            'precio_transaccion': ['first', 'max', 'min', 'last'],
            'montoDynamic': ['sum', 'count']
        })
        velas_compra.columns = ['open', 'high', 'low', 'close', 'volume', 'n_trans']
        self.velas_compra = velas_compra.dropna(subset=['open']).reset_index()
        self.log(f"Velas COMPRA: {len(self.velas_compra):,}")
        
        # Venta
        df_venta = df[df['adv.tradeType'] == 'venta'].set_index('time').sort_index()
        velas_venta = df_venta.resample(self.freq).agg({
            'precio_transaccion': ['first', 'max', 'min', 'last'],
            'montoDynamic': ['sum', 'count']
        })
        velas_venta.columns = ['open', 'high', 'low', 'close', 'volume', 'n_trans']
        self.velas_venta = velas_venta.dropna(subset=['open']).reset_index()
        self.log(f"Velas VENTA: {len(self.velas_venta):,}")
    
    def fase4_calcular_spread(self):
        """Calcula spread entre compra y venta."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Calculando spread")
        self.log("="*70)
        
        compra = self.velas_compra[['time', 'close']].rename(columns={'close': 'precio_compra'})
        venta = self.velas_venta[['time', 'close']].rename(columns={'close': 'precio_venta'})
        
        spread = compra.merge(venta, on='time', how='outer')
        spread['spread'] = spread['precio_compra'] - spread['precio_venta']
        spread['spread_pct'] = spread['spread'] / spread['precio_venta'] * 100
        
        self.velas = self.velas.merge(spread[['time', 'spread', 'spread_pct']], on='time', how='left')
        
        self.log(f"Spread promedio: {spread['spread'].mean():.2f} BOB")
    
    def fase5_guardar_velas(self):
        """Guarda velas en CSV."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Guardando velas")
        self.log("="*70)
        
        self.velas.to_csv(os.path.join(self.dir_velas, f'velas_{self.intervalo}.csv'), index=False)
        self.velas_compra.to_csv(os.path.join(self.dir_velas, f'velas_{self.intervalo}_compra.csv'), index=False)
        self.velas_venta.to_csv(os.path.join(self.dir_velas, f'velas_{self.intervalo}_venta.csv'), index=False)
        
        self.log(f"Archivos guardados en: {self.dir_velas}")
    
    def fase6_visualizacion(self):
        """Genera gráfico de velas."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Generando visualizacion")
        self.log("="*70)
        
        velas = self.velas.tail(200) if len(self.velas) > 200 else self.velas
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        axes[0].plot(velas['time'], velas['close'], color='steelblue', linewidth=1)
        axes[0].fill_between(velas['time'], velas['low'], velas['high'], alpha=0.2, color='steelblue')
        axes[0].set_ylabel('Precio (BOB/USDT)')
        axes[0].set_title(f'Precio USDT/BOB - Velas {self.intervalo}')
        
        colors = ['green' if v else 'red' for v in velas['es_verde']]
        axes[1].bar(velas['time'], velas['volume'] / 1e3, color=colors, alpha=0.7)
        axes[1].set_ylabel('Volumen (Miles USDT)')
        
        if 'spread' in velas.columns:
            axes[2].plot(velas['time'], velas['spread'], color='purple', linewidth=1)
            axes[2].axhline(y=0, color='black', linestyle='-')
            axes[2].set_ylabel('Spread (BOB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_velas, f'velas_{self.intervalo}_grafico.png'), dpi=150)
        plt.close()
        
        self.log("Grafico generado")
    
    def fase7_reporte(self):
        """Genera reporte."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Resumen")
        self.log("="*70)
        
        v = self.velas
        self.log(f"\n  Velas: {len(v):,}")
        self.log(f"  Volumen total: {v['volume'].sum():,.0f} USDT")
        self.log(f"  Precio promedio: {v['close'].mean():.2f} BOB/USDT")
        self.log(f"  Velas verdes: {v['es_verde'].mean()*100:.1f}%")
        
        reporte_path = os.path.join(self.dir_velas, f'velas_{self.intervalo}_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
    
    def ejecutar(self):
        """Ejecuta generación completa."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        
        try:
            self.fase1_cargar_datos()
            self.fase2_generar_velas_globales()
            self.fase3_generar_velas_por_tipo()
            self.fase4_calcular_spread()
            self.fase5_guardar_velas()
            self.fase6_visualizacion()
            self.fase7_reporte()
            
            self.log(f"\nCOMPLETADO EN: {datetime.now() - inicio}")
            return self.velas
            
        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise


def main():
    from tkinter import Tk, filedialog, simpledialog
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("="*70)
    print("GENERADOR DE VELAS OHLCV")
    print("="*70)
    print("\nSelecciona el archivo CSV...")
    
    ruta = filedialog.askopenfilename(
        title='Seleccionar CSV',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta:
        print("Cancelado.")
        return
    
    # Seleccionar intervalo
    print("\nIntervalos disponibles: 1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w")
    intervalo = simpledialog.askstring("Intervalo", "Ingresa el intervalo (default: 1h):", initialvalue="1h")
    
    if not intervalo:
        intervalo = '1h'
    
    generador = GeneradorVelas(ruta, intervalo=intervalo)
    generador.ejecutar()


if __name__ == "__main__":
    main()

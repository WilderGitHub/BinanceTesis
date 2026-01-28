"""
================================================================================
CLUSTERING_ESTRATIFICADO_V2.PY - Taxonomía de Agentes Mejorada
================================================================================
Version mejorada con:
- Ticket promedio como feature
- Features adicionales de comportamiento
- Mejor caracterización de clusters
- Análisis de estabilidad temporal
- Exportación completa para análisis posterior

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

# Configuración de gráficos
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

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
    from sklearn.decomposition import PCA
except ImportError:
    print("ERROR: scikit-learn no instalado")
    print("Ejecutar: pip install scikit-learn")
    sys.exit(1)

CHUNK_SIZE = 500_000


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class ClusteringEstrategificadoV2:
    """Taxonomía de agentes del mercado P2P usando K-Means estratificado."""
    
    def __init__(self, ruta_csv, directorio_salida=None):
        self.ruta_csv = ruta_csv
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_csv)
        
        # Subdirectorios
        self.dir_graficos = os.path.join(self.directorio_salida, 'clustering_graficos')
        self.dir_datos = os.path.join(self.directorio_salida, 'clustering_datos')
        self.dir_tablas = os.path.join(self.directorio_salida, 'clustering_tablas')
        os.makedirs(self.dir_graficos, exist_ok=True)
        os.makedirs(self.dir_datos, exist_ok=True)
        os.makedirs(self.dir_tablas, exist_ok=True)
        
        # Datos
        self.transacciones = None
        self.features_traders = None
        self.clusters_profession = None
        self.clusters_mass = None
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
        self.log(f"Traders unicos: {self.transacciones['userNo'].nunique():,}")
    
    # ========================================================================
    # FASE 2: CALCULAR FEATURES POR TRADER
    # ========================================================================
    def fase2_calcular_features(self):
        """Calcula features completas por trader."""
        self.log("\n" + "="*70)
        self.log("FASE 2: Calculo de features por trader")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Preparar variables temporales
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        df['hora'] = df['time'].dt.hour
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['es_fds'] = (df['dia_semana'] >= 5).astype(int)
        df['es_compra'] = (df['adv.tradeType'] == 'compra').astype(int)
        df['es_venta'] = (df['adv.tradeType'] == 'venta').astype(int)
        
        # Rango de fechas del dataset
        fecha_min = df['fecha'].min()
        fecha_max = df['fecha'].max()
        dias_totales = (fecha_max - fecha_min).days + 1
        
        self.log(f"Periodo: {fecha_min.date()} a {fecha_max.date()} ({dias_totales} dias)")
        
        # ================================================================
        # AGREGAR POR TRADER
        # ================================================================
        self.log("Agregando features por trader...")
        
        features = df.groupby('userNo').agg({
            # Volumen
            'montoDynamic': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            # Precio
            'precio_transaccion': ['mean', 'std'],
            # Temporales
            'fecha': ['min', 'max', 'nunique'],
            'hora': ['mean', 'std'],
            'dia_semana': 'mean',
            # Tipo de transacción
            'es_compra': 'sum',
            'es_venta': 'sum',
            'es_fds': 'mean',
            # Segmento Binance
            'adv.classify': 'first'
        }).reset_index()
        
        # Aplanar columnas
        features.columns = [
            'userNo',
            'volumen_total', 'volumen_mean', 'volumen_std', 'volumen_min', 'volumen_max', 'n_transacciones',
            'precio_mean', 'precio_std',
            'fecha_primera', 'fecha_ultima', 'dias_activo',
            'hora_mean', 'hora_std',
            'dia_semana_mean',
            'n_compras', 'n_ventas',
            'pct_fds',
            'segmento'
        ]
        
        # ================================================================
        # FEATURES DERIVADAS
        # ================================================================
        self.log("Calculando features derivadas...")
        
        # TICKET PROMEDIO (NUEVA FEATURE IMPORTANTE)
        features['ticket_promedio'] = features['volumen_total'] / features['n_transacciones']
        features['ticket_mediano'] = df.groupby('userNo')['montoDynamic'].median().values
        features['ticket_std'] = features['volumen_std']  # Ya calculado
        features['ticket_cv'] = features['ticket_std'] / features['ticket_promedio']  # Coef. de variación
        
        # Distribución del ticket (percentiles)
        ticket_p25 = df.groupby('userNo')['montoDynamic'].quantile(0.25)
        ticket_p75 = df.groupby('userNo')['montoDynamic'].quantile(0.75)
        features['ticket_p25'] = features['userNo'].map(ticket_p25)
        features['ticket_p75'] = features['userNo'].map(ticket_p75)
        features['ticket_iqr'] = features['ticket_p75'] - features['ticket_p25']
        
        # Frecuencia
        features['frecuencia_diaria'] = features['n_transacciones'] / features['dias_activo']
        features['dias_vida'] = (features['fecha_ultima'] - features['fecha_primera']).dt.days + 1
        features['pct_dias_activo'] = features['dias_activo'] / features['dias_vida']
        features['pct_dias_mercado'] = features['dias_activo'] / dias_totales
        
        # Ratio compra/venta
        features['ratio_cv'] = features['n_compras'] / features['n_ventas'].replace(0, np.nan)
        features['pct_compras'] = features['n_compras'] / features['n_transacciones']
        features['es_solo_compra'] = (features['n_ventas'] == 0).astype(int)
        features['es_solo_venta'] = (features['n_compras'] == 0).astype(int)
        features['es_bidireccional'] = ((features['n_compras'] > 0) & (features['n_ventas'] > 0)).astype(int)
        
        # Equilibrio compra/venta (1 = perfectamente equilibrado)
        features['balance_cv'] = 1 - abs(features['pct_compras'] - 0.5) * 2
        
        # Volatilidad del precio
        features['precio_cv'] = features['precio_std'] / features['precio_mean']
        features['precio_cv'] = features['precio_cv'].fillna(0)
        
        # Concentración temporal (Gini de días)
        def calcular_concentracion_temporal(group):
            dias_count = group.groupby('fecha').size()
            if len(dias_count) <= 1:
                return 1.0
            total = dias_count.sum()
            n = len(dias_count)
            sorted_counts = np.sort(dias_count.values)
            cumsum = np.cumsum(sorted_counts)
            gini = 1 - 2 * np.sum(cumsum) / (n * total) + 1/n
            return gini
        
        self.log("  Calculando concentracion temporal...")
        conc_temporal = df.groupby('userNo').apply(calcular_concentracion_temporal)
        features['concentracion_temporal'] = features['userNo'].map(conc_temporal)
        
        # Patrón horario (variabilidad)
        features['hora_cv'] = features['hora_std'] / features['hora_mean'].replace(0, 1)
        features['es_horario_fijo'] = (features['hora_std'] < 3).astype(int)  # Opera en rango de 3 horas
        
        # Patrón fin de semana
        features['prefiere_fds'] = (features['pct_fds'] > 0.35).astype(int)  # >35% en FDS
        features['evita_fds'] = (features['pct_fds'] < 0.15).astype(int)  # <15% en FDS
        
        # Volumen por hora
        features['volumen_por_hora'] = features['volumen_total'] / (features['dias_activo'] * 24)
        
        # Categorías de tamaño
        features['log_volumen'] = np.log1p(features['volumen_total'])
        features['log_ticket'] = np.log1p(features['ticket_promedio'])
        features['log_transacciones'] = np.log1p(features['n_transacciones'])
        
        # ================================================================
        # FEATURES DE COMPORTAMIENTO AVANZADAS
        # ================================================================
        self.log("  Calculando features de comportamiento avanzadas...")
        
        # Tendencia de actividad (creciente vs decreciente)
        def calcular_tendencia_actividad(group):
            daily_vol = group.groupby('fecha')['montoDynamic'].sum()
            if len(daily_vol) < 3:
                return 0
            x = np.arange(len(daily_vol))
            try:
                slope, _, r, _, _ = stats.linregress(x, daily_vol.values)
                return slope / daily_vol.mean() if daily_vol.mean() > 0 else 0
            except:
                return 0
        
        tendencia = df.groupby('userNo').apply(calcular_tendencia_actividad)
        features['tendencia_actividad'] = features['userNo'].map(tendencia)
        features['es_creciente'] = (features['tendencia_actividad'] > 0.01).astype(int)
        features['es_decreciente'] = (features['tendencia_actividad'] < -0.01).astype(int)
        
        # Regularidad (desviación estándar de transacciones diarias)
        def calcular_regularidad(group):
            daily_count = group.groupby('fecha').size()
            if len(daily_count) < 3:
                return 0
            return daily_count.std() / daily_count.mean() if daily_count.mean() > 0 else 0
        
        regularidad = df.groupby('userNo').apply(calcular_regularidad)
        features['irregularidad'] = features['userNo'].map(regularidad)
        features['es_regular'] = (features['irregularidad'] < 0.5).astype(int)
        
        # Limpiar NaN e infinitos
        features = features.replace([np.inf, -np.inf], np.nan)
        for col in features.select_dtypes(include=[np.number]).columns:
            features[col] = features[col].fillna(features[col].median())
        
        self.features_traders = features
        self.log(f"\nFeatures calculadas: {len(features.columns)} columnas")
        self.log(f"Traders: {len(features):,}")
        
        # Guardar
        features.to_csv(os.path.join(self.dir_datos, 'features_traders_completo.csv'), index=False)
    
    # ========================================================================
    # FASE 3: SELECCIONAR FEATURES PARA CLUSTERING
    # ========================================================================
    def fase3_seleccionar_features(self):
        """Selecciona y prepara features para clustering."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Seleccion de features para clustering")
        self.log("="*70)
        
        # Features para clustering (normalizadas y relevantes)
        self.features_clustering = [
            # Volumen y tamaño
            'log_volumen',
            'log_ticket',           # NUEVA - Ticket promedio
            'log_transacciones',
            
            # Frecuencia y actividad
            'frecuencia_diaria',
            'pct_dias_activo',
            
            # Comportamiento compra/venta
            'pct_compras',
            'balance_cv',           # NUEVA - Equilibrio compra/venta
            
            # Patrones temporales
            'pct_fds',
            'hora_mean',
            'concentracion_temporal',
            
            # Variabilidad
            'ticket_cv',            # NUEVA - Variabilidad del ticket
            'irregularidad',        # NUEVA - Regularidad de operaciones
        ]
        
        self.log(f"Features seleccionadas: {len(self.features_clustering)}")
        for f in self.features_clustering:
            self.log(f"  - {f}")
    
    # ========================================================================
    # FASE 4: CLUSTERING POR SEGMENTO
    # ========================================================================
    def fase4_clustering(self):
        """Ejecuta K-Means estratificado por segmento Binance."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Clustering estratificado")
        self.log("="*70)
        
        # Separar por segmento
        profession = self.features_traders[self.features_traders['segmento'] == 'profession'].copy()
        mass = self.features_traders[self.features_traders['segmento'] == 'mass'].copy()
        
        self.log(f"\nSegmento PROFESSION: {len(profession):,} traders")
        self.log(f"Segmento MASS: {len(mass):,} traders")
        
        # ================================================================
        # CLUSTERING PROFESSION
        # ================================================================
        self.log("\n--- Clustering PROFESSION ---")
        self.clusters_profession = self._ejecutar_clustering(
            profession, 
            'profession',
            k_range=(2, 6)
        )
        
        # ================================================================
        # CLUSTERING MASS
        # ================================================================
        self.log("\n--- Clustering MASS ---")
        self.clusters_mass = self._ejecutar_clustering(
            mass, 
            'mass',
            k_range=(2, 8)
        )
        
        # ================================================================
        # COMBINAR RESULTADOS
        # ================================================================
        self.log("\n--- Combinando resultados ---")
        
        # Asignar tipo final
        profession_result = self.clusters_profession['datos'].copy()
        profession_result['tipo_final'] = 'P' + (profession_result['cluster'] + 1).astype(str)
        
        mass_result = self.clusters_mass['datos'].copy()
        mass_result['tipo_final'] = 'M' + (mass_result['cluster'] + 1).astype(str)
        
        # Combinar
        self.features_traders = pd.concat([profession_result, mass_result], ignore_index=True)
        
        self.log(f"Taxonomia final: {self.features_traders['tipo_final'].nunique()} tipos")
        
        # Guardar
        self.features_traders.to_csv(
            os.path.join(self.dir_datos, 'traders_con_clusters.csv'), 
            index=False
        )
    
    def _ejecutar_clustering(self, df, nombre, k_range=(2, 6)):
        """Ejecuta clustering para un segmento."""
        
        # Preparar datos
        X = df[self.features_clustering].copy()
        
        # Escalar con RobustScaler (menos sensible a outliers)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encontrar K óptimo
        self.log(f"  Buscando K optimo en rango {k_range}...")
        
        silhouettes = []
        calinski = []
        inertias = []
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = kmeans.fit_predict(X_scaled)
            
            sil = silhouette_score(X_scaled, labels)
            cal = calinski_harabasz_score(X_scaled, labels)
            
            silhouettes.append(sil)
            calinski.append(cal)
            inertias.append(kmeans.inertia_)
            
            self.log(f"    K={k}: Silhouette={sil:.3f}, Calinski={cal:.0f}")
        
        # Seleccionar mejor K (máximo silhouette)
        k_optimo = k_range[0] + np.argmax(silhouettes)
        mejor_silhouette = max(silhouettes)
        
        self.log(f"  K optimo: {k_optimo} (Silhouette: {mejor_silhouette:.3f})")
        
        # Ejecutar clustering final
        kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=20, max_iter=500)
        df['cluster'] = kmeans_final.fit_predict(X_scaled)
        
        # Calcular silhouette por muestra
        df['silhouette'] = silhouette_samples(X_scaled, df['cluster'])
        
        # Estadísticas por cluster
        self.log(f"\n  Distribucion de clusters {nombre}:")
        for c in range(k_optimo):
            mask = df['cluster'] == c
            n = mask.sum()
            vol = df.loc[mask, 'volumen_total'].sum()
            vol_pct = vol / df['volumen_total'].sum() * 100
            ticket = df.loc[mask, 'ticket_promedio'].mean()
            self.log(f"    Cluster {c}: {n:,} traders ({n/len(df)*100:.1f}%), "
                    f"Vol: {vol_pct:.1f}%, Ticket: ${ticket:,.0f}")
        
        return {
            'datos': df,
            'k_optimo': k_optimo,
            'silhouette': mejor_silhouette,
            'silhouettes': silhouettes,
            'calinski': calinski,
            'inertias': inertias,
            'scaler': scaler,
            'kmeans': kmeans_final
        }
    
    # ========================================================================
    # FASE 5: CARACTERIZACIÓN DE CLUSTERS
    # ========================================================================
    def fase5_caracterizar_clusters(self):
        """Caracteriza cada cluster con estadísticas detalladas."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Caracterizacion de clusters")
        self.log("="*70)
        
        df = self.features_traders.copy()
        
        # Variables de interés para caracterización
        vars_caracterizacion = [
            'volumen_total', 'ticket_promedio', 'n_transacciones',
            'dias_activo', 'frecuencia_diaria', 'pct_compras',
            'balance_cv', 'pct_fds', 'concentracion_temporal'
        ]
        
        # Estadísticas por tipo
        resumen = df.groupby('tipo_final').agg({
            'userNo': 'count',
            'volumen_total': ['sum', 'mean', 'median'],
            'ticket_promedio': ['mean', 'median', 'std'],
            'n_transacciones': ['sum', 'mean', 'median'],
            'dias_activo': ['mean', 'median'],
            'frecuencia_diaria': ['mean', 'median'],
            'pct_compras': 'mean',
            'balance_cv': 'mean',
            'pct_fds': 'mean',
            'concentracion_temporal': 'mean',
            'silhouette': 'mean'
        }).round(2)
        
        resumen.columns = ['_'.join(col).strip() for col in resumen.columns.values]
        resumen = resumen.reset_index()
        
        # Calcular participación de mercado
        vol_total = df['volumen_total'].sum()
        resumen['pct_volumen_mercado'] = resumen['volumen_total_sum'] / vol_total * 100
        
        # Ordenar por volumen
        resumen = resumen.sort_values('volumen_total_sum', ascending=False)
        
        self.log("\n" + "="*70)
        self.log("RESUMEN DE TAXONOMIA")
        self.log("="*70)
        
        self.log(f"\n{'Tipo':<6} {'Traders':>8} {'% Vol':>8} {'Ticket Prom':>12} {'Trans/dia':>10} {'% Compras':>10} {'% FDS':>8}")
        self.log("-" * 75)
        
        for _, row in resumen.iterrows():
            self.log(f"{row['tipo_final']:<6} "
                    f"{int(row['userNo_count']):>8,} "
                    f"{row['pct_volumen_mercado']:>7.1f}% "
                    f"${row['ticket_promedio_mean']:>11,.0f} "
                    f"{row['frecuencia_diaria_mean']:>10.2f} "
                    f"{row['pct_compras_mean']*100:>9.1f}% "
                    f"{row['pct_fds_mean']*100:>7.1f}%")
        
        # Guardar
        resumen.to_csv(os.path.join(self.dir_tablas, 'resumen_taxonomia.csv'), index=False)
        self.resultados['resumen_taxonomia'] = resumen
        
        # ================================================================
        # INTERPRETACIÓN DE TIPOS
        # ================================================================
        self.log("\n" + "="*70)
        self.log("INTERPRETACION DE TIPOS DE AGENTES")
        self.log("="*70)
        
        for _, row in resumen.iterrows():
            tipo = row['tipo_final']
            self.log(f"\n{tipo}:")
            
            # Características principales
            caracteristicas = []
            
            # Volumen
            if row['pct_volumen_mercado'] > 50:
                caracteristicas.append("DOMINANTE en volumen")
            elif row['pct_volumen_mercado'] > 10:
                caracteristicas.append("Alto volumen")
            elif row['pct_volumen_mercado'] < 1:
                caracteristicas.append("Bajo volumen")
            
            # Ticket
            ticket_global = df['ticket_promedio'].median()
            if row['ticket_promedio_mean'] > ticket_global * 2:
                caracteristicas.append("Tickets grandes")
            elif row['ticket_promedio_mean'] < ticket_global * 0.5:
                caracteristicas.append("Tickets pequenos")
            
            # Frecuencia
            if row['frecuencia_diaria_mean'] > 5:
                caracteristicas.append("Alta frecuencia")
            elif row['frecuencia_diaria_mean'] < 1:
                caracteristicas.append("Ocasional")
            
            # Compra/venta
            if row['pct_compras_mean'] > 0.7:
                caracteristicas.append("Principalmente comprador")
            elif row['pct_compras_mean'] < 0.3:
                caracteristicas.append("Principalmente vendedor")
            elif 0.4 <= row['pct_compras_mean'] <= 0.6:
                caracteristicas.append("Equilibrado C/V (posible arbitrajista)")
            
            # FDS
            if row['pct_fds_mean'] > 0.35:
                caracteristicas.append("Activo en FDS")
            elif row['pct_fds_mean'] < 0.2:
                caracteristicas.append("Evita FDS")
            
            for c in caracteristicas:
                self.log(f"  - {c}")
    
    # ========================================================================
    # FASE 6: VISUALIZACIONES
    # ========================================================================
    def fase6_visualizaciones(self):
        """Genera visualizaciones del clustering."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Visualizaciones")
        self.log("="*70)
        
        self._grafico_taxonomia()
        self._grafico_pca()
        self._grafico_radar()
        self._grafico_ticket_volumen()
        self._grafico_silhouette()
        
        self.log("Visualizaciones generadas")
    
    def _grafico_taxonomia(self):
        """Gráfico de distribución de la taxonomía."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        df = self.features_traders.copy()
        resumen = self.resultados['resumen_taxonomia']
        
        # Por número de traders
        tipos_orden = resumen.sort_values('userNo_count', ascending=True)['tipo_final']
        counts = df['tipo_final'].value_counts().reindex(tipos_orden)
        
        colors = ['steelblue' if t.startswith('P') else 'darkorange' for t in tipos_orden]
        
        axes[0].barh(tipos_orden, counts, color=colors)
        axes[0].set_xlabel('Numero de Traders')
        axes[0].set_title('Distribucion por Numero de Traders')
        
        # Por volumen
        vol_por_tipo = df.groupby('tipo_final')['volumen_total'].sum().reindex(tipos_orden)
        vol_pct = vol_por_tipo / vol_por_tipo.sum() * 100
        
        axes[1].barh(tipos_orden, vol_pct, color=colors)
        axes[1].set_xlabel('% del Volumen Total')
        axes[1].set_title('Distribucion por Volumen')
        
        for i, (t, v) in enumerate(zip(tipos_orden, vol_pct)):
            axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '01_taxonomia_distribucion.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_pca(self):
        """Visualización PCA de los clusters."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Separar por segmento
        for idx, (nombre, resultado) in enumerate([
            ('Profession', self.clusters_profession),
            ('Mass', self.clusters_mass)
        ]):
            df = resultado['datos']
            X = df[self.features_clustering]
            X_scaled = resultado['scaler'].transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], 
                                        c=df['cluster'], cmap='viridis', 
                                        alpha=0.6, s=30)
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            axes[idx].set_title(f'Clusters {nombre} (K={resultado["k_optimo"]}, Sil={resultado["silhouette"]:.3f})')
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '02_clusters_pca.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_radar(self):
        """Gráfico radar comparativo de tipos."""
        resumen = self.resultados['resumen_taxonomia']
        
        # Seleccionar top tipos por volumen
        top_tipos = resumen.head(6)['tipo_final'].tolist()
        
        # Variables para radar (normalizadas)
        vars_radar = ['ticket_promedio_mean', 'frecuencia_diaria_mean', 
                      'pct_compras_mean', 'pct_fds_mean', 'balance_cv_mean']
        labels = ['Ticket', 'Frecuencia', '% Compras', '% FDS', 'Balance C/V']
        
        # Normalizar 0-1
        radar_data = resumen[resumen['tipo_final'].isin(top_tipos)][['tipo_final'] + vars_radar].copy()
        for var in vars_radar:
            radar_data[var] = (radar_data[var] - radar_data[var].min()) / (radar_data[var].max() - radar_data[var].min() + 0.001)
        
        # Crear radar
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_tipos)))
        
        for i, (_, row) in enumerate(radar_data.iterrows()):
            values = row[vars_radar].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['tipo_final'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title('Perfil Comparativo de Tipos de Agentes')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_radar_tipos.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_ticket_volumen(self):
        """Scatter plot de ticket vs volumen por tipo."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        df = self.features_traders.copy()
        
        tipos = df['tipo_final'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(tipos)))
        
        for tipo, color in zip(tipos, colors):
            mask = df['tipo_final'] == tipo
            ax.scatter(df.loc[mask, 'log_volumen'], 
                      df.loc[mask, 'log_ticket'],
                      label=tipo, alpha=0.5, s=30, color=color)
        
        ax.set_xlabel('Log(Volumen Total)')
        ax.set_ylabel('Log(Ticket Promedio)')
        ax.set_title('Relacion Volumen vs Ticket por Tipo de Agente')
        ax.legend(title='Tipo', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '04_ticket_vs_volumen.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_silhouette(self):
        """Gráfico de validación del clustering."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Silhouette por K - Profession
        k_range_p = range(2, 2 + len(self.clusters_profession['silhouettes']))
        axes[0, 0].plot(k_range_p, self.clusters_profession['silhouettes'], 'bo-', linewidth=2)
        axes[0, 0].axvline(x=self.clusters_profession['k_optimo'], color='red', linestyle='--')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Seleccion de K - Profession')
        
        # Silhouette por K - Mass
        k_range_m = range(2, 2 + len(self.clusters_mass['silhouettes']))
        axes[0, 1].plot(k_range_m, self.clusters_mass['silhouettes'], 'go-', linewidth=2)
        axes[0, 1].axvline(x=self.clusters_mass['k_optimo'], color='red', linestyle='--')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Seleccion de K - Mass')
        
        # Distribución silhouette por tipo
        df = self.features_traders.copy()
        tipos_orden = df.groupby('tipo_final')['silhouette'].median().sort_values().index
        
        df_plot = df[['tipo_final', 'silhouette']].copy()
        df_plot['tipo_final'] = pd.Categorical(df_plot['tipo_final'], categories=tipos_orden, ordered=True)
        
        df_plot.boxplot(column='silhouette', by='tipo_final', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Tipo')
        axes[1, 0].set_ylabel('Silhouette')
        axes[1, 0].set_title('Silhouette por Tipo de Agente')
        plt.suptitle('')
        
        # Elbow plot
        axes[1, 1].plot(k_range_p, self.clusters_profession['inertias'], 'bo-', label='Profession')
        axes[1, 1].plot(k_range_m, self.clusters_mass['inertias'], 'go-', label='Mass')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('Inertia')
        axes[1, 1].set_title('Metodo del Codo')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '05_validacion_clustering.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # FASE 7: REPORTE
    # ========================================================================
    def fase7_reporte(self):
        """Genera reporte final."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Reporte Final")
        self.log("="*70)
        
        reporte = []
        reporte.append("="*70)
        reporte.append("TAXONOMIA DE AGENTES - CLUSTERING ESTRATIFICADO V2")
        reporte.append("="*70)
        reporte.append(f"\nFecha: {datetime.now()}")
        reporte.append(f"Traders totales: {len(self.features_traders):,}")
        
        reporte.append("\n" + "="*70)
        reporte.append("FEATURES UTILIZADAS PARA CLUSTERING")
        reporte.append("="*70)
        for f in self.features_clustering:
            reporte.append(f"  - {f}")
        
        reporte.append("\n" + "="*70)
        reporte.append("RESULTADOS POR SEGMENTO")
        reporte.append("="*70)
        reporte.append(f"\nPROFESSION:")
        reporte.append(f"  K optimo: {self.clusters_profession['k_optimo']}")
        reporte.append(f"  Silhouette: {self.clusters_profession['silhouette']:.3f}")
        
        reporte.append(f"\nMASS:")
        reporte.append(f"  K optimo: {self.clusters_mass['k_optimo']}")
        reporte.append(f"  Silhouette: {self.clusters_mass['silhouette']:.3f}")
        
        reporte.append("\n" + "="*70)
        reporte.append("ARCHIVOS GENERADOS")
        reporte.append("="*70)
        reporte.append(f"  Datos: {self.dir_datos}")
        reporte.append(f"  Tablas: {self.dir_tablas}")
        reporte.append(f"  Graficos: {self.dir_graficos}")
        
        # Guardar
        reporte_path = os.path.join(self.directorio_salida, 'clustering_reporte.txt')
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
        self.log("CLUSTERING ESTRATIFICADO V2 - TAXONOMIA DE AGENTES")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_calcular_features()
            self.fase3_seleccionar_features()
            self.fase4_clustering()
            self.fase5_caracterizar_clusters()
            self.fase6_visualizaciones()
            self.fase7_reporte()
            
            self.log(f"\n{'='*70}")
            self.log(f"COMPLETADO EN: {datetime.now() - inicio}")
            self.log(f"{'='*70}")
            
            return self.features_traders
            
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
    print("CLUSTERING ESTRATIFICADO V2 - TAXONOMIA DE AGENTES")
    print("="*70)
    print("\nSelecciona el archivo CSV de Binance P2P...")
    
    ruta = filedialog.askopenfilename(
        title='Seleccionar CSV',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta:
        print("Cancelado.")
        return
    
    clustering = ClusteringEstrategificadoV2(ruta)
    clustering.ejecutar()


if __name__ == "__main__":
    main()

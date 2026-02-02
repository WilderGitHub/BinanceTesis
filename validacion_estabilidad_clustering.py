"""
================================================================================
VALIDACION_ESTABILIDAD_CLUSTERING.PY
================================================================================
Validación de estabilidad de la taxonomía de agentes mediante:
1. Bootstrap Resampling (ARI, Jaccard, Consistencia)
2. Sensibilidad a Outliers (remoción progresiva)
3. Validación por Ventanas Temporales
4. Análisis de Frontera (traders cerca de límites de clusters)

Genera evidencia para argumentar robustez ante Silhouette moderado.

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
import warnings
from datetime import datetime
from collections import Counter

# Sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class ValidacionEstabilidadClustering:
    """Validación de estabilidad de la taxonomía mediante múltiples métodos."""
    
    def __init__(self, ruta_features, ruta_clusters, directorio_salida=None):
        """
        Args:
            ruta_features: CSV con features por trader (features_traders_completo.csv)
            ruta_clusters: CSV con asignaciones de clusters (traders_con_clusters.csv)
            directorio_salida: Directorio para outputs
        """
        self.ruta_features = ruta_features
        self.ruta_clusters = ruta_clusters
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_features)
        
        # Subdirectorios
        self.dir_graficos = os.path.join(self.directorio_salida, 'validacion_graficos')
        self.dir_datos = os.path.join(self.directorio_salida, 'validacion_datos')
        self.dir_tablas = os.path.join(self.directorio_salida, 'validacion_tablas')
        
        for d in [self.dir_graficos, self.dir_datos, self.dir_tablas]:
            os.makedirs(d, exist_ok=True)
        
        # Datos
        self.features = None
        self.clusters = None
        self.features_cols = None
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
        """Carga features y clusters originales."""
        self.log("="*70)
        self.log("FASE 1: Carga de datos")
        self.log("="*70)
        
        # Cargar features
        self.features = pd.read_csv(self.ruta_features)
        self.log(f"Features cargados: {len(self.features)} traders")
        
        # Cargar clusters
        self.clusters = pd.read_csv(self.ruta_clusters)
        self.log(f"Clusters cargados: {len(self.clusters)} traders")
        
        # Identificar columna de tipo
        for col in ['tipo_final', 'cluster', 'tipo']:
            if col in self.clusters.columns:
                self.tipo_col = col
                break
        else:
            raise ValueError("No se encontró columna de tipo/cluster")
        
        self.log(f"Columna de tipo: {self.tipo_col}")
        
        # Identificar columna de segmento (puede tener diferentes nombres)
        self.segmento_col = None
        for col in ['adv.classify', 'segmento', 'segment', 'tipo_binance']:
            if col in self.clusters.columns:
                self.segmento_col = col
                break
            elif col in self.features.columns:
                self.segmento_col = col
                break
        
        if self.segmento_col is None:
            # Si no hay columna de segmento, verificar si tipo_final tiene prefijos P/M
            tipos = self.clusters[self.tipo_col].unique()
            if any(str(t).startswith('P') for t in tipos) and any(str(t).startswith('M') for t in tipos):
                self.log("Inferiendo segmento desde tipo_final (P=profession, M=mass)")
                self.clusters['_segmento_inferido'] = self.clusters[self.tipo_col].apply(
                    lambda x: 'profession' if str(x).startswith('P') else 'mass'
                )
                self.segmento_col = '_segmento_inferido'
            else:
                self.log("ADVERTENCIA: No se encontró columna de segmento. Usando 'todos'")
                self.clusters['_segmento_todos'] = 'todos'
                self.segmento_col = '_segmento_todos'
        
        self.log(f"Columna de segmento: {self.segmento_col}")
        
        # Columnas a usar del archivo de clusters
        cols_clusters = ['userNo', self.tipo_col]
        if self.segmento_col in self.clusters.columns:
            cols_clusters.append(self.segmento_col)
        
        # Merge
        self.datos = self.features.merge(
            self.clusters[cols_clusters], 
            on='userNo', 
            how='inner'
        )
        
        # Si el segmento está en features pero no en clusters
        if self.segmento_col in self.features.columns and self.segmento_col not in self.datos.columns:
            pass  # Ya está en datos desde features
        self.log(f"Datos combinados: {len(self.datos)} traders")
        
        # Identificar features para clustering
        exclude_cols = ['userNo', 'adv.classify', 'segmento', self.tipo_col, 'cluster_profession', 
                       'cluster_mass', 'tipo_binance', 'volumen_total', 'n_transacciones',
                       '_segmento_inferido', '_segmento_todos']
        
        self.features_cols = [c for c in self.features.columns 
                             if c not in exclude_cols and self.features[c].dtype in ['float64', 'int64']]
        
        # Filtrar features con varianza
        self.features_cols = [c for c in self.features_cols 
                             if self.features[c].std() > 0]
        
        self.log(f"Features para clustering: {len(self.features_cols)}")
        self.log(f"  {self.features_cols[:5]}...")
    
    # ========================================================================
    # FASE 2: BOOTSTRAP RESAMPLING
    # ========================================================================
    def fase2_bootstrap_resampling(self, n_iterations=100):
        """Evalúa estabilidad mediante bootstrap."""
        self.log("\n" + "="*70)
        self.log(f"FASE 2: Bootstrap Resampling ({n_iterations} iteraciones)")
        self.log("="*70)
        
        resultados_bootstrap = {}
        
        # Detectar segmentos disponibles
        if self.segmento_col in self.datos.columns:
            segmentos_disponibles = self.datos[self.segmento_col].unique().tolist()
            self.log(f"  Segmentos detectados: {segmentos_disponibles}")
        else:
            segmentos_disponibles = ['todos']
            self.log("  Sin columna de segmento, usando todos los datos")
        
        # Inicializar resultados para cada segmento
        for seg in segmentos_disponibles:
            resultados_bootstrap[seg] = {'ari': [], 'consistencia': []}
        
        # Configuración de K óptimo por segmento (puede ajustarse)
        k_por_segmento = {
            'profession': 3,
            'mass': 2,
            'todos': 3  # default
        }
        
        for segmento in segmentos_disponibles:
            self.log(f"\n  Procesando segmento: {segmento.upper()}")
            
            # Filtrar datos del segmento
            if self.segmento_col in self.datos.columns:
                mask = self.datos[self.segmento_col] == segmento
            else:
                mask = pd.Series([True] * len(self.datos))
            
            datos_seg = self.datos[mask].copy()
            
            if len(datos_seg) < 50:
                self.log(f"    Muy pocos datos ({len(datos_seg)}), saltando...")
                continue
            
            # Preparar features
            X = datos_seg[self.features_cols].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            
            # Clustering original
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            k = k_por_segmento.get(segmento, 3)  # default a 3 si no está definido
            kmeans_orig = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_orig = kmeans_orig.fit_predict(X_scaled)
            
            # Bootstrap
            n_traders = len(datos_seg)
            
            for i in range(n_iterations):
                if (i + 1) % 20 == 0:
                    self.log(f"    Iteración {i+1}/{n_iterations}...")
                
                # Remuestreo con reemplazo
                indices = np.random.choice(n_traders, size=n_traders, replace=True)
                X_boot = X_scaled[indices]
                labels_orig_boot = labels_orig[indices]
                
                # Re-clustering
                kmeans_boot = KMeans(n_clusters=k, random_state=i, n_init=10)
                labels_boot = kmeans_boot.fit_predict(X_boot)
                
                # Calcular ARI
                ari = adjusted_rand_score(labels_orig_boot, labels_boot)
                resultados_bootstrap[segmento]['ari'].append(ari)
                
                # Calcular consistencia (% que mantiene asignación)
                # Necesitamos mapear clusters para comparar
                consistencia = self._calcular_consistencia(labels_orig_boot, labels_boot, k)
                resultados_bootstrap[segmento]['consistencia'].append(consistencia)
        
        # Resumen
        self.log("\n  RESULTADOS BOOTSTRAP:")
        self.log("  " + "-"*50)
        
        resumen_bootstrap = []
        
        for segmento in resultados_bootstrap.keys():
            if resultados_bootstrap[segmento]['ari']:
                ari_mean = np.mean(resultados_bootstrap[segmento]['ari'])
                ari_std = np.std(resultados_bootstrap[segmento]['ari'])
                cons_mean = np.mean(resultados_bootstrap[segmento]['consistencia'])
                cons_std = np.std(resultados_bootstrap[segmento]['consistencia'])
                
                self.log(f"  {segmento.upper()}:")
                self.log(f"    ARI: {ari_mean:.3f} ± {ari_std:.3f}")
                self.log(f"    Consistencia: {cons_mean:.1%} ± {cons_std:.1%}")
                
                resumen_bootstrap.append({
                    'segmento': segmento,
                    'ari_mean': ari_mean,
                    'ari_std': ari_std,
                    'consistencia_mean': cons_mean,
                    'consistencia_std': cons_std,
                    'n_iterations': n_iterations
                })
        
        self.resultados['bootstrap'] = resultados_bootstrap
        self.resultados['resumen_bootstrap'] = pd.DataFrame(resumen_bootstrap)
        
        # Guardar
        self.resultados['resumen_bootstrap'].to_csv(
            os.path.join(self.dir_tablas, 'bootstrap_resumen.csv'), index=False
        )
    
    def _calcular_consistencia(self, labels_orig, labels_new, k):
        """Calcula consistencia de asignaciones con mapeo óptimo."""
        from scipy.optimize import linear_sum_assignment
        
        # Matriz de confusión
        confusion = np.zeros((k, k))
        for lo, ln in zip(labels_orig, labels_new):
            confusion[lo, ln] += 1
        
        # Mapeo óptimo (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(-confusion)
        
        # Consistencia = suma de diagonal mapeada / total
        total = confusion.sum()
        if total == 0:
            return 0
        
        consistencia = sum(confusion[row_ind[i], col_ind[i]] for i in range(k)) / total
        return consistencia
    
    # ========================================================================
    # FASE 3: ANÁLISIS JACCARD POR CLUSTER
    # ========================================================================
    def fase3_jaccard_por_cluster(self, n_iterations=50):
        """Calcula estabilidad Jaccard por cada cluster."""
        self.log("\n" + "="*70)
        self.log(f"FASE 3: Estabilidad Jaccard por Cluster ({n_iterations} iteraciones)")
        self.log("="*70)
        
        resultados_jaccard = []
        
        # Detectar segmentos disponibles
        if self.segmento_col in self.datos.columns:
            segmentos_disponibles = self.datos[self.segmento_col].unique().tolist()
        else:
            segmentos_disponibles = ['todos']
        
        k_por_segmento = {'profession': 3, 'mass': 2, 'todos': 3}
        
        for segmento in segmentos_disponibles:
            if self.segmento_col in self.datos.columns:
                mask = self.datos[self.segmento_col] == segmento
            else:
                mask = pd.Series([True] * len(self.datos))
            datos_seg = self.datos[mask].copy()
            
            if len(datos_seg) < 50:
                continue
            
            # Preparar
            X = datos_seg[self.features_cols].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            k = k_por_segmento.get(segmento, 3)
            
            # Clustering original
            kmeans_orig = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_orig = kmeans_orig.fit_predict(X_scaled)
            
            # Jaccard por cluster
            jaccard_por_cluster = {c: [] for c in range(k)}
            
            for i in range(n_iterations):
                # Submuestra (80% sin reemplazo)
                n_sample = int(len(datos_seg) * 0.8)
                indices = np.random.choice(len(datos_seg), size=n_sample, replace=False)
                
                X_sub = X_scaled[indices]
                labels_orig_sub = labels_orig[indices]
                
                # Re-clustering
                kmeans_sub = KMeans(n_clusters=k, random_state=i, n_init=10)
                labels_sub = kmeans_sub.fit_predict(X_sub)
                
                # Mapeo óptimo
                mapping = self._obtener_mapeo_optimo(labels_orig_sub, labels_sub, k)
                labels_sub_mapped = np.array([mapping.get(l, l) for l in labels_sub])
                
                # Jaccard por cluster
                for c in range(k):
                    set_orig = set(np.where(labels_orig_sub == c)[0])
                    set_new = set(np.where(labels_sub_mapped == c)[0])
                    
                    if len(set_orig) == 0 and len(set_new) == 0:
                        jaccard = 1.0
                    elif len(set_orig) == 0 or len(set_new) == 0:
                        jaccard = 0.0
                    else:
                        intersect = len(set_orig & set_new)
                        union = len(set_orig | set_new)
                        jaccard = intersect / union if union > 0 else 0
                    
                    jaccard_por_cluster[c].append(jaccard)
            
            # Resumen
            for c in range(k):
                if jaccard_por_cluster[c]:
                    tipo_nombre = f"P{c+1}" if segmento == 'profession' else f"M{c+1}"
                    resultados_jaccard.append({
                        'segmento': segmento,
                        'cluster': c,
                        'tipo': tipo_nombre,
                        'jaccard_mean': np.mean(jaccard_por_cluster[c]),
                        'jaccard_std': np.std(jaccard_por_cluster[c]),
                        'jaccard_min': np.min(jaccard_por_cluster[c]),
                        'jaccard_max': np.max(jaccard_por_cluster[c])
                    })
        
        df_jaccard = pd.DataFrame(resultados_jaccard)
        self.resultados['jaccard'] = df_jaccard
        
        self.log("\n  ESTABILIDAD JACCARD POR TIPO:")
        self.log("  " + "-"*50)
        for _, row in df_jaccard.iterrows():
            self.log(f"  {row['tipo']}: {row['jaccard_mean']:.3f} ± {row['jaccard_std']:.3f} "
                    f"[{row['jaccard_min']:.3f} - {row['jaccard_max']:.3f}]")
        
        # Guardar
        df_jaccard.to_csv(os.path.join(self.dir_tablas, 'jaccard_por_tipo.csv'), index=False)
    
    def _obtener_mapeo_optimo(self, labels_orig, labels_new, k):
        """Obtiene mapeo óptimo entre clusters."""
        from scipy.optimize import linear_sum_assignment
        
        confusion = np.zeros((k, k))
        for lo, ln in zip(labels_orig, labels_new):
            if lo < k and ln < k:
                confusion[lo, ln] += 1
        
        row_ind, col_ind = linear_sum_assignment(-confusion)
        return {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    
    # ========================================================================
    # FASE 4: SENSIBILIDAD A OUTLIERS
    # ========================================================================
    def fase4_sensibilidad_outliers(self):
        """Evalúa estabilidad removiendo outliers progresivamente."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Sensibilidad a Outliers")
        self.log("="*70)
        
        porcentajes_remover = [0, 1, 2, 5, 10]
        resultados_outliers = []
        
        # Detectar segmentos disponibles
        if self.segmento_col in self.datos.columns:
            segmentos_disponibles = self.datos[self.segmento_col].unique().tolist()
        else:
            segmentos_disponibles = ['todos']
        
        k_por_segmento = {'profession': 3, 'mass': 2, 'todos': 3}
        
        for segmento in segmentos_disponibles:
            if self.segmento_col in self.datos.columns:
                mask = self.datos[self.segmento_col] == segmento
            else:
                mask = pd.Series([True] * len(self.datos))
            datos_seg = self.datos[mask].copy()
            
            if len(datos_seg) < 50:
                continue
            
            X = datos_seg[self.features_cols].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            
            k = k_por_segmento.get(segmento, 3)
            
            # Clustering base (sin remover)
            scaler_base = RobustScaler()
            X_scaled_base = scaler_base.fit_transform(X)
            kmeans_base = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_base = kmeans_base.fit_predict(X_scaled_base)
            sil_base = silhouette_score(X_scaled_base, labels_base)
            
            for pct in porcentajes_remover:
                if pct == 0:
                    sil = sil_base
                    ari = 1.0
                    n_removed = 0
                else:
                    # Identificar outliers por distancia al centroide
                    distances = np.linalg.norm(X_scaled_base - kmeans_base.cluster_centers_[labels_base], axis=1)
                    threshold = np.percentile(distances, 100 - pct)
                    mask_keep = distances <= threshold
                    
                    X_clean = X_scaled_base[mask_keep]
                    labels_base_clean = labels_base[mask_keep]
                    n_removed = (~mask_keep).sum()
                    
                    if len(X_clean) < k * 5:
                        continue
                    
                    # Re-clustering
                    kmeans_clean = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_clean = kmeans_clean.fit_predict(X_clean)
                    
                    sil = silhouette_score(X_clean, labels_clean)
                    ari = adjusted_rand_score(labels_base_clean, labels_clean)
                
                resultados_outliers.append({
                    'segmento': segmento,
                    'pct_removido': pct,
                    'n_removido': n_removed,
                    'silhouette': sil,
                    'ari_vs_base': ari
                })
        
        df_outliers = pd.DataFrame(resultados_outliers)
        self.resultados['outliers'] = df_outliers
        
        self.log("\n  SENSIBILIDAD A OUTLIERS:")
        self.log("  " + "-"*50)
        for segmento in df_outliers['segmento'].unique():
            df_seg = df_outliers[df_outliers['segmento'] == segmento]
            if len(df_seg) > 0:
                self.log(f"\n  {segmento.upper()}:")
                for _, row in df_seg.iterrows():
                    self.log(f"    -{row['pct_removido']}%: Silhouette={row['silhouette']:.3f}, "
                            f"ARI={row['ari_vs_base']:.3f}")
        
        # Guardar
        df_outliers.to_csv(os.path.join(self.dir_tablas, 'sensibilidad_outliers.csv'), index=False)
    
    # ========================================================================
    # FASE 5: VALIDACIÓN POR VENTANAS TEMPORALES
    # ========================================================================
    def fase5_ventanas_temporales(self):
        """Evalúa si la taxonomía es estable en diferentes períodos."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Validación por Ventanas Temporales")
        self.log("="*70)
        
        # Verificar si tenemos datos temporales
        if 'primer_dia' not in self.datos.columns and 'fecha_primera_trans' not in self.datos.columns:
            self.log("  No hay datos temporales disponibles. Saltando...")
            self.resultados['temporal'] = None
            return
        
        # Usar columna temporal disponible
        time_col = 'primer_dia' if 'primer_dia' in self.datos.columns else 'fecha_primera_trans'
        
        try:
            self.datos[time_col] = pd.to_datetime(self.datos[time_col])
        except:
            self.log("  No se pudo parsear fecha. Saltando...")
            self.resultados['temporal'] = None
            return
        
        # Dividir en mitades temporales
        fecha_mediana = self.datos[time_col].median()
        
        resultados_temporal = []
        
        # Detectar segmentos disponibles
        if self.segmento_col in self.datos.columns:
            segmentos_disponibles = self.datos[self.segmento_col].unique().tolist()
        else:
            segmentos_disponibles = ['todos']
        
        k_por_segmento = {'profession': 3, 'mass': 2, 'todos': 3}
        
        for segmento in segmentos_disponibles:
            if self.segmento_col in self.datos.columns:
                mask_seg = self.datos[self.segmento_col] == segmento
            else:
                mask_seg = pd.Series([True] * len(self.datos))
            datos_seg = self.datos[mask_seg].copy()
            
            if len(datos_seg) < 50:
                continue
            
            # Primera mitad
            mask_h1 = datos_seg[time_col] <= fecha_mediana
            # Segunda mitad
            mask_h2 = datos_seg[time_col] > fecha_mediana
            
            datos_h1 = datos_seg[mask_h1]
            datos_h2 = datos_seg[mask_h2]
            
            if len(datos_h1) < 20 or len(datos_h2) < 20:
                continue
            
            k = k_por_segmento.get(segmento, 3)
            
            # Clustering en cada mitad
            for nombre, datos_half in [('H1', datos_h1), ('H2', datos_h2)]:
                X = datos_half[self.features_cols].values
                X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
                
                resultados_temporal.append({
                    'segmento': segmento,
                    'periodo': nombre,
                    'n_traders': len(datos_half),
                    'silhouette': sil
                })
        
        if resultados_temporal:
            df_temporal = pd.DataFrame(resultados_temporal)
            self.resultados['temporal'] = df_temporal
            
            self.log("\n  ESTABILIDAD TEMPORAL:")
            self.log("  " + "-"*50)
            for _, row in df_temporal.iterrows():
                self.log(f"  {row['segmento'].upper()} - {row['periodo']}: "
                        f"n={row['n_traders']}, Silhouette={row['silhouette']:.3f}")
            
            # Guardar
            df_temporal.to_csv(os.path.join(self.dir_tablas, 'validacion_temporal.csv'), index=False)
        else:
            self.resultados['temporal'] = None
    
    # ========================================================================
    # FASE 6: ANÁLISIS DE FRONTERA
    # ========================================================================
    def fase6_analisis_frontera(self):
        """Identifica traders cerca de fronteras entre clusters."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Análisis de Frontera (traders ambiguos)")
        self.log("="*70)
        
        resultados_frontera = []
        
        # Detectar segmentos disponibles
        if self.segmento_col in self.datos.columns:
            segmentos_disponibles = self.datos[self.segmento_col].unique().tolist()
        else:
            segmentos_disponibles = ['todos']
        
        k_por_segmento = {'profession': 3, 'mass': 2, 'todos': 3}
        
        for segmento in segmentos_disponibles:
            if self.segmento_col in self.datos.columns:
                mask = self.datos[self.segmento_col] == segmento
            else:
                mask = pd.Series([True] * len(self.datos))
            datos_seg = self.datos[mask].copy()
            
            if len(datos_seg) < 50:
                continue
            
            X = datos_seg[self.features_cols].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            k = k_por_segmento.get(segmento, 3)
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calcular distancias a todos los centroides
            distancias = np.zeros((len(X_scaled), k))
            for c in range(k):
                distancias[:, c] = np.linalg.norm(X_scaled - kmeans.cluster_centers_[c], axis=1)
            
            # Ratio de ambigüedad: distancia al más cercano / distancia al segundo más cercano
            distancias_sorted = np.sort(distancias, axis=1)
            ratio_ambiguedad = distancias_sorted[:, 0] / (distancias_sorted[:, 1] + 1e-10)
            
            # Traders en frontera: ratio > 0.8 (muy cerca de dos clusters)
            umbral_frontera = 0.8
            en_frontera = ratio_ambiguedad > umbral_frontera
            pct_frontera = en_frontera.mean() * 100
            
            # Silhouette individual
            from sklearn.metrics import silhouette_samples
            sil_samples = silhouette_samples(X_scaled, labels)
            
            # Traders con silhouette negativo (mal asignados)
            mal_asignados = (sil_samples < 0).mean() * 100
            
            resultados_frontera.append({
                'segmento': segmento,
                'n_traders': len(datos_seg),
                'pct_frontera': pct_frontera,
                'pct_mal_asignados': mal_asignados,
                'silhouette_mean': sil_samples.mean(),
                'silhouette_median': np.median(sil_samples)
            })
            
            # Guardar detalle por trader
            datos_seg = datos_seg.copy()
            datos_seg['silhouette_individual'] = sil_samples
            datos_seg['ratio_ambiguedad'] = ratio_ambiguedad
            datos_seg['en_frontera'] = en_frontera
            datos_seg['cluster_asignado'] = labels
            
            datos_seg[['userNo', 'cluster_asignado', 'silhouette_individual', 
                      'ratio_ambiguedad', 'en_frontera']].to_csv(
                os.path.join(self.dir_datos, f'frontera_{segmento}.csv'), index=False
            )
        
        df_frontera = pd.DataFrame(resultados_frontera)
        self.resultados['frontera'] = df_frontera
        
        self.log("\n  ANÁLISIS DE FRONTERA:")
        self.log("  " + "-"*50)
        for _, row in df_frontera.iterrows():
            self.log(f"  {row['segmento'].upper()}:")
            self.log(f"    Traders en frontera: {row['pct_frontera']:.1f}%")
            self.log(f"    Mal asignados (sil<0): {row['pct_mal_asignados']:.1f}%")
            self.log(f"    Silhouette medio: {row['silhouette_mean']:.3f}")
        
        # Guardar
        df_frontera.to_csv(os.path.join(self.dir_tablas, 'analisis_frontera.csv'), index=False)
    
    # ========================================================================
    # FASE 7: GRÁFICOS DE VALIDACIÓN
    # ========================================================================
    def fase7_graficos(self):
        """Genera gráficos de validación."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Generando gráficos")
        self.log("="*70)
        
        # Gráfico 1: Distribución Bootstrap ARI
        if 'bootstrap' in self.resultados:
            segmentos = [s for s in self.resultados['bootstrap'].keys() 
                        if self.resultados['bootstrap'][s]['ari']]
            n_seg = len(segmentos)
            
            if n_seg > 0:
                fig, axes = plt.subplots(1, n_seg, figsize=(7*n_seg, 5))
                if n_seg == 1:
                    axes = [axes]  # Convertir a lista
                
                for i, segmento in enumerate(segmentos):
                    axes[i].hist(self.resultados['bootstrap'][segmento]['ari'], 
                                bins=30, color='steelblue', alpha=0.7, edgecolor='white')
                    axes[i].axvline(x=np.mean(self.resultados['bootstrap'][segmento]['ari']),
                                   color='red', linestyle='--', linewidth=2,
                                   label=f"Media: {np.mean(self.resultados['bootstrap'][segmento]['ari']):.3f}")
                    axes[i].axvline(x=0.7, color='green', linestyle=':', linewidth=2,
                                   label='Umbral aceptable (0.7)')
                    axes[i].set_xlabel('Adjusted Rand Index (ARI)')
                    axes[i].set_ylabel('Frecuencia')
                    axes[i].set_title(f'{segmento.upper()}: Distribución ARI (Bootstrap)')
                    axes[i].legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.dir_graficos, '01_bootstrap_ari.png'), dpi=150)
                plt.close()
        
        # Gráfico 2: Distribución Bootstrap Consistencia
        if 'bootstrap' in self.resultados:
            segmentos = [s for s in self.resultados['bootstrap'].keys() 
                        if self.resultados['bootstrap'][s]['consistencia']]
            n_seg = len(segmentos)
            
            if n_seg > 0:
                fig, axes = plt.subplots(1, n_seg, figsize=(7*n_seg, 5))
                if n_seg == 1:
                    axes = [axes]  # Convertir a lista
                
                for i, segmento in enumerate(segmentos):
                    axes[i].hist(self.resultados['bootstrap'][segmento]['consistencia'], 
                                bins=30, color='darkorange', alpha=0.7, edgecolor='white')
                    mean_cons = np.mean(self.resultados['bootstrap'][segmento]['consistencia'])
                    axes[i].axvline(x=mean_cons, color='red', linestyle='--', linewidth=2,
                                   label=f"Media: {mean_cons:.1%}")
                    axes[i].axvline(x=0.8, color='green', linestyle=':', linewidth=2,
                                   label='Umbral aceptable (80%)')
                    axes[i].set_xlabel('Consistencia de Asignación')
                    axes[i].set_ylabel('Frecuencia')
                    axes[i].set_title(f'{segmento.upper()}: Consistencia (Bootstrap)')
                    axes[i].legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.dir_graficos, '02_bootstrap_consistencia.png'), dpi=150)
                plt.close()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir_graficos, '02_bootstrap_consistencia.png'), dpi=150)
            plt.close()
        
        # Gráfico 3: Jaccard por tipo
        if 'jaccard' in self.resultados and self.resultados['jaccard'] is not None:
            df_j = self.resultados['jaccard']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(df_j))
            colors = ['#3498db' if 'P' in t else '#e74c3c' for t in df_j['tipo']]
            
            ax.bar(x, df_j['jaccard_mean'], yerr=df_j['jaccard_std'], 
                   color=colors, alpha=0.8, capsize=5)
            ax.axhline(y=0.6, color='green', linestyle='--', label='Umbral aceptable (0.6)')
            ax.set_xticks(x)
            ax.set_xticklabels(df_j['tipo'])
            ax.set_ylabel('Índice Jaccard')
            ax.set_xlabel('Tipo de Agente')
            ax.set_title('Estabilidad Jaccard por Tipo de Agente')
            ax.legend()
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir_graficos, '03_jaccard_por_tipo.png'), dpi=150)
            plt.close()
        
        # Gráfico 4: Sensibilidad a Outliers
        if 'outliers' in self.resultados:
            df_o = self.resultados['outliers']
            segmentos = df_o['segmento'].unique().tolist()
            n_seg = len(segmentos)
            
            if n_seg > 0:
                fig, axes = plt.subplots(1, n_seg, figsize=(7*n_seg, 5))
                if n_seg == 1:
                    axes = [axes]  # Convertir a lista
                
                for i, segmento in enumerate(segmentos):
                    df_seg = df_o[df_o['segmento'] == segmento]
                    if len(df_seg) > 0:
                        ax2 = axes[i].twinx()
                        
                        l1 = axes[i].plot(df_seg['pct_removido'], df_seg['silhouette'], 
                                         'bo-', linewidth=2, markersize=8, label='Silhouette')
                        l2 = ax2.plot(df_seg['pct_removido'], df_seg['ari_vs_base'], 
                                     'rs--', linewidth=2, markersize=8, label='ARI vs Base')
                        
                        axes[i].set_xlabel('% Outliers Removidos')
                        axes[i].set_ylabel('Silhouette', color='blue')
                        ax2.set_ylabel('ARI vs Base', color='red')
                        axes[i].set_title(f'{segmento.upper()}: Sensibilidad a Outliers')
                        
                        lines = l1 + l2
                        labels = [l.get_label() for l in lines]
                        axes[i].legend(lines, labels, loc='lower right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.dir_graficos, '04_sensibilidad_outliers.png'), dpi=150)
                plt.close()
        
        # Gráfico 5: Resumen de validación
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metricas = []
        valores = []
        colores = []
        umbrales = []
        
        if 'resumen_bootstrap' in self.resultados:
            for _, row in self.resultados['resumen_bootstrap'].iterrows():
                metricas.append(f"{row['segmento'].upper()}\nARI")
                valores.append(row['ari_mean'])
                colores.append('#3498db' if row['segmento'] == 'profession' else '#e74c3c')
                umbrales.append(0.7)
                
                metricas.append(f"{row['segmento'].upper()}\nConsist.")
                valores.append(row['consistencia_mean'])
                colores.append('#3498db' if row['segmento'] == 'profession' else '#e74c3c')
                umbrales.append(0.8)
        
        if 'frontera' in self.resultados:
            for _, row in self.resultados['frontera'].iterrows():
                metricas.append(f"{row['segmento'].upper()}\n% Estable")
                valores.append(1 - row['pct_frontera']/100)
                colores.append('#3498db' if row['segmento'] == 'profession' else '#e74c3c')
                umbrales.append(0.8)
        
        if metricas:
            x = range(len(metricas))
            bars = ax.bar(x, valores, color=colores, alpha=0.8)
            
            # Umbrales
            for i, (xi, u) in enumerate(zip(x, umbrales)):
                ax.plot([xi-0.4, xi+0.4], [u, u], 'g--', linewidth=2)
            
            ax.set_xticks(x)
            ax.set_xticklabels(metricas)
            ax.set_ylabel('Valor')
            ax.set_title('Resumen de Métricas de Validación de Estabilidad\n(línea verde = umbral aceptable)')
            ax.set_ylim(0, 1.1)
            
            # Añadir valores sobre barras
            for bar, val in zip(bars, valores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '05_resumen_validacion.png'), dpi=150)
        plt.close()
        
        self.log("Gráficos generados")
    
    # ========================================================================
    # FASE 8: TABLA RESUMEN PARA TESIS
    # ========================================================================
    def fase8_tabla_tesis(self):
        """Genera tabla resumen para incluir en la tesis (Tabla 2.3)."""
        self.log("\n" + "="*70)
        self.log("FASE 8: Generando Tabla para Tesis")
        self.log("="*70)
        
        filas = []
        
        # Obtener segmentos disponibles
        segmentos_disponibles = []
        if 'resumen_bootstrap' in self.resultados and len(self.resultados['resumen_bootstrap']) > 0:
            segmentos_disponibles = self.resultados['resumen_bootstrap']['segmento'].unique().tolist()
        elif 'jaccard' in self.resultados and self.resultados['jaccard'] is not None:
            segmentos_disponibles = self.resultados['jaccard']['segmento'].unique().tolist()
        else:
            segmentos_disponibles = ['todos']
        
        for segmento in segmentos_disponibles:
            fila = {'Segmento': segmento.capitalize()}
            
            # Bootstrap
            if 'resumen_bootstrap' in self.resultados:
                df_boot = self.resultados['resumen_bootstrap']
                row_boot = df_boot[df_boot['segmento'] == segmento]
                if len(row_boot) > 0:
                    fila['ARI (bootstrap)'] = f"{row_boot['ari_mean'].values[0]:.3f} ± {row_boot['ari_std'].values[0]:.3f}"
                    fila['Consistencia'] = f"{row_boot['consistencia_mean'].values[0]:.1%}"
            
            # Jaccard promedio
            if 'jaccard' in self.resultados and self.resultados['jaccard'] is not None:
                df_j = self.resultados['jaccard']
                jac_seg = df_j[df_j['segmento'] == segmento]['jaccard_mean'].mean()
                fila['Jaccard (promedio)'] = f"{jac_seg:.3f}"
            
            # Frontera
            if 'frontera' in self.resultados:
                df_f = self.resultados['frontera']
                row_f = df_f[df_f['segmento'] == segmento]
                if len(row_f) > 0:
                    fila['% En frontera'] = f"{row_f['pct_frontera'].values[0]:.1f}%"
                    fila['% Mal asignados'] = f"{row_f['pct_mal_asignados'].values[0]:.1f}%"
            
            filas.append(fila)
        
        df_tesis = pd.DataFrame(filas)
        self.resultados['tabla_tesis'] = df_tesis
        
        self.log("\n  TABLA 2.3: VALIDACIÓN DE ESTABILIDAD DE LA TAXONOMÍA")
        self.log("  " + "="*60)
        self.log(df_tesis.to_string(index=False))
        
        # Guardar
        df_tesis.to_csv(os.path.join(self.dir_tablas, 'tabla_2_3_estabilidad.csv'), index=False)
        
        # Interpretación
        self.log("\n  INTERPRETACIÓN:")
        self.log("  " + "-"*50)
        
        if 'resumen_bootstrap' in self.resultados:
            df_boot = self.resultados['resumen_bootstrap']
            for _, row in df_boot.iterrows():
                seg = row['segmento'].upper()
                ari = row['ari_mean']
                cons = row['consistencia_mean']
                
                if ari >= 0.7:
                    interp_ari = "BUENA"
                elif ari >= 0.5:
                    interp_ari = "MODERADA"
                else:
                    interp_ari = "BAJA"
                
                if cons >= 0.8:
                    interp_cons = "ALTA"
                elif cons >= 0.7:
                    interp_cons = "MODERADA"
                else:
                    interp_cons = "BAJA"
                
                self.log(f"  {seg}:")
                self.log(f"    - Estabilidad ARI: {interp_ari} ({ari:.3f})")
                self.log(f"    - Consistencia: {interp_cons} ({cons:.1%})")
    
    # ========================================================================
    # FASE 9: REPORTE FINAL
    # ========================================================================
    def fase9_reporte(self):
        """Genera reporte final."""
        self.log("\n" + "="*70)
        self.log("FASE 9: Reporte Final")
        self.log("="*70)
        
        reporte_path = os.path.join(self.directorio_salida, 'validacion_estabilidad_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
        
        self.log(f"\nReporte guardado: {reporte_path}")
        self.log(f"Tablas guardadas en: {self.dir_tablas}")
        self.log(f"Gráficos guardados en: {self.dir_graficos}")
    
    # ========================================================================
    # EJECUTAR
    # ========================================================================
    def ejecutar(self, n_bootstrap=100):
        """Ejecuta validación completa."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        self.log("="*70)
        self.log("VALIDACIÓN DE ESTABILIDAD DEL CLUSTERING")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_bootstrap_resampling(n_iterations=n_bootstrap)
            self.fase3_jaccard_por_cluster(n_iterations=50)
            self.fase4_sensibilidad_outliers()
            self.fase5_ventanas_temporales()
            self.fase6_analisis_frontera()
            self.fase7_graficos()
            self.fase8_tabla_tesis()
            self.fase9_reporte()
            
            self.log(f"\n{'='*70}")
            self.log(f"COMPLETADO EN: {datetime.now() - inicio}")
            self.log(f"{'='*70}")
            
            return self.resultados
            
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
    print("VALIDACIÓN DE ESTABILIDAD DEL CLUSTERING")
    print("="*70)
    
    print("\n1. Selecciona el archivo de FEATURES (features_traders_completo.csv)...")
    ruta_features = filedialog.askopenfilename(
        title='Seleccionar CSV de Features',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta_features:
        print("Cancelado.")
        return
    
    print("\n2. Selecciona el archivo de CLUSTERS (traders_con_clusters.csv)...")
    ruta_clusters = filedialog.askopenfilename(
        title='Seleccionar CSV de Clusters',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta_clusters:
        print("Cancelado.")
        return
    
    validador = ValidacionEstabilidadClustering(ruta_features, ruta_clusters)
    validador.ejecutar(n_bootstrap=100)


if __name__ == "__main__":
    main()
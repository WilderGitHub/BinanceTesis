"""
================================================================================
ANALISIS_HETEROGENEIDAD_CLUSTERS.PY
================================================================================
Analisis de:
1. Ticket promedio por dia de semana
2. Efecto fin de semana por tipo de agente (clusters)
3. Modelos con interacciones por tipo

Requiere haber ejecutado previamente clustering_estratificado.py para tener
los archivos de clusters generados.

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
import gc
from datetime import datetime
from scipy import stats

# Configuracion
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Importar modulo core
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
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    print("ERROR: statsmodels no instalado")
    print("Ejecutar: pip install statsmodels")
    sys.exit(1)

CHUNK_SIZE = 500_000


class AnalisisHeterogeneidadClusters:
    """Analisis de heterogeneidad del efecto FDS por tipo de agente."""
    
    def __init__(self, ruta_csv, ruta_clusters=None, directorio_salida=None):
        """
        Args:
            ruta_csv: Ruta al CSV de Binance P2P
            ruta_clusters: Ruta al archivo traders_features_completo.csv del clustering
            directorio_salida: Directorio para outputs
        """
        self.ruta_csv = ruta_csv
        self.ruta_clusters = ruta_clusters
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_csv)
        
        # Subdirectorios
        self.dir_graficos = os.path.join(self.directorio_salida, 'heterogeneidad_graficos')
        self.dir_datos = os.path.join(self.directorio_salida, 'heterogeneidad_datos')
        self.dir_tablas = os.path.join(self.directorio_salida, 'heterogeneidad_tablas')
        os.makedirs(self.dir_graficos, exist_ok=True)
        os.makedirs(self.dir_datos, exist_ok=True)
        os.makedirs(self.dir_tablas, exist_ok=True)
        
        # Datos
        self.transacciones = None
        self.clusters = None
        self.datos_diarios = None
        self.datos_por_tipo = None
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
        """Carga transacciones y datos de clusters."""
        self.log("="*70)
        self.log("FASE 1: Carga de datos")
        self.log("="*70)
        
        # Cargar transacciones
        self.log(f"Cargando transacciones desde: {self.ruta_csv}")
        file_size = os.path.getsize(self.ruta_csv)
        
        if file_size > 1e9:
            self.log(f"Archivo grande ({file_size/1e9:.2f} GB), procesando por chunks...")
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
        
        # Cargar clusters si existe
        if self.ruta_clusters and os.path.exists(self.ruta_clusters):
            self.log(f"\nCargando clusters desde: {self.ruta_clusters}")
            self.clusters = pd.read_csv(self.ruta_clusters)
            self.log(f"Traders con cluster asignado: {len(self.clusters):,}")
        else:
            self.log("\nNo se encontro archivo de clusters. Se usara segmentacion Binance.")
            self.clusters = None
    
    # ========================================================================
    # FASE 2: PREPARAR DATOS
    # ========================================================================
    def fase2_preparar_datos(self):
        """Prepara datos para analisis."""
        self.log("\n" + "="*70)
        self.log("FASE 2: Preparacion de datos")
        self.log("="*70)
        
        df = self.transacciones.copy()
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['dia_nombre'] = df['fecha'].dt.day_name()
        df['es_fds'] = (df['dia_semana'] >= 5).astype(int)
        
        # Merge con clusters si existe
        if self.clusters is not None:
            self.log("Asignando clusters a transacciones...")
            
            # El archivo de clusters tiene userNo y cluster_name o tipo_final
            cols_cluster = self.clusters.columns.tolist()
            
            # Buscar columna de cluster
            cluster_col = None
            for col in ['tipo_final', 'cluster_name', 'cluster', 'tipo']:
                if col in cols_cluster:
                    cluster_col = col
                    break
            
            if cluster_col:
                clusters_map = self.clusters.set_index('userNo')[cluster_col].to_dict()
                df['tipo_cluster'] = df['userNo'].map(clusters_map)
                df['tipo_cluster'] = df['tipo_cluster'].fillna('Sin_Cluster')
                
                # Mostrar distribucion
                dist = df.groupby('tipo_cluster')['montoDynamic'].agg(['count', 'sum'])
                dist['pct_vol'] = dist['sum'] / dist['sum'].sum() * 100
                self.log(f"\nDistribucion por tipo de cluster:")
                for tipo, row in dist.iterrows():
                    self.log(f"  {tipo}: {row['count']:,} trans, {row['pct_vol']:.1f}% vol")
            else:
                self.log("No se encontro columna de cluster. Usando segmentacion Binance.")
                df['tipo_cluster'] = df['adv.classify']
        else:
            df['tipo_cluster'] = df['adv.classify']
        
        self.transacciones = df
        
        # ================================================================
        # DATOS DIARIOS GLOBALES
        # ================================================================
        self.log("\nAgregando datos diarios globales...")
        
        diario = df.groupby('fecha').agg({
            'montoDynamic': ['sum', 'count', 'mean'],
            'precio_transaccion': 'mean',
            'userNo': 'nunique'
        }).reset_index()
        
        diario.columns = ['fecha', 'volumen', 'n_trans', 'ticket_mean', 'precio', 'n_traders']
        
        # Calcular ticket promedio correctamente
        diario['ticket_promedio'] = diario['volumen'] / diario['n_trans']
        
        # Variables calendario
        diario['dia_semana'] = diario['fecha'].dt.dayofweek
        diario['dia_nombre'] = diario['fecha'].dt.day_name()
        diario['es_fds'] = (diario['dia_semana'] >= 5).astype(int)
        diario['tendencia'] = (diario['fecha'] - diario['fecha'].min()).dt.days
        
        # Transformaciones log
        diario['log_volumen'] = np.log1p(diario['volumen'])
        diario['log_trans'] = np.log1p(diario['n_trans'])
        diario['log_ticket'] = np.log1p(diario['ticket_promedio'])
        
        self.datos_diarios = diario
        self.log(f"Dias: {len(diario)}")
        
        # ================================================================
        # DATOS DIARIOS POR TIPO DE CLUSTER
        # ================================================================
        self.log("\nAgregando datos por tipo de cluster...")
        
        tipos_principales = df['tipo_cluster'].value_counts().head(10).index.tolist()
        
        datos_tipo = []
        for tipo in tipos_principales:
            df_tipo = df[df['tipo_cluster'] == tipo]
            
            diario_tipo = df_tipo.groupby('fecha').agg({
                'montoDynamic': ['sum', 'count'],
                'userNo': 'nunique'
            }).reset_index()
            
            diario_tipo.columns = ['fecha', 'volumen', 'n_trans', 'n_traders']
            diario_tipo['ticket_promedio'] = diario_tipo['volumen'] / diario_tipo['n_trans']
            diario_tipo['tipo_cluster'] = tipo
            
            # Variables calendario
            diario_tipo['dia_semana'] = diario_tipo['fecha'].dt.dayofweek
            diario_tipo['es_fds'] = (diario_tipo['dia_semana'] >= 5).astype(int)
            diario_tipo['tendencia'] = (diario_tipo['fecha'] - diario_tipo['fecha'].min()).dt.days
            
            # Log
            diario_tipo['log_volumen'] = np.log1p(diario_tipo['volumen'])
            diario_tipo['log_ticket'] = np.log1p(diario_tipo['ticket_promedio'])
            
            datos_tipo.append(diario_tipo)
        
        self.datos_por_tipo = pd.concat(datos_tipo, ignore_index=True)
        self.log(f"Tipos analizados: {len(tipos_principales)}")
    
    # ========================================================================
    # FASE 3: ANALISIS DE TICKET PROMEDIO
    # ========================================================================
    def fase3_ticket_promedio(self):
        """Analisis del ticket promedio por dia de semana."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Analisis de Ticket Promedio")
        self.log("="*70)
        
        df = self.datos_diarios.copy()
        
        # ================================================================
        # 3.1 Estadisticas descriptivas
        # ================================================================
        self.log("\n--- 3.1 Ticket Promedio por Dia de Semana ---")
        
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_esp = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
        
        stats_dia = df.groupby('dia_nombre').agg({
            'ticket_promedio': ['mean', 'std', 'median'],
            'volumen': 'mean',
            'n_trans': 'mean'
        }).reindex(dias_orden)
        
        stats_dia.columns = ['ticket_mean', 'ticket_std', 'ticket_median', 'vol_mean', 'trans_mean']
        
        # Calcular ticket desde agregados (para verificar)
        stats_dia['ticket_calc'] = stats_dia['vol_mean'] / stats_dia['trans_mean']
        
        ticket_global = df['ticket_promedio'].mean()
        stats_dia['var_pct'] = (stats_dia['ticket_mean'] / ticket_global - 1) * 100
        
        self.log(f"\n{'Dia':<12} {'Ticket Prom':>12} {'Var vs Media':>12} {'Volumen':>15} {'Trans':>10}")
        self.log("-" * 65)
        
        for i, dia in enumerate(dias_orden):
            row = stats_dia.loc[dia]
            self.log(f"{dias_esp[i]:<12} ${row['ticket_mean']:>11,.0f} {row['var_pct']:>+11.1f}% {row['vol_mean']:>15,.0f} {row['trans_mean']:>10,.0f}")
        
        self.log(f"\nTicket promedio global: ${ticket_global:,.2f}")
        
        # ================================================================
        # 3.2 Comparacion FDS vs Entre Semana
        # ================================================================
        self.log("\n--- 3.2 Ticket: Fin de Semana vs Entre Semana ---")
        
        fds = df[df['es_fds'] == 1]
        es = df[df['es_fds'] == 0]
        
        ticket_es = es['ticket_promedio'].mean()
        ticket_fds = fds['ticket_promedio'].mean()
        dif_pct = (ticket_fds / ticket_es - 1) * 100
        
        self.log(f"\n  Ticket entre semana:    ${ticket_es:,.2f}")
        self.log(f"  Ticket fin de semana:   ${ticket_fds:,.2f}")
        self.log(f"  Diferencia:             {dif_pct:+.2f}%")
        
        # Test t
        t_stat, p_value = stats.ttest_ind(es['ticket_promedio'], fds['ticket_promedio'])
        self.log(f"\n  Test t: t={t_stat:.3f}, p-value={p_value:.4f}")
        
        if p_value < 0.05:
            self.log(f"  -> Diferencia SIGNIFICATIVA (p < 0.05)")
        else:
            self.log(f"  -> Diferencia NO significativa")
        
        # ================================================================
        # 3.3 Regresion del Ticket
        # ================================================================
        self.log("\n--- 3.3 Regresion: Efecto FDS en Ticket ---")
        
        formula = 'log_ticket ~ es_fds + tendencia'
        modelo = smf.ols(formula, data=df).fit(cov_type='HC3')
        
        self.log(f"\n{modelo.summary()}")
        
        coef_fds = modelo.params['es_fds']
        pval_fds = modelo.pvalues['es_fds']
        efecto_pct = (np.exp(coef_fds) - 1) * 100
        
        self.log(f"\n-> Interpretacion:")
        self.log(f"   Coeficiente es_fds: {coef_fds:.4f}")
        self.log(f"   Efecto en ticket: {efecto_pct:+.2f}%")
        self.log(f"   p-value: {pval_fds:.4f}")
        
        self.resultados['ticket_efecto_fds'] = efecto_pct
        self.resultados['ticket_pvalue'] = pval_fds
        
        # ================================================================
        # 3.4 Descomposicion del Efecto FDS
        # ================================================================
        self.log("\n--- 3.4 Descomposicion del Efecto FDS ---")
        
        vol_es = es['volumen'].mean()
        vol_fds = fds['volumen'].mean()
        trans_es = es['n_trans'].mean()
        trans_fds = fds['n_trans'].mean()
        
        efecto_vol = (vol_fds / vol_es - 1) * 100
        efecto_trans = (trans_fds / trans_es - 1) * 100
        efecto_ticket = (ticket_fds / ticket_es - 1) * 100
        
        self.log(f"\n  Efecto FDS en Volumen:       {efecto_vol:+.1f}%")
        self.log(f"  Efecto FDS en Transacciones: {efecto_trans:+.1f}%")
        self.log(f"  Efecto FDS en Ticket:        {efecto_ticket:+.1f}%")
        
        # Descomposicion aproximada
        self.log(f"\n  Descomposicion aproximada:")
        self.log(f"  Efecto Volumen = Efecto Trans + Efecto Ticket")
        self.log(f"  {efecto_vol:+.1f}% ≈ {efecto_trans:+.1f}% + {efecto_ticket:+.1f}%")
        
        contribucion_trans = efecto_trans / efecto_vol * 100 if efecto_vol != 0 else 0
        contribucion_ticket = efecto_ticket / efecto_vol * 100 if efecto_vol != 0 else 0
        
        self.log(f"\n  Contribucion al efecto total:")
        self.log(f"    - Por menos transacciones: {contribucion_trans:.1f}%")
        self.log(f"    - Por ticket mas pequeno:  {contribucion_ticket:.1f}%")
        
        self.resultados['descomposicion'] = {
            'efecto_vol': efecto_vol,
            'efecto_trans': efecto_trans,
            'efecto_ticket': efecto_ticket,
            'contrib_trans': contribucion_trans,
            'contrib_ticket': contribucion_ticket
        }
        
        # Guardar stats
        stats_dia.to_csv(os.path.join(self.dir_tablas, 'ticket_por_dia.csv'))
    
    # ========================================================================
    # FASE 4: EFECTO FDS POR TIPO DE CLUSTER
    # ========================================================================
    def fase4_efecto_por_tipo(self):
        """Analisis del efecto FDS por tipo de cluster."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Efecto Fin de Semana por Tipo de Agente")
        self.log("="*70)
        
        df_tipo = self.datos_por_tipo.copy()
        
        # ================================================================
        # 4.1 Estadisticas descriptivas por tipo
        # ================================================================
        self.log("\n--- 4.1 Efecto FDS Descriptivo por Tipo ---")
        
        tipos = df_tipo['tipo_cluster'].unique()
        resultados_tipo = []
        
        self.log(f"\n{'Tipo':<25} {'Vol ES':>12} {'Vol FDS':>12} {'Efecto':>10} {'p-value':>10}")
        self.log("-" * 75)
        
        for tipo in tipos:
            df_t = df_tipo[df_tipo['tipo_cluster'] == tipo]
            
            es = df_t[df_t['es_fds'] == 0]
            fds = df_t[df_t['es_fds'] == 1]
            
            if len(es) < 5 or len(fds) < 5:
                continue
            
            vol_es = es['volumen'].mean()
            vol_fds = fds['volumen'].mean()
            efecto = (vol_fds / vol_es - 1) * 100
            
            t_stat, p_value = stats.ttest_ind(es['volumen'], fds['volumen'])
            
            sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
            
            self.log(f"{tipo:<25} {vol_es:>12,.0f} {vol_fds:>12,.0f} {efecto:>+9.1f}% {p_value:>9.4f} {sig}")
            
            resultados_tipo.append({
                'tipo': tipo,
                'vol_es': vol_es,
                'vol_fds': vol_fds,
                'efecto_pct': efecto,
                't_stat': t_stat,
                'p_value': p_value,
                'n_dias_es': len(es),
                'n_dias_fds': len(fds)
            })
        
        df_resultados = pd.DataFrame(resultados_tipo)
        df_resultados = df_resultados.sort_values('vol_es', ascending=False)
        df_resultados.to_csv(os.path.join(self.dir_tablas, 'efecto_fds_por_tipo.csv'), index=False)
        
        self.resultados['efecto_por_tipo'] = df_resultados
        
        # ================================================================
        # 4.2 Regresiones por tipo
        # ================================================================
        self.log("\n--- 4.2 Regresiones por Tipo de Agente ---")
        
        for tipo in df_resultados['tipo'].head(5):  # Top 5 tipos
            df_t = df_tipo[df_tipo['tipo_cluster'] == tipo]
            
            if len(df_t) < 30:
                continue
            
            self.log(f"\n  {tipo}:")
            
            try:
                formula = 'log_volumen ~ es_fds + tendencia'
                modelo = smf.ols(formula, data=df_t).fit(cov_type='HC3')
                
                coef = modelo.params['es_fds']
                pval = modelo.pvalues['es_fds']
                efecto = (np.exp(coef) - 1) * 100
                
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                self.log(f"    Efecto FDS: {efecto:+.1f}% (p={pval:.4f}) {sig}")
                self.log(f"    R2: {modelo.rsquared:.3f}, n={int(modelo.nobs)}")
            except Exception as e:
                self.log(f"    Error: {e}")
        
        # ================================================================
        # 4.3 Modelo con interacciones
        # ================================================================
        self.log("\n--- 4.3 Modelo con Interacciones por Tipo ---")
        
        # Identificar tipos principales para el modelo
        tipos_principales = df_resultados.head(3)['tipo'].tolist()
        
        if len(tipos_principales) >= 2:
            tipo_ref = tipos_principales[0]  # Referencia (el mas grande)
            
            # Crear dummies
            df_modelo = df_tipo[df_tipo['tipo_cluster'].isin(tipos_principales)].copy()
            
            for tipo in tipos_principales[1:]:
                col_name = f"es_{tipo.replace(' ', '_').replace('/', '_')}"
                df_modelo[col_name] = (df_modelo['tipo_cluster'] == tipo).astype(int)
            
            # Formula con interacciones
            dummies = [f"es_{t.replace(' ', '_').replace('/', '_')}" for t in tipos_principales[1:]]
            interacciones = [f"es_fds:{d}" for d in dummies]
            
            formula = f"log_volumen ~ es_fds + {' + '.join(dummies)} + {' + '.join(interacciones)} + tendencia"
            
            self.log(f"\n  Modelo: {formula}")
            self.log(f"  Tipo de referencia: {tipo_ref}")
            
            try:
                modelo_inter = smf.ols(formula, data=df_modelo).fit(cov_type='HC3')
                self.log(f"\n{modelo_inter.summary()}")
                
                # Interpretar
                self.log(f"\n-> Interpretacion de Interacciones:")
                
                coef_fds_base = modelo_inter.params['es_fds']
                efecto_base = (np.exp(coef_fds_base) - 1) * 100
                self.log(f"   Efecto FDS en {tipo_ref}: {efecto_base:+.1f}%")
                
                for d in interacciones:
                    if d in modelo_inter.params:
                        coef_inter = modelo_inter.params[d]
                        pval_inter = modelo_inter.pvalues[d]
                        efecto_adicional = (np.exp(coef_inter) - 1) * 100
                        
                        tipo_name = d.replace('es_fds:', '').replace('es_', '').replace('_', ' ')
                        efecto_total = (np.exp(coef_fds_base + coef_inter) - 1) * 100
                        
                        sig = '***' if pval_inter < 0.01 else '**' if pval_inter < 0.05 else '*' if pval_inter < 0.1 else 'n.s.'
                        self.log(f"   Efecto FDS en {tipo_name}: {efecto_total:+.1f}% (interaccion p={pval_inter:.4f}) {sig}")
                
                self.resultados['modelo_interacciones'] = modelo_inter
                
            except Exception as e:
                self.log(f"  Error en modelo con interacciones: {e}")
    
    # ========================================================================
    # FASE 5: TICKET PROMEDIO POR TIPO
    # ========================================================================
    def fase5_ticket_por_tipo(self):
        """Analisis de ticket promedio por tipo de agente."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Ticket Promedio por Tipo de Agente")
        self.log("="*70)
        
        df_tipo = self.datos_por_tipo.copy()
        
        tipos = df_tipo['tipo_cluster'].unique()
        
        self.log(f"\n{'Tipo':<25} {'Ticket ES':>12} {'Ticket FDS':>12} {'Efecto':>10}")
        self.log("-" * 65)
        
        for tipo in tipos:
            df_t = df_tipo[df_tipo['tipo_cluster'] == tipo]
            
            es = df_t[df_t['es_fds'] == 0]
            fds = df_t[df_t['es_fds'] == 1]
            
            if len(es) < 5 or len(fds) < 5:
                continue
            
            ticket_es = es['ticket_promedio'].mean()
            ticket_fds = fds['ticket_promedio'].mean()
            efecto = (ticket_fds / ticket_es - 1) * 100
            
            self.log(f"{tipo:<25} ${ticket_es:>11,.0f} ${ticket_fds:>11,.0f} {efecto:>+9.1f}%")
    
    # ========================================================================
    # FASE 6: VISUALIZACIONES
    # ========================================================================
    def fase6_visualizaciones(self):
        """Genera visualizaciones."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Visualizaciones")
        self.log("="*70)
        
        self._grafico_ticket_dia_semana()
        self._grafico_descomposicion()
        self._grafico_efecto_por_tipo()
        
        self.log("Visualizaciones generadas")
    
    def _grafico_ticket_dia_semana(self):
        """Grafico de ticket promedio por dia de semana."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        df = self.datos_diarios.copy()
        
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_esp = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        colores = ['steelblue']*5 + ['darkorange']*2
        
        # Ticket promedio
        ticket_dia = df.groupby('dia_nombre')['ticket_promedio'].mean().reindex(dias_orden)
        ticket_global = df['ticket_promedio'].mean()
        
        axes[0].bar(dias_esp, ticket_dia.values, color=colores)
        axes[0].axhline(y=ticket_global, color='red', linestyle='--', 
                        label=f'Media: ${ticket_global:,.0f}')
        axes[0].set_ylabel('Ticket Promedio (USDT)')
        axes[0].set_title('Ticket Promedio por Dia de Semana')
        axes[0].legend()
        
        # Variacion porcentual
        var_pct = (ticket_dia / ticket_global - 1) * 100
        colors_var = ['green' if v >= 0 else 'red' for v in var_pct.values]
        
        axes[1].bar(dias_esp, var_pct.values, color=colors_var)
        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_ylabel('Variacion vs Media (%)')
        axes[1].set_title('Efecto Dia de Semana en Ticket')
        
        for i, v in enumerate(var_pct.values):
            axes[1].text(i, v + 0.3 if v >= 0 else v - 0.8, f'{v:+.1f}%', 
                        ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '01_ticket_dia_semana.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_descomposicion(self):
        """Grafico de descomposicion del efecto FDS."""
        if 'descomposicion' not in self.resultados:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        desc = self.resultados['descomposicion']
        
        # Efectos
        categorias = ['Volumen', 'Transacciones', 'Ticket']
        valores = [desc['efecto_vol'], desc['efecto_trans'], desc['efecto_ticket']]
        colores = ['red' if v < 0 else 'green' for v in valores]
        
        bars = axes[0].bar(categorias, valores, color=colores)
        axes[0].axhline(y=0, color='black', linewidth=1)
        axes[0].set_ylabel('Efecto Fin de Semana (%)')
        axes[0].set_title('Efecto FDS por Componente')
        
        for bar, val in zip(bars, valores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:+.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        # Contribucion
        contrib = [abs(desc['contrib_trans']), abs(desc['contrib_ticket'])]
        labels = ['Menos\nTransacciones', 'Ticket\nmas pequeno']
        colores = ['steelblue', 'darkorange']
        
        axes[1].pie(contrib, labels=labels, colors=colores, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 11})
        axes[1].set_title('Contribucion al Efecto FDS en Volumen')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '02_descomposicion_efecto.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _grafico_efecto_por_tipo(self):
        """Grafico del efecto FDS por tipo de agente."""
        if 'efecto_por_tipo' not in self.resultados:
            return
        
        df = self.resultados['efecto_por_tipo'].head(6).copy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(df))
        colores = ['green' if p < 0.05 else 'gray' for p in df['p_value']]
        
        bars = ax.bar(x, df['efecto_pct'], color=colores)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axhline(y=df['efecto_pct'].mean(), color='red', linestyle='--',
                   label=f'Promedio: {df["efecto_pct"].mean():.1f}%')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df['tipo'], rotation=45, ha='right')
        ax.set_ylabel('Efecto Fin de Semana (%)')
        ax.set_title('Efecto FDS por Tipo de Agente\n(Verde = significativo al 5%)')
        ax.legend()
        
        for bar, val, pval in zip(bars, df['efecto_pct'], df['p_value']):
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else ''
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                   f'{val:.1f}%{sig}', ha='center', va='top', 
                   fontsize=10, color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_efecto_por_tipo.png'),
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
        reporte.append("ANALISIS DE HETEROGENEIDAD Y TICKET PROMEDIO")
        reporte.append("="*70)
        reporte.append(f"\nFecha: {datetime.now()}")
        
        # Resumen ticket
        if 'descomposicion' in self.resultados:
            desc = self.resultados['descomposicion']
            reporte.append("\n" + "="*70)
            reporte.append("DESCOMPOSICION DEL EFECTO FIN DE SEMANA")
            reporte.append("="*70)
            reporte.append(f"\n  Efecto en Volumen:       {desc['efecto_vol']:+.1f}%")
            reporte.append(f"  Efecto en Transacciones: {desc['efecto_trans']:+.1f}%")
            reporte.append(f"  Efecto en Ticket:        {desc['efecto_ticket']:+.1f}%")
            reporte.append(f"\n  Contribucion:")
            reporte.append(f"    - Por menos transacciones: {desc['contrib_trans']:.1f}%")
            reporte.append(f"    - Por ticket mas pequeno:  {desc['contrib_ticket']:.1f}%")
        
        # Resumen por tipo
        if 'efecto_por_tipo' in self.resultados:
            df = self.resultados['efecto_por_tipo']
            reporte.append("\n" + "="*70)
            reporte.append("EFECTO FDS POR TIPO DE AGENTE")
            reporte.append("="*70)
            for _, row in df.iterrows():
                sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else ''
                reporte.append(f"  {row['tipo']:<25}: {row['efecto_pct']:+.1f}% {sig}")
        
        # Guardar
        reporte_path = os.path.join(self.directorio_salida, 'heterogeneidad_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
            f.write('\n\n')
            f.write('\n'.join(reporte))
        
        self.log(f"\nReporte: {reporte_path}")
        self.log(f"Graficos: {self.dir_graficos}")
        self.log(f"Tablas: {self.dir_tablas}")
    
    # ========================================================================
    # EJECUTAR
    # ========================================================================
    def ejecutar(self):
        """Ejecuta analisis completo."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        self.log("="*70)
        self.log("ANALISIS DE HETEROGENEIDAD Y TICKET PROMEDIO")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_preparar_datos()
            self.fase3_ticket_promedio()
            self.fase4_efecto_por_tipo()
            self.fase5_ticket_por_tipo()
            self.fase6_visualizaciones()
            self.fase7_reporte()
            
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
    print("ANALISIS DE HETEROGENEIDAD Y TICKET PROMEDIO")
    print("="*70)
    
    print("\n[1/2] Selecciona el archivo CSV de Binance P2P...")
    ruta_csv = filedialog.askopenfilename(
        title='Seleccionar CSV de Binance P2P',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta_csv:
        print("Cancelado.")
        return
    
    print("\n[2/2] Selecciona el archivo de clusters (traders_features_completo.csv)")
    print("      (Presiona Cancelar si no tienes el archivo)")
    
    ruta_clusters = filedialog.askopenfilename(
        title='Seleccionar archivo de clusters (opcional)',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta_clusters:
        print("Sin archivo de clusters. Se usara segmentacion Binance (Profession/Mass).")
        ruta_clusters = None
    
    analisis = AnalisisHeterogeneidadClusters(
        ruta_csv=ruta_csv,
        ruta_clusters=ruta_clusters
    )
    analisis.ejecutar()


if __name__ == "__main__":
    main()

"""
================================================================================
EDA_VISUALIZACIONES_AVANZADAS.PY - Visualizaciones Avanzadas
================================================================================
Genera gráficos avanzados del mercado P2P:
- Chord Diagram: Flujos entre segmentos/tipos
- Parallel Sets: Relaciones entre variables categóricas
- Stream Graph: Evolución temporal por categoría
- Sunburst Diagram: Jerarquías de mercado
- Treemap: Concentración de mercado

Requiere: plotly, squarify, matplotlib

Autor: Para tesis de maestría en finanzas
Fecha: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import os
import sys
import gc
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
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

# Verificar librerías opcionales
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_DISPONIBLE = True
except ImportError:
    PLOTLY_DISPONIBLE = False
    print("ADVERTENCIA: plotly no instalado. Algunos graficos no estaran disponibles.")
    print("Instalar con: pip install plotly")

try:
    import squarify
    SQUARIFY_DISPONIBLE = True
except ImportError:
    SQUARIFY_DISPONIBLE = False
    print("ADVERTENCIA: squarify no instalado. Treemap matplotlib no disponible.")
    print("Instalar con: pip install squarify")

CHUNK_SIZE = 500_000


class VisualizacionesAvanzadas:
    """Genera visualizaciones avanzadas del mercado P2P."""
    
    def __init__(self, ruta_csv, ruta_clusters=None, directorio_salida=None):
        self.ruta_csv = ruta_csv
        self.ruta_clusters = ruta_clusters
        
        if directorio_salida:
            self.directorio_salida = directorio_salida
        else:
            self.directorio_salida = os.path.dirname(ruta_csv)
        
        self.dir_graficos = os.path.join(self.directorio_salida, 'visualizaciones_avanzadas')
        os.makedirs(self.dir_graficos, exist_ok=True)
        
        self.transacciones = None
        self.clusters = None
        self.reporte = []
    
    def log(self, mensaje):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {mensaje}")
        self.reporte.append(f"[{ts}] {mensaje}")
    
    # ========================================================================
    # FASE 1: CARGA DE DATOS
    # ========================================================================
    def fase1_cargar_datos(self):
        """Carga transacciones y clusters."""
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
        
        # Preparar variables
        df = self.transacciones
        df['fecha'] = pd.to_datetime(df['time'].dt.date)
        df['mes'] = df['time'].dt.to_period('M').astype(str)
        df['semana'] = df['time'].dt.to_period('W').astype(str)
        df['hora'] = df['time'].dt.hour
        df['dia_semana'] = df['time'].dt.day_name()
        
        self.log(f"Transacciones cargadas: {len(df):,}")
        
        # Cargar clusters si existe
        if self.ruta_clusters and os.path.exists(self.ruta_clusters):
            self.log(f"\nCargando clusters desde: {self.ruta_clusters}")
            self.clusters = pd.read_csv(self.ruta_clusters)
            
            # Buscar columna de tipo
            for col in ['tipo_final', 'cluster', 'tipo']:
                if col in self.clusters.columns:
                    self.cluster_col = col
                    break
            else:
                self.cluster_col = None
            
            if self.cluster_col:
                # Merge con transacciones
                self.transacciones = self.transacciones.merge(
                    self.clusters[['userNo', self.cluster_col]],
                    on='userNo',
                    how='left'
                )
                self.transacciones[self.cluster_col] = self.transacciones[self.cluster_col].fillna('Sin Cluster')
                self.log(f"Clusters asignados. Columna: {self.cluster_col}")
        else:
            self.cluster_col = None
            self.log("No se cargaron clusters. Usando segmentacion Binance.")
    
    # ========================================================================
    # FASE 2: TREEMAP - Concentración de Mercado
    # ========================================================================
    def fase2_treemap(self):
        """Genera Treemap de concentración de mercado."""
        self.log("\n" + "="*70)
        self.log("FASE 2: Treemap - Concentracion de Mercado")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Volumen por trader
        vol_trader = df.groupby(['userNo', 'adv.classify']).agg({
            'montoDynamic': 'sum'
        }).reset_index()
        vol_trader.columns = ['userNo', 'segmento', 'volumen']
        vol_trader = vol_trader.sort_values('volumen', ascending=False)
        
        # Top 50 traders + "Otros"
        top_n = 50
        top_traders = vol_trader.head(top_n).copy()
        otros_vol = vol_trader.iloc[top_n:]['volumen'].sum()
        
        if otros_vol > 0:
            otros = pd.DataFrame({
                'userNo': ['Otros'],
                'segmento': ['mass'],
                'volumen': [otros_vol]
            })
            top_traders = pd.concat([top_traders, otros], ignore_index=True)
        
        # ================================================================
        # TREEMAP CON MATPLOTLIB + SQUARIFY
        # ================================================================
        if SQUARIFY_DISPONIBLE:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            sizes = top_traders['volumen'].values
            labels = []
            for i, row in top_traders.iterrows():
                pct = row['volumen'] / vol_trader['volumen'].sum() * 100
                if row['userNo'] == 'Otros':
                    labels.append(f"Otros\n{pct:.1f}%")
                else:
                    labels.append(f"#{i+1}\n{pct:.1f}%")
            
            # Colores por segmento
            colors = ['#3498db' if s == 'profession' else '#e74c3c' for s in top_traders['segmento']]
            
            squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax,
                         text_kwargs={'fontsize': 8})
            
            ax.set_title('Treemap: Concentracion de Mercado por Trader\n(Azul=Profession, Rojo=Mass)', 
                        fontsize=14)
            ax.axis('off')
            
            # Leyenda
            legend_patches = [
                mpatches.Patch(color='#3498db', label='Profession'),
                mpatches.Patch(color='#e74c3c', label='Mass')
            ]
            ax.legend(handles=legend_patches, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir_graficos, '01_treemap_concentracion.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log("Treemap (matplotlib) generado")
        
        # ================================================================
        # TREEMAP INTERACTIVO CON PLOTLY
        # ================================================================
        if PLOTLY_DISPONIBLE:
            # Preparar datos jerárquicos
            vol_total = vol_trader['volumen'].sum()
            
            # Por segmento y tipo (si hay clusters)
            if self.cluster_col:
                df_hier = df.groupby(['adv.classify', self.cluster_col])['montoDynamic'].sum().reset_index()
                df_hier.columns = ['segmento', 'tipo', 'volumen']
                
                fig = px.treemap(
                    df_hier,
                    path=['segmento', 'tipo'],
                    values='volumen',
                    color='volumen',
                    color_continuous_scale='Blues',
                    title='Treemap: Jerarquia Segmento -> Tipo de Agente'
                )
            else:
                # Solo por segmento y top traders
                top_traders['label'] = top_traders.apply(
                    lambda x: x['userNo'][:8] + '...' if len(str(x['userNo'])) > 8 else str(x['userNo']), 
                    axis=1
                )
                
                fig = px.treemap(
                    top_traders,
                    path=['segmento', 'label'],
                    values='volumen',
                    color='volumen',
                    color_continuous_scale='Blues',
                    title='Treemap: Concentracion por Segmento y Trader'
                )
            
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            fig.write_html(os.path.join(self.dir_graficos, '01_treemap_interactivo.html'))
            
            self.log("Treemap interactivo (Plotly) generado")
    
    # ========================================================================
    # FASE 3: SUNBURST - Jerarquía del Mercado
    # ========================================================================
    def fase3_sunburst(self):
        """Genera Sunburst de jerarquía del mercado."""
        self.log("\n" + "="*70)
        self.log("FASE 3: Sunburst - Jerarquia del Mercado")
        self.log("="*70)
        
        if not PLOTLY_DISPONIBLE:
            self.log("Plotly no disponible. Saltando Sunburst.")
            return
        
        df = self.transacciones.copy()
        
        # Determinar jerarquía
        if self.cluster_col:
            # Segmento -> Tipo -> Operación
            df_hier = df.groupby(['adv.classify', self.cluster_col, 'adv.tradeType']).agg({
                'montoDynamic': 'sum',
                'userNo': 'nunique'
            }).reset_index()
            df_hier.columns = ['segmento', 'tipo', 'operacion', 'volumen', 'traders']
            
            fig = px.sunburst(
                df_hier,
                path=['segmento', 'tipo', 'operacion'],
                values='volumen',
                color='traders',
                color_continuous_scale='Viridis',
                title='Sunburst: Segmento -> Tipo de Agente -> Operacion'
            )
        else:
            # Segmento -> Operación -> Día
            df_hier = df.groupby(['adv.classify', 'adv.tradeType', 'dia_semana']).agg({
                'montoDynamic': 'sum'
            }).reset_index()
            df_hier.columns = ['segmento', 'operacion', 'dia', 'volumen']
            
            fig = px.sunburst(
                df_hier,
                path=['segmento', 'operacion', 'dia'],
                values='volumen',
                color='volumen',
                color_continuous_scale='Blues',
                title='Sunburst: Segmento -> Tipo Operacion -> Dia'
            )
        
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
        fig.write_html(os.path.join(self.dir_graficos, '02_sunburst_jerarquia.html'))
        
        # También generar imagen estática
        fig.write_image(os.path.join(self.dir_graficos, '02_sunburst_jerarquia.png'), 
                       width=1000, height=800, scale=2)
        
        self.log("Sunburst generado")
    
    # ========================================================================
    # FASE 4: STREAM GRAPH - Evolución Temporal
    # ========================================================================
    def fase4_stream_graph(self):
        """Genera Stream Graph de evolución temporal."""
        self.log("\n" + "="*70)
        self.log("FASE 4: Stream Graph - Evolucion Temporal")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Determinar categoría para stream
        if self.cluster_col:
            cat_col = self.cluster_col
        else:
            cat_col = 'adv.classify'
        
        # Agregar por semana y categoría
        stream_data = df.groupby(['semana', cat_col])['montoDynamic'].sum().unstack(fill_value=0)
        
        # ================================================================
        # STREAM GRAPH CON MATPLOTLIB
        # ================================================================
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = range(len(stream_data))
        categories = stream_data.columns.tolist()
        
        # Colores
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        # Stackplot para simular stream
        ax.stackplot(x, *[stream_data[cat].values / 1e6 for cat in categories],
                    labels=categories, colors=colors, alpha=0.8)
        
        # Configurar ejes
        n_ticks = min(12, len(stream_data))
        tick_positions = np.linspace(0, len(stream_data)-1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([stream_data.index[i] for i in tick_positions], rotation=45, ha='right')
        
        ax.set_xlabel('Semana')
        ax.set_ylabel('Volumen (Millones USDT)')
        ax.set_title(f'Stream Graph: Evolucion del Volumen por {cat_col}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '03_stream_graph.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("Stream Graph (matplotlib) generado")
        
        # ================================================================
        # STREAM GRAPH INTERACTIVO CON PLOTLY
        # ================================================================
        if PLOTLY_DISPONIBLE:
            # Preparar datos long format
            stream_long = stream_data.reset_index().melt(
                id_vars='semana', 
                var_name='categoria', 
                value_name='volumen'
            )
            
            fig = px.area(
                stream_long,
                x='semana',
                y='volumen',
                color='categoria',
                title=f'Stream Graph Interactivo: Volumen por {cat_col}',
                labels={'volumen': 'Volumen (USDT)', 'semana': 'Semana'}
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                legend_title_text='Categoria',
                hovermode='x unified'
            )
            
            fig.write_html(os.path.join(self.dir_graficos, '03_stream_graph_interactivo.html'))
            
            self.log("Stream Graph interactivo (Plotly) generado")
    
    # ========================================================================
    # FASE 5: PARALLEL SETS - Relaciones Categóricas
    # ========================================================================
    def fase5_parallel_sets(self):
        """Genera Parallel Sets (Sankey categórico)."""
        self.log("\n" + "="*70)
        self.log("FASE 5: Parallel Sets - Relaciones Categoricas")
        self.log("="*70)
        
        if not PLOTLY_DISPONIBLE:
            self.log("Plotly no disponible. Saltando Parallel Sets.")
            return
        
        df = self.transacciones.copy()
        
        # Crear categorías adicionales
        df['tamano_ticket'] = pd.cut(
            df['montoDynamic'],
            bins=[0, 100, 500, 1000, 5000, np.inf],
            labels=['Micro (<100)', 'Pequeno (100-500)', 'Mediano (500-1K)', 
                   'Grande (1K-5K)', 'Muy Grande (>5K)']
        )
        
        df['horario'] = pd.cut(
            df['hora'],
            bins=[-1, 6, 12, 18, 24],
            labels=['Madrugada (0-6)', 'Manana (6-12)', 'Tarde (12-18)', 'Noche (18-24)']
        )
        
        df['es_fds'] = df['time'].dt.dayofweek >= 5
        df['dia_tipo'] = df['es_fds'].map({True: 'Fin de Semana', False: 'Entre Semana'})
        
        # Determinar dimensiones
        if self.cluster_col:
            dimensions = ['adv.classify', self.cluster_col, 'adv.tradeType', 'tamano_ticket']
            dim_labels = ['Segmento', 'Tipo Agente', 'Operacion', 'Tamano Ticket']
        else:
            dimensions = ['adv.classify', 'adv.tradeType', 'dia_tipo', 'tamano_ticket']
            dim_labels = ['Segmento', 'Operacion', 'Dia', 'Tamano Ticket']
        
        # Muestra para velocidad
        df_sample = df.sample(min(50000, len(df)), random_state=42)
        
        # Crear Parallel Categories
        fig = px.parallel_categories(
            df_sample,
            dimensions=dimensions,
            color='montoDynamic',
            color_continuous_scale='Viridis',
            labels={d: l for d, l in zip(dimensions, dim_labels)},
            title='Parallel Sets: Flujo de Transacciones por Categorias'
        )
        
        fig.update_layout(margin=dict(t=80, l=50, r=50, b=50))
        fig.write_html(os.path.join(self.dir_graficos, '04_parallel_sets.html'))
        
        self.log("Parallel Sets generado")
    
    # ========================================================================
    # FASE 6: CHORD DIAGRAM - Flujos entre Categorías
    # ========================================================================
    def fase6_chord_diagram(self):
        """Genera Chord Diagram de flujos."""
        self.log("\n" + "="*70)
        self.log("FASE 6: Chord Diagram - Flujos entre Categorias")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # Para Chord necesitamos flujos bidireccionales
        # Usaremos: Categoría origen (comprador) -> Categoría destino (vendedor)
        # Aproximación: por hora del día
        
        # Crear matriz de flujos por hora
        df['hora_cat'] = pd.cut(
            df['hora'],
            bins=[0, 6, 12, 18, 24],
            labels=['Madrugada', 'Manana', 'Tarde', 'Noche'],
            include_lowest=True
        )
        
        # Matriz de transición (simplificada por tipo de operación y hora)
        if self.cluster_col:
            # Flujos entre tipos de agentes
            categorias = df[self.cluster_col].unique()
            
            # Simular flujos: compradores de un tipo, vendedores de otro
            compras = df[df['adv.tradeType'] == 'compra'].groupby(self.cluster_col)['montoDynamic'].sum()
            ventas = df[df['adv.tradeType'] == 'venta'].groupby(self.cluster_col)['montoDynamic'].sum()
            
            # Crear matriz de flujos proporcional
            n = len(categorias)
            matriz = np.zeros((n, n))
            
            for i, cat_i in enumerate(categorias):
                for j, cat_j in enumerate(categorias):
                    if i != j:
                        # Flujo proporcional a compras de i y ventas de j
                        if cat_i in compras.index and cat_j in ventas.index:
                            matriz[i, j] = compras[cat_i] * ventas[cat_j] / df['montoDynamic'].sum()
            
            # Normalizar
            matriz = matriz / matriz.sum() * 100
        else:
            # Flujos entre segmentos y horarios
            categorias = ['Prof-Manana', 'Prof-Tarde', 'Prof-Noche', 
                         'Mass-Manana', 'Mass-Tarde', 'Mass-Noche']
            n = len(categorias)
            matriz = np.random.rand(n, n) * 10  # Placeholder
            np.fill_diagonal(matriz, 0)
        
        # ================================================================
        # CHORD DIAGRAM CON MATPLOTLIB (Aproximación con Sankey)
        # ================================================================
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Crear visualización circular simplificada
        n_cats = len(categorias) if self.cluster_col else 6
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False)
        
        # Dibujar nodos
        radius = 1
        node_size = 0.15
        
        for i, (angle, cat) in enumerate(zip(angles, categorias)):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Nodo
            circle = plt.Circle((x, y), node_size, color=plt.cm.tab10(i/n_cats), alpha=0.8)
            ax.add_patch(circle)
            
            # Etiqueta
            label_radius = radius + 0.25
            ax.text(label_radius * np.cos(angle), label_radius * np.sin(angle), 
                   str(cat), ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Dibujar conexiones (arcos)
        if self.cluster_col and matriz.sum() > 0:
            for i in range(n_cats):
                for j in range(n_cats):
                    if i != j and matriz[i, j] > 0.5:  # Solo flujos significativos
                        x1 = radius * np.cos(angles[i])
                        y1 = radius * np.sin(angles[i])
                        x2 = radius * np.cos(angles[j])
                        y2 = radius * np.sin(angles[j])
                        
                        # Curva bezier simplificada
                        alpha = min(0.8, matriz[i, j] / 10)
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                   arrowprops=dict(arrowstyle='->', color=plt.cm.tab10(i/n_cats),
                                                  alpha=alpha, connectionstyle='arc3,rad=0.3',
                                                  linewidth=matriz[i, j]/2))
        
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Chord Diagram: Flujos entre Tipos de Agentes\n(Grosor proporcional al volumen)', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '05_chord_diagram.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("Chord Diagram (matplotlib) generado")
        
        # ================================================================
        # CHORD DIAGRAM INTERACTIVO CON PLOTLY (Sankey)
        # ================================================================
        if PLOTLY_DISPONIBLE:
            # Crear Sankey como aproximación de Chord
            df_flows = df.groupby(['adv.classify', 'adv.tradeType']).agg({
                'montoDynamic': 'sum'
            }).reset_index()
            
            # Nodos
            if self.cluster_col:
                tipos = df[self.cluster_col].unique().tolist()
                operaciones = ['Compra', 'Venta']
                all_nodes = tipos + operaciones
            else:
                all_nodes = ['Profession', 'Mass', 'Compra', 'Venta']
            
            node_indices = {node: i for i, node in enumerate(all_nodes)}
            
            # Links
            source = []
            target = []
            value = []
            
            for _, row in df_flows.iterrows():
                seg = row['adv.classify']
                op = 'Compra' if row['adv.tradeType'] == 'compra' else 'Venta'
                
                if seg in node_indices and op in node_indices:
                    source.append(node_indices[seg])
                    target.append(node_indices[op])
                    value.append(row['montoDynamic'] / 1e6)
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=all_nodes,
                    color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(all_nodes)]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])
            
            fig.update_layout(
                title='Diagrama Sankey: Flujos Segmento -> Tipo de Operacion',
                font_size=12
            )
            
            fig.write_html(os.path.join(self.dir_graficos, '05_sankey_flujos.html'))
            
            self.log("Sankey Diagram (Plotly) generado")
    
    # ========================================================================
    # FASE 7: GRÁFICOS ADICIONALES
    # ========================================================================
    def fase7_graficos_adicionales(self):
        """Genera gráficos adicionales complementarios."""
        self.log("\n" + "="*70)
        self.log("FASE 7: Graficos Adicionales")
        self.log("="*70)
        
        df = self.transacciones.copy()
        
        # ================================================================
        # RIDGELINE PLOT - Distribución por día
        # ================================================================
        fig, ax = plt.subplots(figsize=(14, 10))
        
        dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_esp = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
        colors = plt.cm.viridis(np.linspace(0, 1, 7))
        
        for i, (dia, dia_esp, color) in enumerate(zip(dias, dias_esp, colors)):
            data = df[df['dia_semana'] == dia]['montoDynamic']
            data = data[data < data.quantile(0.99)]  # Eliminar outliers extremos
            
            # KDE
            from scipy.stats import gaussian_kde
            if len(data) > 100:
                kde = gaussian_kde(data)
                x_range = np.linspace(0, data.quantile(0.95), 200)
                y = kde(x_range)
                
                # Escalar y desplazar
                y_scaled = y / y.max() * 0.8
                y_offset = i
                
                ax.fill_between(x_range, y_offset, y_offset + y_scaled, alpha=0.7, color=color)
                ax.plot(x_range, y_offset + y_scaled, color='black', linewidth=0.5)
        
        ax.set_yticks(range(7))
        ax.set_yticklabels(dias_esp)
        ax.set_xlabel('Monto de Transaccion (USDT)')
        ax.set_title('Ridgeline Plot: Distribucion de Tickets por Dia de Semana')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_graficos, '06_ridgeline_dias.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("Ridgeline Plot generado")
        
        # ================================================================
        # BUBBLE CHART - Volumen x Transacciones x Traders
        # ================================================================
        if PLOTLY_DISPONIBLE:
            # Por día y segmento
            bubble_data = df.groupby(['dia_semana', 'adv.classify']).agg({
                'montoDynamic': 'sum',
                'userNo': ['nunique', 'count']
            }).reset_index()
            bubble_data.columns = ['dia', 'segmento', 'volumen', 'traders', 'transacciones']
            
            # Orden de días
            dia_orden = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                        'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            bubble_data['dia_num'] = bubble_data['dia'].map(dia_orden)
            
            fig = px.scatter(
                bubble_data,
                x='dia_num',
                y='volumen',
                size='transacciones',
                color='segmento',
                hover_name='dia',
                title='Bubble Chart: Volumen vs Dia (tamano = transacciones)',
                labels={'dia_num': 'Dia de Semana', 'volumen': 'Volumen (USDT)'}
            )
            
            fig.update_xaxes(
                ticktext=['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'],
                tickvals=list(range(7))
            )
            
            fig.write_html(os.path.join(self.dir_graficos, '07_bubble_chart.html'))
            
            self.log("Bubble Chart generado")
    
    # ========================================================================
    # FASE 8: REPORTE
    # ========================================================================
    def fase8_reporte(self):
        """Genera reporte final."""
        self.log("\n" + "="*70)
        self.log("FASE 8: Reporte Final")
        self.log("="*70)
        
        reporte_path = os.path.join(self.directorio_salida, 'visualizaciones_avanzadas_reporte.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.reporte))
        
        self.log(f"\nReporte guardado: {reporte_path}")
        self.log(f"Graficos guardados en: {self.dir_graficos}")
        
        # Listar archivos generados
        self.log("\nArchivos generados:")
        for f in sorted(os.listdir(self.dir_graficos)):
            self.log(f"  - {f}")
    
    # ========================================================================
    # EJECUTAR
    # ========================================================================
    def ejecutar(self):
        """Ejecuta generación completa de visualizaciones."""
        inicio = datetime.now()
        self.log(f"Inicio: {inicio}")
        self.log("="*70)
        self.log("VISUALIZACIONES AVANZADAS")
        self.log("="*70)
        
        try:
            self.fase1_cargar_datos()
            self.fase2_treemap()
            self.fase3_sunburst()
            self.fase4_stream_graph()
            self.fase5_parallel_sets()
            self.fase6_chord_diagram()
            self.fase7_graficos_adicionales()
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
    from tkinter import Tk, filedialog, messagebox
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("="*70)
    print("VISUALIZACIONES AVANZADAS")
    print("="*70)
    print("\nSelecciona el archivo CSV de transacciones...")
    
    ruta_csv = filedialog.askopenfilename(
        title='Seleccionar CSV de Transacciones',
        filetypes=[('CSV', '*.csv'), ('All', '*')]
    )
    
    if not ruta_csv:
        print("Cancelado.")
        return
    
    # Preguntar por clusters
    usar_clusters = messagebox.askyesno(
        "Clusters",
        "¿Deseas cargar archivo de clusters?\n(traders_con_clusters.csv)"
    )
    
    ruta_clusters = None
    if usar_clusters:
        ruta_clusters = filedialog.askopenfilename(
            title='Seleccionar CSV de Clusters',
            filetypes=[('CSV', '*.csv'), ('All', '*')]
        )
    
    viz = VisualizacionesAvanzadas(ruta_csv, ruta_clusters)
    viz.ejecutar()


if __name__ == "__main__":
    main()

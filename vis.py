import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class DataFrameAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def plot_route_graph(self, route_id: str):

        # analicemos esto. ¿ por que hay paradas con nodos que no son de la forma i to (i+1) ?
        route_df = self.df[self.df['ruta_id_c'] == route_id].sort_values('orden_en_ruta')

        if route_df.empty:
            print(f"No se encontraron datos para la ruta {route_id}")
            return 

        G = nx.DiGraph()

        for _, row in route_df.iterrows():
            G.add_node(row['orden_en_ruta'],
                       tiempo_parada = row['tiempo_parada'],
                       city=row['addr_city'])
        
        for i in range(len(route_df) - 1):
            source_node = route_df.iloc[i]['orden_en_ruta']
            target_node = route_df.iloc[i+1]['orden_en_ruta']
            duration = route_df.iloc[i]['tiempo_parada']
            G.add_edge(source_node, target_node, weight = duration)

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        node_labels = {node: f"Parada {node}\nT: {G.nodes[node]['tiempo_parada']:.2f}" for node in G.nodes()}

        plt.figure(figsize=(10, 8))
        # TODO: modificar hasta que se vea bien
        edge_label_bbox = {
            'facecolor': 'white', # color de fondo blanco
            'alpha': 0.7,        # 70% de opacidad
            'edgecolor': 'none', # sin borde
            'boxstyle': 'round,pad=0.2' # esquinas redondeadas y un pequeño relleno
        }
        
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=900, alpha=0.9)
    
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=3, arrowsize=22)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},font_size=8, verticalalignment='baseline', bbox=edge_label_bbox)

        plt.title(f"Grafo de la ruta: {route_id}")
        plt.axis('off')
        plt.show()

    def count_city_clusters(self):
        city_counts = self.df['addr_city'].value_counts()
        print('agrupaciones por ciudad')
        print(f"numero total de ciudades unicas {len(city_counts)}")

        # TODO: Quizas sea buena idea eliminar las ciudades de paradas unicas ya que son outliers.
        # no hay informacion suficiente para inferir si se van a comportar bien o no.
        print("\n conteo de paradas por ciudad")
        print(city_counts)
        return city_counts
    
    def remove_single_stop_cities(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        city_counts = self.df['addr_city'].value_counts()
        single_stop_cities = city_counts[city_counts==1].index.tolist()
        df_filtered = self.df[~self.df['addr_city'].isin(single_stop_cities)].copy()
        df_removed = self.df[self.df['addr_city'].isin(single_stop_cities)].copy()

        print("\n--- Limpieza de Ciudades con Paradas Únicas ---")
        print(f"Ciudades con una sola parada a ser eliminadas: {single_stop_cities}")
        print(f"Número de registros eliminados: {len(df_removed)}")
        print(f"Tamaño del DataFrame limpio: {len(df_filtered)}")
        
        return df_filtered, df_removed
    

    # TODO: revisar esta funcion
    def compare_predictions_vs_actual(self, y_true: pd.Series, y_pred: pd.Series):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        print("\n analisis de rendimiento del modelo")

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"R cuadrado (R2) {r2:.4f}")
        print(f"Error absoluto medio {mae:.4f}")
        print(f"Raiz del error cuadratico medio {rmse:.4f}")

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Línea de Predicción Perfecta')
        plt.title('Comparación: Predicciones vs. Valores Reales (Contramuestra)')
        plt.xlabel('Tiempo de Parada Real')
        plt.ylabel('Tiempo de Parada Predicho')
        plt.grid(True)
        plt.legend()
        plt.show()
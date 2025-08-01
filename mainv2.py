import pandas as pd
from data_load import load_data, validate_data, load_counter_data
from data_processor import process_complete_pipeline
from models import GLMGamlogModel, TwoPartModel
from vis import DataFrameAnalyzer
from models import check_short_time_behavior, visualize_time_distribution, visualize_time_distribution_by_city, compare_distributions_by_threshold
# MAIN 1 solo procesa el modelo GLMGamLogModel y el TwoPartModel
# contramuestra

def main():
    try:
        perfiles, paradas, eventos = load_data()
        if not validate_data(perfiles, paradas, eventos):
            print('La validacion fallo')
            return
    
        df_processed = process_complete_pipeline(perfiles, paradas, eventos, bulk_clients_only=True)
        # pendiente: remover ciudades unicas
        # pendiente: analizar con la contramuestra
        print('\n Columnas procesadas', df_processed.columns.tolist())

        analyzer = DataFrameAnalyzer(df_processed)

        routes_with_multiple_stops = df_processed['ruta_id_c'].value_counts()
        routes_with_multiple_stops = routes_with_multiple_stops[routes_with_multiple_stops > 1]

        if not routes_with_multiple_stops.empty:
            sample_route_id = routes_with_multiple_stops.index[0]
            print(f"\n -- visualizando una ruta de ejemplo con multiples paradas {sample_route_id}")
            # TODO: corregir el grafico 
            analyzer.plot_route_graph(sample_route_id)

        else:
            print("\n No se encontraron rutas con mas de una parada para visitar")
        analyzer.count_city_clusters()

        # test 
        df_processed, point_null = analyzer.remove_single_stop_cities()
        print(df_processed.head())
        features = [
            'dia_semana', 'load', 'demand_delivery_priority',
            'start_time_window_at_hora', 'end_time_window_at_hora', 
            'lunch_start_time_window_at_hora', 'lunch_end_time_window_at_hora',
            #'hora_llegada',
            'en_colacion', 'max_load_in_ruta',

            # no debiese usar las latitudes o longitudes sueltas si no la distancia previa
            #'latitud_perfil',
            #'longitud_perfil',
            'n_paradas_en_ruta',
            'carga_acumulada',
            'dist_prev',
            'dist_prev_sq'#, 'load_sq', 'n_paradas_en_ruta_sq'
        ]
        categorical_cols = ['addr_city']
        target_column = 'tiempo_parada'

        # GLM
        print('\n--Entrenando y evaluando GLM Gamma log Model')

        # experimento de visualizaciones
        
        # quizas aca conviene hacer el histograma por ciudad
        #visualize_time_distribution(df_processed, 'tiempo_parada', threshold=12)
        # Construir una visualizacion del threshold
        #compare_distributions_by_threshold(df_processed, 'tiempo_parada', threshold=13)
        #visualize_time_distribution_by_city(df_processed, 'tiempo_parada', threshold=13)
        #check_short_time_behavior(df_processed, 'tiempo_parada', threshold=12)
        
        # flag in here lol
        two_part_model = TwoPartModel(
            features=features, 
            categorical_cols=categorical_cols,
            target_column=target_column,
            threshold_short_time=13
        )

        print(f"\n test ")

        # no hay un codigo que dumpee el dataframe y permita visualizarlo?
        print(df_processed.head())
        df_processed.to_csv('df_processed_snapshot.csv', index=False)

        #two_part_model.fit(df_processed)
        
        # pendiente: Describir el modelo de clasificacion
        # hacer histograma del modelo de primera clasificacion
        # hacer histograma por ciudad quizas

        # TODO: construir un summary del two part y ver que caracteristicas fueron relevantes
        # TODO: construir el summary del GLM en el two part model
        #two_part_model.plot_diagnostics()
        #two_part_model.plot_feature_importance_logistic_regression()
        
        #######################################################################
        # flag in here
        # im gonna comment in here 

        glm_model = GLMGamlogModel(target_column=target_column, features=features, categorical_cols=categorical_cols)

        glm_model.fit(df_processed)
        glm_model.plot_diagnostics()
        glm_model.summary()

        # hay un error aca
        glm_model.evaluate()
        ##################################################
        # TODO: hay que hacer el dumping del modelo de manera que se analice con la contramuestra
        #if not df_processed.empty:
        sample_data_glm = df_processed.sample(min(5, len(df_processed)))
        glm_predictions = glm_model.predict(sample_data_glm)
        print('\n Predicciones glm de ejemplo:', glm_predictions.tolist())
        
        # debiese hacer una visualizacion de resultados
        # histograma, histograma obtenidos, analisis de residuos
        
        # TODO: analizar con la contramuestra
        print(f"\n iniciando analisis de contramuestra")

        counter_perfiles, counter_paradas, counter_eventos = load_counter_data()
        counter_df = process_complete_pipeline(
            counter_perfiles, counter_paradas, counter_eventos, bulk_clients_only=True
        )

        # TODO: terminar el analisis de la contramuestra
        print('inicializando counter analyzer')
        counter_analyzer = DataFrameAnalyzer(counter_df)
        counter_df_ready, _ = counter_analyzer.remove_single_stop_cities()

        counter_df_ready = counter_df_ready[counter_df_ready['tiempo_parada'] > 0]
        y_true_counter = counter_df_ready['tiempo_parada']

        print(f"Flag in here")
        if not counter_df_ready.empty:
            y_pred_counter = two_part_model.predict(counter_df_ready)
            counter_analyzer.compare_predictions_vs_actual(y_true_counter, y_pred_counter)
        else:
            print('El data frame de la contramuestra esta vacio')
        

    except Exception as e:
        print(f"Ocurrio un error en el flujo principal {e}")
    

if __name__ == '__main__':
    main()
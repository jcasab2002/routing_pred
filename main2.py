import pandas as pd
from data_load import load_data, validate_data
from data_processor import process_complete_pipeline
from models import GLMGamlogModel, QuantileRegressionModel, XGBoostRegModel

def main():
    try:
        perfiles, paradas, eventos = load_data()

        if not validate_data(perfiles, paradas, eventos):
            print('La validacion fallo')
            return
    
        df_processed = process_complete_pipeline(perfiles, paradas, eventos, bulk_clients_only=True)
        print('\n Columnas procesadas', df_processed.columns.tolist())
        
        # test 
        print(df_processed.head())
        features = [
            'dia_semana', 'load', 'demand_delivery_priority',
            'start_time_window_at_hora', 'end_time_window_at_hora', 
            'lunch_start_time_window_at_hora', 'lunch_end_time_window_at_hora',
            'hora_llegada', 'en_colacion', 'max_load_in_ruta',
            'latitud_perfil', 'longitud_perfil', 'n_paradas_en_ruta',
            'carga_acumulada', 'dist_prev'
        ]
        categorical_cols = ['addr_city']
        target_column = 'tiempo_parada'

        #################################################
        # Quantile regression model
        print('\n entrenando y evaluando quantile reg model')
        qr_model = QuantileRegressionModel(target_column=target_column, features=features, categorical_cols=categorical_cols)
        qr_model.fit(df_processed)
        qr_model.analyze_multicollinearity(plot_heatmap=True, threshold=0.7) 
        qr_model.summary(quantile=0.5)

        if not df_processed.empty:
            sample_data_qr = df_processed.sample(min(5, len(df_processed)))
            qr_predictions = qr_model.predict(sample_data_qr)
            print('\n Predicciones quantile reg de ejemplo', qr_predictions.tolist())
        qr_model.plot_coef_trends()

        print('\n entrenando y evaluando xgboost reg model')
        xgb_model = XGBoostRegModel(target_column=target_column, features= features, categorical_cols = categorical_cols)
        xgb_model.fit(df_processed)
        xgb_model.evaluate()

        if not df_processed.empty:
            sample_data_xgb = df_processed.sample(min(5, len(df_processed)))
            xgb_predictions = xgb_model.predict(sample_data_xgb)
            print('\n Predicciones XGboost de ejemplo', xgb_predictions.tolist())

    except Exception as e:
        print(f"Ocurrio un error en el flujo principal {e}")
    

if __name__ == '__main__':
    main()
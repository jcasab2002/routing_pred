import pandas as pd
from data_load import load_data, validate_data
from data_processor import process_complete_pipeline
import pandas as pd
import numpy as np
import torch
from tflow import TimeDurationVAEPytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline # Nueva importación

def main_vae_pytorch():
    try:
        perfiles, paradas, eventos = load_data()
        
        if not validate_data(perfiles, paradas, eventos):
            print('La validacion fallo')
            return
            
        df_processed = process_complete_pipeline(perfiles, paradas, eventos, bulk_clients_only=True)
        print('\nColumnas procesadas', df_processed.columns.tolist())

        features = [
            'dia_semana', 'load', 'demand_delivery_priority',
            'start_time_window_at_hora', 'end_time_window_at_hora', 
            'lunch_start_time_window_at_hora', 'lunch_end_time_window_at_hora',
            'hora_llegada', 'en_colacion', 'max_load_in_ruta',
            'latitud_perfil', 'longitud_perfil', 'n_paradas_en_ruta',
            'carga_acumulada', 'dist_prev',
            'dist_prev_sq', 'load_sq', 'n_paradas_en_ruta_sq'
        ]
        categorical_cols = ['addr_city']
        target_column = 'tiempo_parada'
        
        X = df_processed[features + categorical_cols].copy()
        y = df_processed[target_column].copy()
        
        X = X[y > 0]
        y = y[y > 0]
        
        # OHE para variables categóricas
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Separar los datos antes de pre-procesar para evitar data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- NUEVA SECCIÓN: Crear un pipeline de pre-procesamiento ---
        preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Ajustar el pipeline SÓLO en los datos de entrenamiento
        X_train_scaled = preprocessing_pipeline.fit_transform(X_train)
        
        # Transformar los datos de entrenamiento y prueba usando el pipeline ajustado
        X_test_scaled = preprocessing_pipeline.transform(X_test)
        
        # Convertir a DataFrame con los nombres de columnas correctos
        imputed_feature_names = preprocessing_pipeline.named_steps['imputer'].get_feature_names_out(input_features=X_train.columns)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=imputed_feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=imputed_feature_names, index=X_test.index)

        
        input_dim = X_train_scaled.shape[1]
        vae_model = TimeDurationVAEPytorch(input_dim=input_dim, latent_dim=10, hidden_dim=128)

        # quizas debiesemos ajustar el parametro de epochs
        vae_model.fit(X_train_scaled, y_train, epochs=10)
        
        if not X_test_scaled.empty:
            print('\n--- Visualizando resultados del VAE en el conjunto de prueba ---')
            vae_model.plot_predictions_vs_actual(X_test_scaled, y_test)
            
            X_sample_test = X_test_scaled.sample(min(4, len(X_test_scaled)))
            vae_model.plot_learned_distributions(X_sample_test)
            
    except Exception as e:
        print(f"Ocurrio un error en el flujo principal: {e}")

if __name__ == '__main__':
    main_vae_pytorch()
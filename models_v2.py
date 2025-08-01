# version de modelos de prueba pasados que finalmente fue deprecada!!!
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor 

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix



class QuantileRegressionModel:
    def __init__(self, target_column = 'tiempo_parada', features=None, categorical_cols = None):
        self.target_column = target_column
        self.features = features if features is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.formula_str = None
        self.imputer = None
        self.final_features = None

    def _clean_column_name(self, col_name: str) -> str:
        import unicodedata
        c_norm = unicodedata.normalize('NFKD', col_name).encode('ascii', 'ignore').decode('ascii')
        c_norm = c_norm.replace(' ', '_').replace('.', '').replace('-', '_').replace('(', '').replace(')', '')
        return c_norm

    # codigo potencialmente corregido
    def prepare_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """prepara los datos para el modelado de regresion por cuantiles"""
        print("\n--Inicio: prepare data ---")

        print(f"DEBUG: df_raw shape al inicio: {df_raw.shape}")
        print(f"DEBUG: df_raw columns al inicio: {df_raw.columns.tolist()}")

        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError('la entrada df_raw debe ser un objeto')
        
        base_cols_set = set(self.features + self.categorical_cols + [self.target_column])
        existing_base_cols = [col for col in base_cols_set if col in df_raw.columns]

        if self.target_column not in existing_base_cols:
            raise ValueError(f"la columna objetivo {self.target_column} no se encuentra en el DataFrame inicial")
        
        df_base = df_raw[existing_base_cols].copy()
        print(f"Debug: df_base shape despues de seleccionar columnas {df_base.shape}")
        print(f"Debug: df_base columns despues de seleccionar columnas: {df_base.columns.tolist()}")

        if df_base[self.target_column].isnull().any():
            print(f"Debug alerta: NaNs en target_column {self.target_column} antes de filtrar")
        
        df_base = df_base[df_base[self.target_column] > 0]
        print(f"Debug: df_base shape despues de filtrar target > 0: {df_base.shape}")
        if df_base.empty:
            print('Error: df_base esta vacio')
        
        cols_to_dummy = [col for col in self.categorical_cols if col in df_base.columns]
        df_processed = pd.get_dummies(df_base, columns = cols_to_dummy, drop_first=True)
        print(f"Debug: df_processed shape despues de get_dummies {df_processed.shape}")
        print(f"debug: df_processed columns despues de get_dummies : {df_processed.columns.tolist()}")

        df_processed.columns = [self._clean_column_name(c) for c in df_processed.columns]
        original_target_cleaned = self._clean_column_name(self.target_column)
        if original_target_cleaned != self.target_column and original_target_cleaned in df_processed.columns:
            self.target_column = original_target_cleaned
        print(f"Debug: df_processed columns despues de limpiar nombres: {df_processed.columns.tolist()}")

        duplicate_columns = df_processed.columns[df_processed.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"Error: Columnas duplicadas encontradas despuesde la limpieza de nombres {duplicate_columns}")
            df_processed = df_processed.loc[:,~df_processed.columns.duplicated()]
            print("DEBUG: Columnas duplicadas eliminadas/manejadas. Nuevo df_processed shape:", df_processed.shape)
            print("DEBUG: Nuevas columnas:", df_processed.columns.tolist())

        for col_to_drop in ['profile_nid', 'patente']:
            cleaned_col_to_drop = self._clean_column_name(col_to_drop)
            if cleaned_col_to_drop in df_processed.columns:
                df_processed = df_processed.drop(columns=[cleaned_col_to_drop])
                print(f"Debug: columna {cleaned_col_to_drop} eliminada")

        print(f"debug: df_processed shape despues de eliminar columnas especificas {df_processed.shape}")
        print(f"debug: df_processed shape despues de eliminar columnas especificas {df_processed.columns.tolist()}")

        # debugging in progress
        print(df_processed.head())
        for col in df_processed.columns:
            if df_processed[col].dtype == 'bool':
                df_processed[col] = df_processed[col].astype(int)

        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        print(f"procesamiento de columnas realizado")

        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        print(f"Debug: df_processed shape despues de convertir a numerico y manejar inf: {df_processed.shape}")

        y = df_processed[self.target_column]
        X = df_processed.drop(columns = self.target_column)
        print(f"Debug: X shape despues de separar y: {X.shape}")
        print(f"Debug: y shape despues de separar y: {y.shape}")

        X = X.dropna(axis=1, how='all')
        X = X.loc[:, X.nunique(dropna=True)>1]
        print(f"Debug: X shape despues de eliminar columnas constantes / all-NaN: {X.shape}")

        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy='median')
        X_index = X.index

        if X.shape[1] == 0:
            print("error critico: X no tiene columnas")
            raise ValueError('no hay caracteristicas validas')
        
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns = X.columns, index=X_index)
        print(f"Debug X_imputed shape despues de imputacion {X_imputed.shape}")
        print(f"Debug X imputed columns despues de imputacion {X_imputed.columns.tolist()}" )

        mask = y.notna() & X_imputed.notna().all(axis=1)
        X_clean = X_imputed.loc[mask]
        y_clean = y.loc[mask]
        print(f"Debug X clean shape despues de manejar NaNs finales {X_clean.shape}")
        print(f"Debug: y_clean shape despues de manejar nans finales {y_clean.shape}")

        if X_clean.empty or y_clean.empty:
            print("ERROR: X_clean o y_clean están vacíos después de la limpieza final. Revisa tus datos y filtros.")
            raise ValueError("El DataFrame final está vacío después de la limpieza de NaNs.")


        df_final = pd.concat([y_clean, X_clean], axis=1) # El DataFrame final para el split y el modelo
        print(f"DEBUG: df_final shape final: {df_final.shape}")
        print(f"DEBUG: df_final columns final: {df_final.columns.tolist()}")
        
        self.final_features = X_clean.columns.tolist() 
        self.formula_str = f"{self.target_column} ~ " + " + ".join(self.final_features)

        print(f"DEBUG: Formula final construida: {self.formula_str}")
        print(f"len(df_final): {len(df_final)}")
        print("--- FIN: prepare_data ---")
        return df_final
  
    def fit(self, df_raw):
        # llamamos al metodo de preparacion
        print(f"hola")
        df_clean = self.prepare_data(df_raw)

        print(f"DEBUG: df_clean shape antes de split: {df_clean.shape}")
        print(f"DEBUG: df_clean columns antes de split: {df_clean.columns.tolist()}")
        print(f"DEBUG: target_column en fit: {self.target_column}")


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df_clean.drop(columns = self.target_column), df_clean[self.target_column], train_size=0.8, test_size=0.2, random_state=42
        )

        train_df_for_qr = pd.concat([self.y_train, self.X_train], axis=1)

        print(f"DEBUG: X_train shape: {self.X_train.shape}")
        print(f"DEBUG: y_train shape: {self.y_train.shape}")
        print(f"DEBUG: train_df_for_qr shape: {train_df_for_qr.shape}")
        print(f"DEBUG: train_df_for_qr columns: {train_df_for_qr.columns.tolist()}")
        print(f"DEBUG: self.formula_str: {self.formula_str}")

        # Verifica si hay NaNs o infinitos en train_df_for_qr
        if train_df_for_qr.isnull().any().any():
            print("ERROR DEBUG: NaNs encontrados en train_df_for_qr antes de ajustar el modelo.")
        if np.isinf(train_df_for_qr.values).any():
            print("ERROR DEBUG: Infinitos encontrados en train_df_for_qr antes de ajustar el modelo.")
            print("\n -- entrenando modelos de regresion por cuantiles ---")

        print(f"Fórmula que se usará para quantreg:\n{self.formula_str}\n")
        print(f"Columnas en train_df_for_qr:\n{train_df_for_qr.columns.tolist()}\n")

        for q in self.quantiles:
            print(f'\n-- ajustando para el cuantil {q}')
            try:
                mod = smf.quantreg(self.formula_str, train_df_for_qr)
                res = mod.fit(q=q)
                self.models[q] = res
                print(f"resumen para cuantil {q}")
                print(res.summary().as_text())
            except Exception as e:
                print(f"Error al ajustar el modelo para el cuantil {q}: {e}")
        
    def summary(self, quantile=0.50):
        if quantile not in self.models:
            raise ValueError(f"Modelo para cuantil {quantile} no encontrado. Ajuste el modelo")
        print(self.models[quantile].summary())

    
    def plot_coef_trends(self):
        import matplotlib.pyplot as plt
        if not self.models:
            print('No hay modelos entrenados')
            return

        params_df = pd.DataFrame()
        for q, res in self.models.items():
            params_df[q] = res.params

        params_df = params_df.drop('const', errors='ignore')
        params_df = params_df.drop('Intercept', errors='ignore')

        params_df.T.plot(figsize=(12, 6), marker='o')
        plt.title('Tendencia de Coeficientes a traves de los cuantiles')
        plt.xlabel('Cuantil')
        plt.ylabel('Valor del coeficiente')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def predict(self, new_data: pd.DataFrame) -> pd.Series :
        if 0.50 not in self.models:
            raise ValueError('Modelo para el cuantil 0.5 no encontrado')
            
        required_cols = list(set(self.features + self.categorical_cols))
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas requeridas para la prediccion faltan {missing_cols}")
        
        temp_df_for_dummies = new_data[required_cols].copy()
        cols_to_dummy_predict = [col for col in self.categorical_cols if col in temp_df_for_dummies.columns]
        temp_df_dummies = pd.get_dummies(temp_df_for_dummies, columns = cols_to_dummy_predict, drop_first=True)

        temp_df_dummies.columns = [self._clean_column_name(c) for c in temp_df_dummies.columns]
        X_predict_aligned = temp_df_dummies.reindex(columns = self.final_features, fill_value=0)

        X_predict_aligned = X_predict_aligned.apply(pd.to_numeric, errors='coerce')
        X_predict_aligned = X_predict_aligned.replace([np.inf, -np.inf], np.nan)

        if self.imputer:
            X_predict_imputed_array = self.imputer.transform(X_predict_aligned)
            X_predict_aligned = pd.DataFrame(X_predict_imputed_array, columns = self.final_features, index=X_predict_aligned.index)
        else:
            raise RuntimeError('imputer no entreando')

        X_predict_aligned = X_predict_aligned.astype('float64')
        predictions = self.models[0.5].predict(X_predict_aligned)
        return predictions
    
    def analyze_multicollinearity(self, plot_heatmap: bool = True, threshold: float = 0.8):
        if self.X_train is None:
            print(f'no entrenado')
            return 
        print('\n --- Analizando multicolinealidad')

        print('Calculando matriz de correlacion...')
        correlation_matrix = self.X_train.corr()
        high_corr_pairs = []
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        for i in range(len(upper_tri.columns)):
            for j in range(i + 1, len(upper_tri.columns)):
                col1 = upper_tri.columns[i]
                col2 = upper_tri.columns[j]
                corr_val = upper_tri.iloc[i, j] # Acceder directamente al valor
                if pd.notna(corr_val) and abs(corr_val) >= threshold:
                    high_corr_pairs.append((col1, col2, corr_val))

        if high_corr_pairs:
            print(f"\n--- PARES DE VARIABLES CON ALTA CORRELACIÓN (>{threshold}) ---")
            for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  - '{col1}' y '{col2}': {corr_val:.4f}")
        else:
            print(f"\nNo se encontraron pares de variables con correlación > {threshold}.")

        if plot_heatmap:
            print("\nGenerando Heatmap de la Matriz de Correlación...")
            plt.figure(figsize=(16, 14)) # Ajusta el tamaño según el número de columnas
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Matriz de Correlación de Variables Predictoras')
            plt.tight_layout()
            plt.show()

        print("\n--- CALCULANDO FACTOR DE INFLACIÓN DE LA VARIANZA (VIF) ---")
        X_for_vif = self.X_train.select_dtypes(include=np.number) # Solo columnas numéricas
        X_for_vif = X_for_vif.replace([np.inf, -np.inf], np.nan)
        X_for_vif = X_for_vif.dropna(axis=1, how='all') # Eliminar columnas totalmente NaN
        X_for_vif = X_for_vif.loc[:, X_for_vif.nunique(dropna=True) > 1] # Eliminar columnas constantes

        if X_for_vif.empty or X_for_vif.shape[1] == 0:
            print("Advertencia: No hay columnas numéricas válidas para calcular VIF después de la limpieza.")
            return

        # Para VIF, necesitamos añadir una constante al intercepto (si el modelo la usa)
        # Esto es más relevante para OLS, pero para la consistencia:
        # X_vif_const = sm.add_constant(X_for_vif, has_constant='add') # Comentar si no usas intercepto forzado

        vif_data = pd.DataFrame()
        vif_data["feature"] = X_for_vif.columns
        # Usa range(X_for_vif.shape[1]) si no añades constante
        vif_data["VIF"] = [variance_inflation_factor(X_for_vif.values, i) for i in range(X_for_vif.shape[1])]
        
        # Ordenar por VIF descendente
        vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
        
        print(vif_data)
        print("\nInterpretación de VIF:")
        print("  - VIF = 1: No hay correlación entre la variable y las otras predictoras.")
        print("  - 1 < VIF < 5: Correlación moderada, generalmente aceptable.")
        print("  - 5 < VIF < 10: Fuerte correlación, puede ser problemático.")
        print("  - VIF >= 10: Multicolinealidad severa, definitivamente es problemático.")

        print("\n--- FIN ANÁLISIS DE MULTICOLINEALIDAD ---")    


class XGBoostRegModel:
    def __init__(self, target_column='tiempo_parada', features = None, categorical_cols = None, xgb_params=None):
        """ inicializa la clase de modelo de regresion xgboost"""
        self.target_column = target_column
        self.features = features if features is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []

        self.xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200, 
            'learning_rate': 0.1,
            'max_depth': 5, 
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42, 
            'n_jobs' : -1
        }

        if xgb_params:
            self.xgb_params.update(xgb_params)

        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.final_feature_columns = None

    def prepare_data(self, df_raw):
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError('la entrada df raw debe ser un objeto pandas')
        
        base_cols_set = set(self.features + self.categorical_cols + [self.target_column])
        existing_base_cols = [col for col in base_cols_set if col in df_raw.columns]

        if self.target_column not in existing_base_cols:
            raise ValueError(f"La columna objetivo {self.target_column}")
        
        df_base = df_raw[existing_base_cols].copy()
        df_base = df_base[df_base[self.target_column] > 0]

        cols_to_dummy = [col for col in self.categorical_cols if col in df_base.columns]
        df_dum = pd.get_dummies(df_base, columns=cols_to_dummy, drop_first=True)

        y = df_dum[self.target_column]
        X = df_dum.drop(columns=self.target_column)

        for col in ['profile_nid', 'patente']:
            if col in X.columns:
                X = X.drop(columns=col)

        X = X.apply(pd.to_numeric, errors='coerce')
        y = y.apply(pd.to_numeric, errors = 'coerce')

        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        X = X.astype('float64')
        y = y.astype('float64')

        print('len(X)=', len(X), 'len(y)=', len(y))
        print('indices iguales', X.index.equals(y.index))

        self.final_feature_columns = X.columns.tolist()

        return X, y
    
    def fit(self, df_raw):
        X, y = self.prepare_data(df_raw)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        print('Modelo XGBoost entrenado exitosamente')

    def evaluate(self):
        """Evalua el modelo entrenado en el conjunto de prueba"""
        if self.model is None:
            raise ValueError('El modelo no ha sido entrenado')
        if self.X_test is None or self.y_test is None:
            raise ValueError('No se han dividido los datos')
        
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        print(f'MSE {mse:.2f}')
        print(f"Coeficiente de det {r2:.2f}")
        return mse, r2
    
    def predict(self, new_data):
        if self.model is None:
            raise ValueError('El modelo no ha sido entrenado')
        
        required_cols = list(set(self.features + self.categorical_cols))
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas requeridas faltan {missing_cols}")
        
        temp_df = new_data[required_cols].copy()
        cols_to_dummy = [col for col in self.categorical_cols if col in temp_df.columns]
        temp_df_dummies = pd.get_dummies(temp_df, columns = cols_to_dummy, drop_first=True)

        X_predict_aligned = pd.DataFrame(columns = self.final_feature_columns)
        X_predict_aligned = pd.concat([X_predict_aligned, temp_df_dummies], ignore_index=True)

        for col in self.final_feature_columns:
            if col not in X_predict_aligned.columns:
                X_predict_aligned[col] = 0

        extra_cols = [col for col in X_predict_aligned.columns if col not in self.final_feature_columns]
        if extra_cols:
            X_predict_aligned = X_predict_aligned.drop(columns = extra_cols)

        X_predict_aligned = X_predict_aligned[self.final_feature_columns]

        X_predict_aligned = X_predict_aligned.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        predictions = self.model.predict(X_predict_aligned)
        return predictions

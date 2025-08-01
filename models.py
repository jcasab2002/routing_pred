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


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred)
    }

def prepare_data_for_modelling(df: pd.DataFrame, target_col: str = 'tiempo_parada',
                               exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    print("preparando datos para modelado")

    if exclude_cols is None:
        exclude_cols = ['ruta_id_c', 'profile_nid', 'patente', 'dia_inicio']
    
    df_clean = df[df[target_col] > 0].copy()

    feature_cols = [col for col in df_clean.columns if col != target_col and col not in exclude_cols]
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"Datos preparados {len(X)} registros, {len(feature_cols)} features")
    return X, y

# Idea: separar el GLM en un modelo two stage
# Idea: primero observar como se comportan los tiempos de llegada

def visualize_time_distribution(df: pd.DataFrame, time_column: str, threshold: int = 60):

    df = df.dropna(subset=[time_column])
    df = df[df[time_column] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(df[time_column], bins=100, kde=False, ax=axes[0])
    axes[0].set_title('Histograma de Tiempos de Parada')
    axes[0].set_xlabel('Tiempo de parada (segundo)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].axvline(threshold, color='red', linestyle='--', label=f'Umbral: {threshold}s')
    axes[0].legend()

    sns.kdeplot(df[time_column], shade=True, ax=axes[1])
    axes[1].set_title('Grafico de densidad de tiempos de parada KDE')
    axes[1].set_xlabel('Tiempo de parada (segundos)')
    axes[1].set_ylabel('Densidad')
    axes[1].axvline(threshold, color='red', linestyle='--', label=f'Umbral: {threshold}s')
    axes[1].legend()

    axes[0].set_xlim(0, threshold + 30)
    axes[1].set_xlim(0, threshold + 30)

    plt.tight_layout()
    plt.show()

def visualize_time_distribution_by_city(df: pd.DataFrame, time_column: str, threshold: int = 60, top_n: int = 5):
    df = df.dropna(subset=[time_column, 'addr_city'])
    df = df[df[time_column] > 0]

    city_counts = df['addr_city'].value_counts()
    top_cities = city_counts.nlargest(top_n).index.tolist()

    if not top_cities:
        print('No se encontraron ciudades para visualizar')
        return
    
    print(f"\n -- visualizando distribuciones de tiempo para las {top_n} ciudades principales --")
    fig, axes = plt.subplots(top_n, 2, figsize=(15, top_n * 5))

    for i, city in enumerate(top_cities):
        city_df = df[df['addr_city'] == city]

        sns.histplot(city_df[time_column], bins=50, kde=False, ax=axes[i, 0])
        axes[i, 0].set_title(f"histograma - ciudad {city}")
        axes[i, 0].set_xlabel('tiempo de parada en segundos')
        axes[i, 0].set_ylabel('frecuencia')
        axes[i, 0].axvline(threshold, color='red', linestyle='--', label=f'umbral: {threshold}s')
        axes[i, 0].legend()
        axes[i, 0].set_xlim(0, threshold + 30)

        sns.kdeplot(city_df[time_column], shade=True, ax=axes[i, 1])
        axes[i, 1].set_title(f'Densidad (KDE) - Ciudad: {city}')
        axes[i, 1].set_xlabel('Tiempo de parada (segundos)')
        axes[i, 1].set_ylabel('Densidad')
        axes[i, 1].axvline(threshold, color='red', linestyle='--', label=f'Umbral: {threshold}s')
        axes[i, 1].legend()
        axes[i, 1].set_xlim(0, threshold + 30)

    plt.tight_layout()
    plt.show()

def check_short_time_behavior(df: pd.DataFrame, time_column: str, threshold: int = 60):
    df = df.dropna(subset=[time_column])
    df_short = df[df[time_column] <= threshold]
    df_normal = df[df[time_column] > threshold]

    total_count = len(df)
    short_count = len(df_short)
    normal_count = len(df_normal)

    print(f"--- Análisis del Comportamiento de Tiempos de Parada ---")
    print(f"Umbral de 'tiempo corto' establecido en: {threshold} segundos.")
    print("-" * 50)
    print(f"Número total de paradas: {total_count}")
    print(f"Paradas cortas (<= {threshold}s): {short_count} ({short_count/total_count:.2%})")
    print(f"Paradas normales (> {threshold}s): {normal_count} ({normal_count/total_count:.2%})")
    print("-" * 50)
    
    print("\nEstadísticas Descriptivas:")
    print("--- Tiempos de Parada Cortos ---")
    print(df_short[time_column].describe())
    
    print("\n--- Tiempos de Parada Normales ---")
    print(df_normal[time_column].describe())

def compare_distributions_by_threshold(df:pd.DataFrame, time_column: str, threshold: int=60):
    df = df.dropna(subset=[time_column])
    df = df[df[time_column] > 0]

    df_short = df[df[time_column] <= threshold]
    df_normal = df[df[time_column] > threshold]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.histplot(df_short[time_column], bins=30, kde=False, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title(f"Histograma: Paradas Cortas (≤ {threshold}s)")
    axes[0, 0].set_xlabel('Tiempo de parada (segundos)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_xlim(0, threshold + 10)

        # Densidad - Paradas Cortas
    sns.kdeplot(df_short[time_column], shade=True, ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title(f"Densidad KDE: Paradas Cortas (≤ {threshold}s)")
    axes[0, 1].set_xlabel('Tiempo de parada (segundos)')
    axes[0, 1].set_ylabel('Densidad')
    axes[0, 1].set_xlim(0, threshold + 10)

    # Histograma - Paradas Normales
    sns.histplot(df_normal[time_column], bins=50, kde=False, ax=axes[1, 0], color='salmon')
    axes[1, 0].set_title(f"Histograma: Paradas Normales (> {threshold}s)")
    axes[1, 0].set_xlabel('Tiempo de parada (segundos)')
    axes[1, 0].set_ylabel('Frecuencia')

    # Densidad - Paradas Normales
    sns.kdeplot(df_normal[time_column], shade=True, ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title(f"Densidad KDE: Paradas Normales (> {threshold}s)")
    axes[1, 1].set_xlabel('Tiempo de parada (segundos)')
    axes[1, 1].set_ylabel('Densidad')

    plt.tight_layout()
    plt.show()

class GLMGamlogModel:
    def __init__(self, target_column='tiempo_parada', features=None, categorical_cols=None):
        self.target_column = target_column
        self.features = features if features is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.model = None
        self.res = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.final_X_columns = None

    def prepare_data(self, df_raw) -> Tuple[pd.DataFrame, pd.Series]:
        """prepara los datos para el modelado glm siguiendo """
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError('La entrada df raw debe ser un obj pandas')
        
        base_cols_set = set(self.features + self.categorical_cols + [self.target_column])
        existing_base_cols = [col for col in base_cols_set if col in df_raw.columns]

        if self.target_column not in existing_base_cols:
            # aqui es el tiempo de parada el que no esta saliendo
            raise ValueError(f"La columna objetivo {self.target_column} no se encuentra en el DataFrame inicial")
        
        df_base = df_raw[existing_base_cols].copy()
        df_base = df_base[df_base[self.target_column] > 0]

        # potencial experimento
        df_base[self.target_column] = np.log(df_base[self.target_column])


        cols_to_dummy = [col for col in self.categorical_cols if col in df_base.columns]
        df_dum = pd.get_dummies(df_base, columns = cols_to_dummy, drop_first=True)

        y = df_dum[self.target_column]
        X = df_dum.drop(columns = self.target_column)

        for col in ['profile_nid', 'patente']:
            if col in X.columns:
                X = X.drop(columns = col)


        X = X.apply(pd.to_numeric, errors = 'coerce')
        y = y.apply(pd.to_numeric, errors = 'coerce')

        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        # si hay columnas que no tienen ningun dato debiesemos dropearlas

        # por que esta esto asi
        # es la estrategia para rellenar datos faltantes
        # si nos quedamos con eliminar todos los nan nos quedamos sin DF
        X = X.dropna(axis=1, how='all')
        index_original = X.index
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=index_original)
        y = y.dropna()
        X = X.loc[y.index]

        # tratar sparsity : TEST
        # hay un error de memoria
        X = X.loc[:, X.nunique() > 1]
        threshold = 0.8  # al menos 20% de datos no nulos
        X = X.dropna(axis=1, thresh=int(threshold * len(X)))


        X = X.astype('float64')
        y = y.astype('float64')

        X = sm.add_constant(X, has_constant='add')
        self.final_X_columns = X.columns.tolist()

        return X, y
    
    def fit_data(self, X: pd.DataFrame, y: pd.Series):
        if 'const' not in X.columns:
            X = sm.add_constant(X, has_constant='add')
        
        self.X_train = X
        self.y_train = y

        print(f"Testeando tweenie")
        self.model = sm.GLM(self.y_train, self.X_train, family=sm.families.Tweedie(link=sm.families.links.log(), var_power=2.5))
        self.res = self.model.fit()
        self.final_X_columns = X.columns.tolist()
        print(f"Modelo entrenado con exito")

    def fit(self, df_raw):
        """
        prepara los datos
        """
        # creo que hay un error aca
        X, y = self.prepare_data(df_raw)
        print(X.head())
        # correccion aca
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=0.8, test_size=0.2, random_state=42
        )

        # Gamma o Inverse Gaussian?
        # probar valores en el power
        self.model = sm.GLM(self.y_train, self.X_train, family = sm.families.Tweedie(link=sm.families.links.log(), var_power=1.5))
        self.res = self.model.fit()
        print("Modelo GLM entrenado exitosamente")

    def summary(self):
        if self.res is None:
            raise ValueError('El modelo no ha sido entrenado')
        print(self.res.summary())
    
    def evaluate(self):
        if self.res is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # test de alineamiento de indices
        y_true = self.y_test 
        if isinstance(y_true, pd.Series):
            y_true = y_true.loc[self.X_test.index]
        
        y_pred = self.res.predict(self.X_test)

        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        mse = np.mean((self.y_test - y_pred)**2)
        
        try: 
            r2_mcfadden = 1 - (self.res.deviance / self.res.null_deviance)
        except Exception:
            r2_mcfadden = np.nan

        print(f"Error cuadratico {mse:.2f}")
        print(f"R cuadrado aproximado {r2_mcfadden:.4f}")
        # TODO: comentar el mcfadden
        return mse, r2_mcfadden
    
    def predict(self, new_data):
        if self.res is None:
            raise ValueError('El modelo no ha sido entrenado')
        
        cols_to_dummy = [col for col in self.categorical_cols if col in new_data.columns]
        required_cols = list(set(self.features + self.categorical_cols))
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas requeridas faltan {missing_cols}")
        
        temp_df_dum = pd.get_dummies(new_data[required_cols], columns = cols_to_dummy, drop_first=True)

        X_new_aligned = pd.DataFrame(columns = self.final_X_columns)
        X_new_aligned = pd.concat([X_new_aligned, temp_df_dum], ignore_index=True)

        if 'const' in X_new_aligned.columns:
            X_new_aligned = X_new_aligned.drop(columns = 'const')

        for col in self.final_X_columns:
            if col not in X_new_aligned.columns and col != 'const':
                X_new_aligned[col] = 0

        X_new_aligned = X_new_aligned[ [col for col in self.final_X_columns if col != 'const']]
        X_new_aligned = X_new_aligned.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        if self.X_train is not None:
            for col in X_new_aligned.columns:
                if col in self.X_train.columns:
                    X_new_aligned[col] = X_new_aligned[col].fillna(self.X_train[col].mean())
                else:
                    X_new_aligned[col] = X_new_aligned[col].fillna(0)
                
        #error corregido
        X_new_aligned = sm.add_constant(X_new_aligned, has_constant='add')
        X_new_aligned = X_new_aligned[self.final_X_columns]

        predictions_linear = self.res.predict(X_new_aligned, linear=True)
        # error potencialmente corregido
        predictions = np.exp(np.asarray(predictions_linear, dtype=float))
        return predictions
    
    def plot_diagnostics(self):
        if self.res is None:
            raise ValueError('El modelo no ha sido entrenado')
        
        fig, axes = plt.subplots(1, 3, figsize = (18, 5))
        deviance_residuals = self.res.resid_deviance
        fitted_values = self.res.fittedvalues

        axes[0].scatter(fitted_values, deviance_residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_title('residuos de varianza vs valores ajustados')
        axes[0].set_xlabel('valores ajustados (predicted)')
        axes[0].set_ylabel('residuos de varianza')

        import scipy.stats as stats
        stats.probplot(deviance_residuals, dist='norm', plot=axes[1])
        axes[1].set_title('Q-Q plot de residuos de varianza')
        axes[2].hist(deviance_residuals, bins=50, edgecolor='k')
        axes[2].set_xlabel('residuos de varianza')
        axes[2].set_ylabel('frecuencia')

        plt.tight_layout()
        plt.show()

    def plot_residuals_vs_features(self):
        if self.res is None:
            raise ValueError('el modelo no ha sido entrenado')
        deviance_residuals = self.res.resid_deviance

        # como sacar las top features
        # podria dumpearlas de la signifcancia del GLM
        top_features = ['dist_prev', 'load', 'hora_llegada', 'n_paradas_en_ruta']
        fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 5))

        for i, feature in enumerate(top_features):
            if feature in self.X_train.columns:
                sns.scatterplot(x=self.X_train[feature], y=deviance_residuals, ax= axes[i], alpha=0.5)
                axes[i].axhline(y=0, color='r', linestyle='--')
                axes[i].set_title(f'Residuos vs. {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Residuos de Devianza')
                
        plt.tight_layout()
        plt.show()

class TwoPartModel:
    # notar que el threshold short time es un parametro
    # TODO: instanciar eso apropiadamente (?)
    # TODO: 
    def __init__(self, features, categorical_cols, target_column, threshold_short_time=13):
        self.features = features
        self.categorical_cols = categorical_cols
        self.target_column = target_column
        self.threshold_short_time = threshold_short_time

        self.logistic_model = None
        self.glmgamlog_model = GLMGamlogModel(target_column, features, categorical_cols)

        self.X_all = None
        self.y_binary = None
        self.X_normal = None
        self.y_normal = None

        self.imputer = None
        self.preprocessor_pipeline = None

    
    def prepare_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        print(f"\n prepare data for two part model")
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError('must be a pandas object - cris')
        
        # there is an error in prepare data method
        df_processed = self.glmgamlog_model.prepare_data(df_raw.copy())
        # I should probably take a look at this returns 
        X_glm, y_glm = self.glmgamlog_model.prepare_data(df_raw.copy())

        # correction in here
        is_normal = (y_glm > np.log(self.threshold_short_time)).astype(int)
        X_all = X_glm.loc[is_normal.index]
        y_binary = is_normal

        df_normal = X_glm.loc[is_normal==1]
        y_normal = y_glm.loc[is_normal==1]

        self.X_all = X_all
        self.y_binary = y_binary
        self.X_normal = df_normal
        self.y_normal = y_normal

        print(f"total de datos {len(X_all)}")
        print(f"paradas normales (> {self.threshold_short_time}s) {len(self.X_normal)} ({len(self.X_normal)/len(X_all):.2%})")
        print(f"paradas cortas (<= {self.threshold_short_time}s) {len(X_all) - len(self.X_normal)} ({(len(X_all) - len(self.X_normal)) / len(X_all):.2%})")

        return X_all, y_binary, df_normal, y_normal
    
    def fit(self, df_raw: pd.DataFrame):
        self.prepare_data(df_raw)
        print(f"\n entrenando clasificacion")
        self.logistic_model = LogisticRegression(solver='liblinear', random_state=42)
        self.logistic_model.fit(self.X_all, self.y_binary)
        
        print("Métricas de la parte de clasificación:")
        y_pred_binary = self.logistic_model.predict(self.X_all)
        print(f"  - Accuracy: {accuracy_score(self.y_binary, y_pred_binary):.4f}")
        print(f"  - AUC-ROC: {roc_auc_score(self.y_binary, self.logistic_model.predict_proba(self.X_all)[:, 1]):.4f}")

        print("\n -- entrenando regresion glm gamma")
        self.glmgamlog_model.fit_data(self.X_normal, self.y_normal)
        print("\nEntrenamiento del modelo de dos partes completado.")


    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if self.logistic_model is None or self.glmgamlog_model.res is None:
            raise ValueError('Los modelos no han sido entrenados')
        
        X_test_processed, _ = self.glmgamlog_model.prepare_data(new_data)
        X_test_aligned_logreg = X_test_processed.reindex(columns=self.X_all.columns, fill_value=0)
        X_test_aligned_glm = X_test_processed.reindex(columns = self.glmgamlog_model.final_X_columns, fill_value=0)

        prob_normal = self.logistic_model.predict_proba(X_test_aligned_logreg)[:, 1]
        pred_duration_normal = self.glmgamlog_model.res.predict(X_test_aligned_glm)

        return pd.Series(prob_normal.values * pred_duration_normal.values, index=new_data.index)
    
    def plot_diagnostics(self):
        if self.logistic_model is None or self.glmgamlog_model.res is None:
            raise ValueError('ambos modelos deben estar entrenados')
        
        print('\n diagnosticos del modelo de clasificacion')
        y_scores = self.logistic_model.predict_proba(self.X_all)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color = 'navy', lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos positivos')
        ax.set_ylabel('tasa de verdaderos positivos')
        ax.set_title('Curva roc para la clasificacion')
        ax.legend(loc='lower right')
        plt.show()

        y_pred_binary = self.logistic_model.predict(self.X_all)
        cm = confusion_matrix(self.y_binary, y_pred_binary)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Prediccion')
        ax.set_ylabel('Valor real')
        ax.set_title('matriz de confusion')
        ax.xaxis.set_ticklabels(['Corta', 'Normal'])
        ax.yaxis.set_ticklabels(['Corta', 'Normal'])
        plt.show()
        
        # --- Parte 2: Diagnósticos del Modelo de Regresión (GLM) ---
        print("\n--- Diagnósticos del Modelo de Regresión (Parte 2) ---")
        self.glmgamlog_model.plot_diagnostics()
        print("\n doing summary")
        self.glmgamlog_model.summary()
        self.glmgamlog_model.evaluate()
        self.glmgamlog_model.plot_residuals_vs_features()

    def plot_feature_importance_logistic_regression(self):
        if self.logistic_model is None:
            raise ValueError('no trained model')
        
        feature_names = self.X_all.columns
        coefficients = self.logistic_model.coef_[0]

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })

        feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)
        feature_importance_df = feature_importance_df.head(8)
        # recortar y solo poner las 8 mas relevantes

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Importancia de Features en el modelo de clasificacion (regresion logistica)')
        plt.xlabel('Valor del coeficiente')
        plt.ylabel('Feature')
        plt.show()


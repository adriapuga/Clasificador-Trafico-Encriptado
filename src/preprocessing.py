import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_dataset(path: str) -> pd.DataFrame:
    """Carga el dataset ARFF y decodifica las etiquetas."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df['class1'] = df['class1'].str.decode('utf-8')
    return df

def remove_outliers(df: pd.DataFrame, threshold: float = 99.9) -> pd.DataFrame:
    """Elimina outliers extremos por percentil."""
    features = df.drop(columns='class1')
    mask = pd.Series([True] * len(df))
    for col in features.columns:
        upper = np.percentile(df[col], threshold)
        mask = mask & (df[col] <= upper)
    df_clean = df[mask].reset_index(drop=True)
    print(f"Filas eliminadas por outliers: {len(df) - len(df_clean)}")
    return df_clean

def preprocess(path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline completo de preprocessing.
    Devuelve X_train, X_test, y_train, y_test y guarda el scaler.
    """
    # Cargar
    df = load_dataset(path)
    print(f"Dataset cargado: {df.shape}")

    # Eliminar outliers
    df = remove_outliers(df)
    print(f"Dataset tras outliers: {df.shape}")

    # Separar features y etiqueta
    X = df.drop(columns='class1').values
    y = (df['class1'] == 'VPN').astype(int).values  # VPN=1, Non-VPN=0

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Guardar scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler guardado en models/scaler.pkl")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
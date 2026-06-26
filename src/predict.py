import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FEATURE_NAMES = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

def load_model(model_name: str = 'random_forest'):
    """Carga el modelo y el scaler guardados."""
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_flow(features: dict, model_name: str = 'random_forest') -> dict:
    """
    Clasifica un flujo de red como VPN o Non-VPN.

    Args:
        features: dict con las 23 features del flujo
        model_name: 'random_forest' o 'svm'

    Returns:
        dict con la predicción y la probabilidad
    """
    model, scaler = load_model(model_name)

    # Construir vector de features en el orden correcto
    x = np.array([[features[f] for f in FEATURE_NAMES]])

    # Normalizar
    x_scaled = scaler.transform(x)

    # Predecir
    prediction = model.predict(x_scaled)[0]
    probability = model.predict_proba(x_scaled)[0]

    label = 'VPN' if prediction == 1 else 'Non-VPN'
    confidence = probability[prediction]

    return {
        'label': label,
        'confidence': round(float(confidence), 4),
        'prob_nonvpn': round(float(probability[0]), 4),
        'prob_vpn': round(float(probability[1]), 4)
    }

def predict_batch(X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
    """
    Clasifica un batch de flujos de red.

    Args:
        X: array de shape (n_samples, 23)
        model_name: 'random_forest' o 'svm'

    Returns:
        array de etiquetas ('VPN' o 'Non-VPN')
    """
    model, scaler = load_model(model_name)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return np.where(predictions == 1, 'VPN', 'Non-VPN')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_NAMES = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

def get_feature_importance(model, feature_names: list = FEATURE_NAMES) -> pd.DataFrame:
    """Extrae y ordena la importancia de features del modelo."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return df

def plot_feature_importance(model, feature_names: list = FEATURE_NAMES, top_n: int = 15):
    """Genera gráfica de importancia de features y la guarda."""
    df = get_feature_importance(model, feature_names)
    df_top = df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df_top['feature'][::-1], df_top['importance'][::-1], color='#e74c3c')
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} features más importantes - Random Forest')
    plt.tight_layout()

    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Guardado en results/feature_importance.png")
    return df

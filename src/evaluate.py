import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score, f1_score)
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """Evalúa un modelo y devuelve un dict con las métricas principales."""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_vpn': f1_score(y_test, y_pred, pos_label=1),
        'f1_nonvpn': f1_score(y_test, y_pred, pos_label=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
    }

    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=['Non-VPN', 'VPN']))
    return metrics

def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model", color: str = "Blues"):
    """Genera y guarda la confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color,
                xticklabels=['Non-VPN', 'VPN'],
                yticklabels=['Non-VPN', 'VPN'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()

    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    filename = f"cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    plt.show()

def plot_roc_curve(model, X_test, y_test, model_name: str = "Model"):
    """Genera y guarda la curva ROC. Solo para modelos con predict_proba."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.tight_layout()

    results_dir = os.path.join(BASE_DIR, 'results')
    filename = f"roc_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"AUC-ROC: {roc_auc:.4f}")
    return roc_auc

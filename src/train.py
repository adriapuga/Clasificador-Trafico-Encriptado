import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train_random_forest(X_train, y_train, random_state: int = 42):
    """Entrena un Random Forest y lo guarda."""
    print("Entrenando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Random Forest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    print(f"Modelo guardado en models/random_forest.pkl")
    return rf

def train_svm(X_train, y_train, random_state: int = 42):
    """Entrena un SVM y lo guarda."""
    print("Entrenando SVM...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=random_state
    )
    svm.fit(X_train, y_train)

    cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"SVM CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(svm, os.path.join(MODELS_DIR, 'svm.pkl'))
    print(f"Modelo guardado en models/svm.pkl")
    return svm

# Clasificador de Tráfico Encriptado

> Demostración práctica de Machine Learning aplicado a seguridad de redes:
> clasificación de tráfico VPN vs Non-VPN a partir de features estadísticas
> de flujo, sin descifrar el contenido de los paquetes.

![Python](https://img.shields.io/badge/python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-ISCX_VPN_2016-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/status-Completado-green?style=for-the-badge)

---

## Objetivo

Este proyecto demuestra cómo aplicar un pipeline completo de Machine Learning a un problema real de seguridad de redes: **detectar si un flujo de tráfico usa VPN o no, sin inspeccionar el contenido de los paquetes**.

El objetivo no es construir una herramienta lista para producción, sino demostrar que las técnicas de ML son aplicables a problemas reales de redes y seguridad, usando únicamente estadísticas a nivel de flujo (tiempos entre paquetes, bytes por segundo, duración...) que no requieren romper el cifrado TLS/HTTPS.

---

## Aclaraciones importantes

### Enfoque académico y demostrativo

Este proyecto es un ejercicio práctico con fines educativos. Hay aspectos que en un entorno de producción real serían más complejos:

- El dataset es de **2016**: en una red real actual habría que capturar tráfico propio y actualizar las features.
- El modelo está entrenado y evaluado sobre el **mismo dominio de datos**: en producción habría que validar contra tráfico de otras redes.
- No se ha optimizado el tiempo de inferencia para uso en tiempo real.

Lo hacemos así por dos razones:

- Es un proyecto **académico**: el objetivo es demostrar que entendemos el pipeline completo de ML aplicado a redes, no desplegar un producto.
- Tiene que ser **reproducible**: cualquier persona con el dataset puede ejecutar exactamente los mismos pasos y obtener los mismos resultados.

### ¿Por qué este problema es interesante?

La detección de tráfico cifrado sin inspeccionarlo es un problema activo en la industria. Firewalls de nueva generación, sistemas SIEM y herramientas de monitorización de red lo hacen exactamente así: analizan el comportamiento del flujo, no su contenido. La VPN añade overhead de encapsulado y latencia que se refleja en las estadísticas del flujo, y eso es lo que el modelo aprende.

---

## Resultados

| Modelo | Accuracy | F1 VPN | F1 Non-VPN | AUC-ROC |
|---|---|---|---|---|
| Random Forest | **91%** | **0.92** | **0.91** | **0.97** |
| SVM (RBF) | 64% | 0.70 | 0.55 | — |

El Random Forest supera claramente al SVM en este dataset. Las features de flujo de red tienen escalas muy dispares y distribuciones con colas largas, lo que favorece a los modelos basados en árboles frente a los de margen.

### Features más relevantes

| # | Feature | Importancia | Interpretación |
|---|---|---|---|
| 1 | `total_biat` | 0.107 | Bytes en dirección backward — el overhead del encapsulado VPN altera el volumen de respuesta |
| 2 | `flowBytesPerSecond` | 0.086 | Tasa de bytes del flujo — el cifrado añadido cambia la densidad del tráfico |
| 3 | `min_flowiat` | 0.075 | Tiempo mínimo entre paquetes — la latencia extra del túnel VPN es detectable |
| 4 | `max_flowiat` | 0.070 | Tiempo máximo entre paquetes — los picos de latencia delatan el encapsulado |
| 5 | `flowPktsPerSecond` | 0.068 | Paquetes por segundo — el overhead de headers VPN reduce la tasa efectiva |

---

## Stack tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.13 |
| ML | scikit-learn | 1.x |
| Datos | pandas, numpy | — |
| Visualización | matplotlib, seaborn | — |
| Serialización | joblib | — |
| Dataset | ISCX VPN-NonVPN 2016 (UNB) | — |
| Entorno | Parrot OS | — |

---

## Dataset

**ISCX VPN-NonVPN Traffic Dataset 2016**
Universidad de New Brunswick (UNB) — Canadian Institute for Cybersecurity

| Característica | Valor |
|---|---|
| Total de muestras | 18.758 flujos |
| Features | 23 (estadísticas de flujo) |
| Clases | VPN (9.793) / Non-VPN (8.965) |
| Balance | ~52% / ~48% |
| Formato | ARFF |

El dataset contiene tráfico de red real capturado y etiquetado. Las features son estadísticas temporales y de volumen a nivel de flujo: no se incluye ningún byte del payload. Descarga disponible en: https://www.unb.ca/cic/datasets/vpn.html

---

## Arquitectura del pipeline

| Paso | Módulo | Descripción |
|---|---|---|
| 1 | `preprocessing.py` | Carga del ARFF, eliminación de outliers (percentil 99.9), train/test split estratificado (80/20), normalización StandardScaler |
| 2 | `train.py` | Entrenamiento de Random Forest (100 árboles) y SVM (kernel RBF), cross-validation 5-fold |
| 3 | `evaluate.py` | Classification report, confusion matrix, curva ROC y AUC-ROC |
| 4 | `features.py` | Extracción y visualización de importancia de features |
| 5 | `predict.py` | Inferencia individual (`predict_flow`) y en batch (`predict_batch`) |

---

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `src/preprocessing.py` | Carga, limpieza, normalización y split del dataset |
| `src/train.py` | Entrenamiento de Random Forest y SVM |
| `src/evaluate.py` | Métricas, confusion matrix y curva ROC |
| `src/features.py` | Análisis e importancia de features |
| `src/predict.py` | Inferencia sobre flujos individuales y en batch |
| `notebooks/01_eda.ipynb` | Análisis exploratorio del dataset |
| `notebooks/02_preprocessing.ipynb` | Validación del pipeline de preprocessing |
| `notebooks/03_training.ipynb` | Entrenamiento y evaluación de modelos |
| `notebooks/04_features.ipynb` | Visualización de importancia de features |
| `models/random_forest.pkl` | Modelo entrenado (Random Forest) |
| `models/scaler.pkl` | Scaler ajustado sobre train |
| `results/` | Gráficas generadas (confusion matrix, ROC, features) |
| `data/download_info.md` | Instrucciones para descargar el dataset |

---

## Reproducir el proyecto

### Requisitos previos

- Python 3.10+
- Dataset ISCX VPN-NonVPN 2016 descargado en `data/Scenario A1-ARFF/`

### Instalación

```bash
git clone https://github.com/adriapuga/Clasificador-Trafico-Encriptado
cd Clasificador-Trafico-Encriptado
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Inferencia sobre un flujo nuevo

```python
from src.predict import predict_flow

flujo = {
    'duration': 9368711.0,
    'total_fiat': 16.0,
    'total_biat': 4.0,
    'min_fiat': 1564818.0,
    'min_biat': 1549373.0,
    'max_fiat': 190205.28,
    'max_biat': 203290.45,
    'mean_fiat': 389822.39,
    'mean_biat': 370323.71,
    'flowPktsPerSecond': 10.35,
    'flowBytesPerSecond': 1500.0,
    'min_flowiat': 100000.0,
    'max_flowiat': 500000.0,
    'mean_flowiat': 300000.0,
    'std_flowiat': 267600.0,
    'min_active': 1871488.0,
    'mean_active': 1983656.0,
    'max_active': 2195089.0,
    'std_active': 183219.0,
    'min_idle': 1234883.0,
    'mean_idle': 1420565.0,
    'max_idle': 1523088.0,
    'std_idle': 161096.0
}

resultado = predict_flow(flujo)
print(resultado)
# {'label': 'Non-VPN', 'confidence': 0.94, 'prob_nonvpn': 0.94, 'prob_vpn': 0.06}
```

---

## Aviso legal

> Este proyecto está diseñado **exclusivamente con fines educativos y de portfolio**. El dataset utilizado es público y fue capturado en entornos controlados por la Universidad de New Brunswick. Las técnicas descritas son de análisis pasivo de tráfico de red y no implican interceptación, modificación ni acceso no autorizado a sistemas de terceros.

---

## Autor

Proyecto desarrollado como práctica de Machine Learning aplicado a redes por Adrià Puga.

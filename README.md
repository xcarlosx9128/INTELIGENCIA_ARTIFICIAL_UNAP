# Sistema Inteligente de Detección y Clasificación de Amenazas Cibernéticas

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> Proyecto de Investigación Formativa en Inteligencia Artificial  
> Universidad Nacional de la Amazonía Peruana - UNAP  
> Facultad de Ingeniería de Sistemas e Informática

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Resultados](#-resultados)
- [Tecnologías](#-tecnologías)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Dataset](#-dataset)
- [Metodología](#-metodología)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)
- [Contacto](#-contacto)

## 🎯 Descripción

Este proyecto desarrolla un sistema inteligente basado en **Machine Learning** para la detección y clasificación automática de amenazas cibernéticas utilizando registros de seguridad de **FortiEDR**. El sistema implementa algoritmos de **ensemble learning** para identificar comportamientos maliciosos con alta precisión y sensibilidad.

### Problema Abordado

Los sistemas de detección tradicionales basados en firmas y reglas predefinidas presentan limitaciones significativas:
- Incapacidad para detectar amenazas de día cero
- Altas tasas de falsos positivos
- Falta de adaptación a nuevos patrones de ataque
- Sobrecarga de analistas de seguridad

### Solución Propuesta

Sistema de clasificación binaria que:
- Utiliza **Gradient Boosting** para clasificación de eventos
- Alcanza **94.31% de accuracy** y **99.47% de recall**
- Procesa eventos en tiempo real (<5ms por clasificación)
- Se adapta mediante reentrenamiento periódico

## ✨ Características

- **🎯 Alta Precisión**: 94.31% de accuracy en detección de amenazas
- **🚀 Alto Recall**: 99.47% - solo 3 amenazas no detectadas de 571
- **⚡ Eficiente**: Tiempo de entrenamiento <3 segundos, inferencia <5ms
- **📊 Interpretable**: Análisis de importancia de características
- **🔄 Adaptable**: Pipeline de reentrenamiento automatizado
- **📈 Validado**: Validación cruzada 5-fold con ±0.51% desviación

## 📊 Resultados

### Métricas del Modelo Gradient Boosting

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 94.31% | Exactitud global del modelo |
| **Precision** | 94.67% | Confiabilidad de alertas positivas |
| **Recall** | 99.47% | Detección de amenazas reales |
| **F1-Score** | 97.01% | Balance precision/recall |

### Validación Cruzada (5-fold)

| Métrica | Promedio | Desviación |
|---------|----------|------------|
| Accuracy | 91.95% | ±0.51% |
| Precision | 93.38% | ±0.31% |
| Recall | 98.30% | ±0.87% |
| F1-Score | 95.77% | ±0.29% |

### Comparación de Modelos

```
Gradient Boosting:  ████████████████████ 94.31% (Seleccionado)
XGBoost:           ███████████████████▌ 93.82%
Random Forest:     ███████████████      78.21%
```

### Matriz de Confusión

```
                 Predicción
              NORMAL  PELIGROSO
Real  NORMAL     12       32
      PELIGROSO   3      568

✓ Verdaderos Positivos: 568 (99.47% de amenazas detectadas)
✓ Falsos Negativos: 3 (0.53% de amenazas no detectadas)
```

## 🛠️ Tecnologías

### Lenguajes y Frameworks
- **Python 3.9+**
- **Scikit-learn 1.2.2** - Algoritmos de ML
- **XGBoost 1.7.5** - Gradient boosting optimizado
- **Pandas 1.5.3** - Manipulación de datos
- **NumPy 1.23.5** - Operaciones numéricas

### Herramientas de Análisis
- **Matplotlib 3.7.1** - Visualización
- **Seaborn 0.12.2** - Gráficos estadísticos
- **Joblib 1.2.0** - Serialización de modelos

### Entorno de Desarrollo
- **Jupyter Notebook** - Exploración interactiva
- **Git/GitHub** - Control de versiones
- **VS Code** - Editor de código

## 📥 Instalación

### Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- 4GB RAM mínimo (recomendado 8GB)
- 500MB espacio en disco

### Pasos de Instalación

1. **Clonar el repositorio**

```bash
git clone https://github.com/xcarlosx9128/INTELIGENCIA_ARTIFICIAL_UNAP.git
cd INTELIGENCIA_ARTIFICIAL_UNAP
```

2. **Crear entorno virtual**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

4. **Verificar instalación**

```bash
python -c "import sklearn, xgboost, pandas; print('✓ Instalación exitosa')"
```

## 🚀 Uso

### 1. Procesamiento de Datos

Procesa datos crudos de FortiEDR y genera dataset preparado:

```bash
python src/1_procesar_datos_crudos.py --input datos_crudos.csv --output dataset_procesado.xlsx
```

**Salida:**
- Dataset filtrado (eventos con Activity Name '*Block*')
- 19 características derivadas
- Etiquetas binarias (PELIGROSO/NORMAL)

### 2. Entrenamiento del Modelo

Entrena y compara múltiples algoritmos:

```bash
python src/2_entrenar_modelo.py --dataset dataset_procesado.xlsx --output modelo_entrenado.pkl
```

**Salida:**
- Modelo entrenado serializado (.pkl)
- Métricas de rendimiento
- Matriz de confusión
- Importancia de características

### 3. Clasificación de Eventos

Clasifica nuevos eventos con el modelo entrenado:

```bash
python src/3_predecir_nuevos_datos.py --input eventos_nuevos.xlsx --output predicciones.xlsx
```

**Salida:**
- Clasificación binaria (0=NORMAL, 1=PELIGROSO)
- Probabilidad de amenaza (0-1)
- Recomendaciones de acción

### Ejemplo de Uso Interactivo

```python
import joblib
import pandas as pd

# Cargar modelo entrenado
modelo = joblib.load('models/modelo_deteccion_amenazas.pkl')

# Preparar evento de ejemplo
evento = {
    'longitud_nombre': 35,
    'contiene_numeros': 1,
    'tiene_caracteres_repetidos': 1,
    'hora_dia': 23,
    'es_horario_laboral': 0,
    # ... resto de características
}

# Predecir
probabilidad = modelo.predict_proba([evento])[0][1]
clasificacion = 'PELIGROSO' if probabilidad > 0.5 else 'NORMAL'

print(f"Clasificación: {clasificacion} (Probabilidad: {probabilidad:.2%})")
```

## 📁 Estructura del Proyecto

```
INTELIGENCIA_ARTIFICIAL_UNAP/
│
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos crudos originales
│   │   └── fortiedr_events.csv
│   ├── processed/                 # Datos procesados
│   │   └── dataset_ml_final.xlsx
│   └── README.md                  # Descripción de datos
│
├── models/                        # Modelos entrenados
│   ├── modelo_deteccion_amenazas.pkl  (399 KB)
│   ├── info_modelo.pkl           # Metadata del modelo
│   └── README.md                  # Descripción de modelos
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_ingenieria_caracteristicas.ipynb
│   └── 03_evaluacion_modelos.ipynb
│
├── src/                          # Código fuente
│   ├── 1_procesar_datos_crudos.py
│   ├── 2_entrenar_modelo.py
│   ├── 3_predecir_nuevos_datos.py
│   └── utils/                    # Utilidades
│       ├── feature_engineering.py
│       └── evaluation_metrics.py
│
├── docs/                         # Documentación
│   ├── informe_final.pdf
│   ├── manual_usuario.pdf
│   └── presentacion.pptx
│
├── tests/                        # Tests unitarios
│   ├── test_processing.py
│   └── test_model.py
│
├── requirements.txt              # Dependencias Python
├── .gitignore                   # Archivos ignorados por Git
├── LICENSE                      # Licencia del proyecto
└── README.md                    # Este archivo
```

## 📊 Dataset

### Características del Dataset

| Atributo | Valor |
|----------|-------|
| **Fuente** | FortiEDR (Endpoint Detection and Response) |
| **Período** | Febrero - Septiembre 2025 (7 meses) |
| **Registros totales** | 100,044 eventos |
| **Registros procesados** | 6,149 eventos relevantes |
| **Clases** | PELIGROSO (92.8%), NORMAL (7.2%) |
| **Ratio de desbalance** | 12.9:1 |

### Distribución por Activity Name

```
Malicious-Block:              4,601 eventos → PELIGROSO
Suspicious-Block:               832 eventos → PELIGROSO
PUP-Block:                      665 eventos → NORMAL
Malicious-SimulationBlock:       24 eventos → PELIGROSO
Suspicious-SimulationBlock:      22 eventos → PELIGROSO
PUP-SimulationBlock:              5 eventos → NORMAL
```

### Características Derivadas (19 en total)

**Características basadas en nombre del proceso:**
- `longitud_nombre`: Longitud del nombre
- `contiene_numeros`: Presencia de dígitos
- `tiene_extension_sospechosa`: Extensiones .exe, .dll, .bat, etc.
- `tiene_caracteres_repetidos`: Repetición >3 caracteres
- `es_comando`: Comandos de sistema

**Análisis léxico:**
- `contiene_descargar`: Palabras 'download', 'wget', 'curl'
- `contiene_malware`: Términos 'trojan', 'virus', 'ransomware'
- `contiene_script`: Indicadores de scripting
- `es_cracker`: Términos 'crack', 'keygen', 'patch'

**Características temporales:**
- `hora_dia`: Hora de ejecución (0-23)
- `dia_semana`: Día de la semana (0-6)
- `es_horario_laboral`: Boolean horario 8am-6pm
- `es_fin_semana`: Boolean sábado/domingo

### Importancia de Características

```
longitud_nombre:                29.4% ███████████████████████████████
tiene_caracteres_repetidos:     18.2% ██████████████████████
contiene_numeros:               12.8% ███████████████
hora_dia:                        9.7% ███████████
es_horario_laboral:              7.5% █████████
Otras características:          22.4% ████████████████████████
```

## 🔬 Metodología

### 1. Procesamiento de Datos (CRISP-DM)

```
Datos Crudos (100,044 registros)
    ↓
Filtrado (Activity Name '*Block*')
    ↓
Dataset Relevante (6,149 eventos)
    ↓
Ingeniería de Características (19 features)
    ↓
División Estratificada (90% train / 10% test)
    ↓
Dataset ML Listo
```

### 2. Entrenamiento y Validación

```python
# Pseudocódigo del proceso
for modelo in [RandomForest, GradientBoosting, XGBoost]:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    evaluar_metricas(y_test, y_pred)
    
mejor_modelo = seleccionar_por_f1_score()

# Validación cruzada
cv_scores = cross_validate(mejor_modelo, X_train, y_train, cv=5)
```

### 3. Hiperparámetros del Modelo Seleccionado

**Gradient Boosting:**
```python
GradientBoostingClassifier(
    n_estimators=100,      # Número de árboles
    max_depth=5,           # Profundidad máxima
    learning_rate=0.1,     # Tasa de aprendizaje
    min_samples_split=20,  # Mínimo para dividir nodo
    random_state=42        # Reproducibilidad
)
```

### 4. Evaluación

- **División holdout**: 90% entrenamiento, 10% prueba
- **Estratificación**: Mantiene proporción de clases
- **Validación cruzada**: 5-fold estratificada
- **Métricas**: Accuracy, Precision, Recall, F1-Score
- **Análisis de errores**: Estudio de FP y FN

## 📚 Documentación Adicional

- **[Informe Técnico Completo](docs/informe_final.pdf)** - Documento académico con metodología y resultados detallados
- **[Manual de Usuario](docs/manual_usuario.pdf)** - Guía de instalación y uso
- **[Presentación del Proyecto](docs/presentacion.pptx)** - Slides para defensa
- **[Notebooks de Análisis](notebooks/)** - Exploración interactiva de datos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas de Mejora Potencial

- [ ] Implementar arquitecturas de deep learning (LSTM, Transformers)
- [ ] Expandir dataset con más eventos de clase NORMAL
- [ ] Agregar explicabilidad con SHAP values
- [ ] Desarrollar API REST para integración
- [ ] Crear dashboard de monitoreo en tiempo real
- [ ] Implementar aprendizaje continuo (online learning)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Contacto

**Autor:** Carlos [Apellido]  
**Universidad:** Universidad Nacional de la Amazonía Peruana (UNAP)  
**Facultad:** Ingeniería de Sistemas e Informática  
**Curso:** Inteligencia Artificial  
**Docente:** Dr. Ing. Carlos Alberto García Cortegano

**Repositorio:** [https://github.com/xcarlosx9128/INTELIGENCIA_ARTIFICIAL_UNAP](https://github.com/xcarlosx9128/INTELIGENCIA_ARTIFICIAL_UNAP)

---

## 🎓 Agradecimientos

- **Dr. Ing. Carlos Alberto García Cortegano** - Docente del curso de Inteligencia Artificial
- **Facultad de Ingeniería de Sistemas UNAP** - Recursos y apoyo académico
- **FortiEDR** - Plataforma de datos de seguridad
- **Comunidad de Scikit-learn** - Implementaciones robustas de ML

---

## 📈 Estadísticas del Proyecto

![Lenguajes](https://img.shields.io/github/languages/top/xcarlosx9128/INTELIGENCIA_ARTIFICIAL_UNAP)
![Tamaño del código](https://img.shields.io/github/languages/code-size/xcarlosx9128/INTELIGENCIA_ARTIFICIAL_UNAP)
![Última actualización](https://img.shields.io/github/last-commit/xcarlosx9128/INTELIGENCIA_ARTIFICIAL_UNAP)

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella ⭐**

**[⬆ Volver arriba](#sistema-inteligente-de-detección-y-clasificación-de-amenazas-cibernéticas)**

</div>

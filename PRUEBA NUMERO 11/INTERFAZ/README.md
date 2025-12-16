# 🛡️ Detector de Amenazas - UNAP

Sistema de detección de amenazas en logs de antivirus usando Machine Learning.

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

O manualmente:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly openpyxl
```

### 2. Ejecutar la aplicación

**En Windows:**
- Haz doble clic en `EJECUTAR.bat`

**Desde CMD/Terminal:**
```bash
streamlit run app_prediccion_mejorada.py
```

### 3. Usar la aplicación

1. Se abrirá tu navegador en `http://localhost:8501`
2. Sube tu archivo CSV/Excel con logs
3. Haz clic en "ANALIZAR AMENAZAS"
4. Revisa los resultados y descarga el análisis

## 📋 Formato del archivo

Tu CSV debe tener estas columnas con separador **punto y coma (;)**:

```csv
Activity Name;Process Name;Process Path;Count
Communication Blocked;powershell.exe;C:\Windows\System32\powershell.exe;5
File Quarantined;suspicious.exe;C:\Users\Admin\AppData\Temp\suspicious.exe;1
```

## 📁 Archivos necesarios

```
DetectorAmenazas/
├── app_prediccion_mejorada.py    ← Aplicación
├── mejor_modelo.pkl               ← Modelo (IMPORTANTE)
├── EJECUTAR.bat                   ← Ejecutar en Windows
├── requirements.txt               ← Dependencias
└── GUIA_COMPLETA.md              ← Documentación detallada
```

## 📊 Características del Modelo

- **Algoritmo:** XGBoost
- **Accuracy:** 82.5%
- **Recall:** 89.9%
- **Features:** 20 características

## ❓ Problemas comunes

### "No se encontró el archivo mejor_modelo.pkl"
→ Asegúrate de que `mejor_modelo.pkl` esté en la misma carpeta

### "No module named streamlit"
→ Ejecuta: `pip install -r requirements.txt`

### "Faltan columnas"
→ Tu CSV debe tener: Activity Name, Process Name, Process Path, Count

## 📖 Documentación completa

Ver archivo: **GUIA_COMPLETA.md**

## 🎓 Desarrollado por

Universidad Nacional del Altiplano - UNAP  
Noviembre 2025

---

**¿Primera vez usando la aplicación?** Lee la **GUIA_COMPLETA.md** para instrucciones detalladas.

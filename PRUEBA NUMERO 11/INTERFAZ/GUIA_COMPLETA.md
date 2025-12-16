# 🛡️ DETECTOR DE AMENAZAS - GUÍA COMPLETA

## 📋 DESCRIPCIÓN
Sistema de detección de amenazas en logs de antivirus usando Machine Learning (XGBoost).
Desarrollado para la Universidad Nacional del Altiplano - UNAP.

---

## 📦 REQUISITOS PREVIOS

### 1. Python 3.8 o superior
Verifica tu versión:
```bash
python --version
```

Si no tienes Python instalado, descárgalo desde: https://www.python.org/downloads/

### 2. Librerías necesarias
Instala todas las dependencias:

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly openpyxl
```

O usa este comando todo en uno:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly openpyxl --upgrade
```

---

## 📁 ESTRUCTURA DE ARCHIVOS

Organiza tus archivos así:

```
DetectorAmenazas/
│
├── app_prediccion_mejorada.py    ← Aplicación principal (archivo que acabas de descargar)
├── mejor_modelo.pkl               ← Modelo entrenado (debe estar en la misma carpeta)
│
├── datos/                         ← (Opcional) Carpeta para tus CSV
│   ├── datos_prueba.csv
│   └── otros_logs.csv
│
└── resultados/                    ← (Opcional) Para guardar análisis
    └── predicciones_20250104.csv
```

**IMPORTANTE:** Los archivos `app_prediccion_mejorada.py` y `mejor_modelo.pkl` DEBEN estar en la misma carpeta.

---

## 🚀 CÓMO EJECUTAR LA APLICACIÓN

### Método 1: Desde CMD/Terminal (Recomendado)

1. Abre CMD (Windows) o Terminal (Mac/Linux)
2. Navega a la carpeta donde están los archivos:
   ```bash
   cd C:\DetectorAmenazas
   ```
   o en Mac/Linux:
   ```bash
   cd /ruta/a/tu/carpeta/DetectorAmenazas
   ```

3. Ejecuta la aplicación:
   ```bash
   streamlit run app_prediccion_mejorada.py
   ```

4. Se abrirá automáticamente tu navegador en: `http://localhost:8501`

### Método 2: Doble clic (Windows)

1. Crea un archivo `ejecutar.bat` con este contenido:
   ```batch
   @echo off
   cd /d "%~dp0"
   streamlit run app_prediccion_mejorada.py
   pause
   ```

2. Guarda el archivo en la misma carpeta que `app_prediccion_mejorada.py`

3. Haz doble clic en `ejecutar.bat`

### ⚠️ Si el puerto 8501 está ocupado:

Usa otro puerto:
```bash
streamlit run app_prediccion_mejorada.py --server.port 8502
```

---

## 📝 CÓMO USAR LA APLICACIÓN

### Paso 1: Preparar tus datos

Tu archivo CSV/Excel debe tener estas columnas:

| Columna       | Descripción                          | Ejemplo                                    |
|---------------|--------------------------------------|--------------------------------------------|
| Activity Name | Tipo de evento                       | Communication Blocked, File Quarantined    |
| Process Name  | Nombre del proceso                   | powershell.exe, cmd.exe                    |
| Process Path  | Ruta completa del proceso            | C:\Windows\System32\powershell.exe         |
| Count         | Número de veces que ocurrió          | 5, 10, 1                                   |

**Formato CSV:**
- Separador: punto y coma (;)
- Codificación: UTF-8

**Ejemplo de archivo CSV:**
```csv
Activity Name;Process Name;Process Path;Count
Communication Blocked;powershell.exe;C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe;5
File Quarantined;suspicious.exe;C:\Users\Admin\AppData\Local\Temp\suspicious.exe;1
Process Execution;chrome.exe;C:\Program Files\Google\Chrome\Application\chrome.exe;10
```

### Paso 2: Cargar archivo en la aplicación

1. Abre la aplicación (se abrirá en tu navegador)
2. Haz clic en "Browse files" o arrastra tu archivo
3. Verifica que se cargó correctamente (verás el número de registros)

### Paso 3: Analizar

1. Haz clic en el botón "🚀 ANALIZAR AMENAZAS"
2. Espera unos segundos mientras el modelo procesa los datos
3. Verás los resultados del análisis

### Paso 4: Revisar resultados

La aplicación te mostrará:

- **Métricas generales:**
  - Total de registros analizados
  - Cantidad de amenazas peligrosas
  - Cantidad de registros normales
  - Amenazas críticas

- **Gráficos:**
  - Distribución de amenazas (gráfico de torta)
  - Distribución por nivel de riesgo (gráfico de barras)
  - Matriz de confusión (si hay etiquetas reales)

- **Tabla detallada:**
  - Cada registro con su predicción
  - Nivel de riesgo (Crítico, Alto, Medio, Bajo)
  - Probabilidad de peligro

- **Top 10 amenazas más críticas:**
  - Lista expandible con las 10 amenazas más peligrosas

### Paso 5: Filtrar resultados

En la barra lateral puedes:

- **Ajustar umbral de probabilidad:** Mostrar solo amenazas con probabilidad mayor a X%
- **Mostrar solo peligrosos:** Ocultar los registros seguros
- Ver información del modelo

### Paso 6: Descargar resultados

Tienes dos opciones de descarga:

1. **📥 Descargar Todos los Resultados:**
   - Archivo CSV con todos los registros analizados
   - Incluye predicciones y probabilidades

2. **🔴 Descargar Solo Amenazas Peligrosas:**
   - Archivo CSV solo con amenazas detectadas
   - Útil para reportes de seguridad

---

## 🎨 NIVELES DE RIESGO

La aplicación clasifica las amenazas en 4 niveles:

| Emoji | Nivel    | Probabilidad | Descripción                        |
|-------|----------|--------------|-----------------------------------|
| 🔴    | CRÍTICO  | ≥ 80%        | Amenaza muy peligrosa - Acción inmediata |
| 🟠    | ALTO     | 60-79%       | Amenaza significativa - Revisar pronto   |
| 🟡    | MEDIO    | 40-59%       | Posible amenaza - Monitorear            |
| 🟢    | BAJO     | < 40%        | Bajo riesgo - Proceso normal             |

---

## 📊 INTERPRETACIÓN DE MÉTRICAS

### Accuracy (Exactitud)
- Porcentaje de predicciones correctas
- **Modelo actual: 82.5%**
- Significa que 8 de cada 10 predicciones son correctas

### Precision (Precisión)
- De todas las amenazas que predijo, ¿cuántas eran realmente peligrosas?
- **Modelo actual: 77.7%**
- Evita falsos positivos

### Recall (Sensibilidad)
- De todas las amenazas reales, ¿cuántas detectó el modelo?
- **Modelo actual: 89.9%**
- Detecta 9 de cada 10 amenazas reales

### F1-Score
- Balance entre Precision y Recall
- Métrica combinada de rendimiento

---

## ❓ SOLUCIÓN DE PROBLEMAS

### Error: "No se encontró el archivo 'mejor_modelo.pkl'"
**Solución:** Asegúrate de que `mejor_modelo.pkl` esté en la misma carpeta que `app_prediccion_mejorada.py`

### Error: "No module named 'streamlit'"
**Solución:** Instala las librerías:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly openpyxl
```

### Error: "Faltan las siguientes columnas: ..."
**Solución:** Tu CSV debe tener las columnas requeridas:
- Activity Name
- Process Name
- Process Path
- Count

Verifica que los nombres sean exactos (mayúsculas y espacios)

### La aplicación no se abre automáticamente
**Solución:** Abre manualmente tu navegador y ve a:
```
http://localhost:8501
```

### El puerto 8501 está ocupado
**Solución:** Usa otro puerto:
```bash
streamlit run app_prediccion_mejorada.py --server.port 8502
```
Y abre: http://localhost:8502

### Aparece "UnicodeDecodeError" al cargar CSV
**Solución:** Guarda tu CSV con codificación UTF-8 o intenta cambiar el separador a punto y coma (;)

---

## 💡 CONSEJOS Y MEJORES PRÁCTICAS

### Para obtener mejores resultados:

1. **Limpia tus datos:**
   - Elimina registros duplicados
   - Verifica que no haya valores vacíos en columnas importantes

2. **Usa separador punto y coma (;):**
   - Es el más compatible con este sistema
   - Evita problemas con comas en los textos

3. **Revisa manualmente los casos críticos:**
   - Amenazas con probabilidad > 90% requieren atención inmediata
   - Investiga los procesos desconocidos en rutas sospechosas

4. **Actualiza el modelo periódicamente:**
   - El modelo aprende de datos históricos
   - Entrénalo con nuevos datos cada cierto tiempo

5. **Filtra por nivel de riesgo:**
   - Enfócate primero en amenazas críticas y altas
   - Las amenazas medias pueden ser falsos positivos

6. **Descarga resultados regularmente:**
   - Mantén un historial de amenazas detectadas
   - Útil para auditorías y reportes de seguridad

---

## 📈 EJEMPLO DE USO COMPLETO

### Escenario: Análisis diario de logs

1. **Exporta logs del antivirus** (FortiEDR, Symantec, etc.) en formato CSV

2. **Abre CMD y ejecuta:**
   ```bash
   cd C:\DetectorAmenazas
   streamlit run app_prediccion_mejorada.py
   ```

3. **Carga el archivo** en la aplicación web (ej: logs_04nov2025.csv)

4. **Haz clic en "ANALIZAR AMENAZAS"**

5. **Revisa los resultados:**
   - Total: 500 registros
   - Peligrosos: 87 (17.4%)
   - No Peligrosos: 413 (82.6%)
   - Críticos: 12 (2.4%)

6. **Investiga las 12 amenazas críticas** en el Top 10

7. **Filtra solo peligrosos** con probabilidad > 70%

8. **Descarga CSV con amenazas peligrosas** para compartir con el equipo

9. **Toma acciones:**
   - Bloquea procesos sospechosos
   - Actualiza reglas del firewall
   - Documenta incidentes

---

## 🔐 CARACTERÍSTICAS DEL MODELO

- **Algoritmo:** XGBoost (Gradient Boosting)
- **Features:** 20 características extraídas
- **Dataset de entrenamiento:** 5,153 logs de FortiEDR
- **Accuracy:** 82.5%
- **Recall:** 89.9% (alta capacidad de detección)
- **Actualización:** Noviembre 2025

---

## 📞 SOPORTE Y CONTACTO

**Universidad Nacional del Altiplano - UNAP**

Para soporte técnico o consultas:
- Revisa la documentación en esta guía
- Consulta la sección de "Solución de Problemas"

---

## 🚀 PRÓXIMAS MEJORAS

- [ ] Soporte para más formatos de antivirus
- [ ] Análisis en tiempo real
- [ ] Dashboard de monitoreo continuo
- [ ] Integración con sistemas SIEM
- [ ] Exportar reportes en PDF
- [ ] Detección de patrones de ataque
- [ ] Alertas automáticas por email

---

## 📄 LICENCIA Y USO

Este sistema fue desarrollado con fines educativos y de investigación para la Universidad Nacional del Altiplano.

**Uso permitido:**
✅ Análisis de seguridad interno
✅ Investigación académica
✅ Reportes de seguridad
✅ Auditorías de sistemas

**Uso NO permitido:**
❌ Distribución comercial sin autorización
❌ Modificación del código fuente sin créditos
❌ Uso malicioso o ilegal

---

## 📚 REFERENCIAS Y RECURSOS

- **Documentación de Streamlit:** https://docs.streamlit.io/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Scikit-learn Guide:** https://scikit-learn.org/stable/

---

## ✅ CHECKLIST DE INSTALACIÓN

Antes de usar la aplicación, verifica:

- [ ] Python 3.8+ instalado
- [ ] Todas las librerías instaladas (`pip install ...`)
- [ ] Archivos en la misma carpeta:
  - [ ] app_prediccion_mejorada.py
  - [ ] mejor_modelo.pkl
- [ ] CSV con las columnas correctas:
  - [ ] Activity Name
  - [ ] Process Name
  - [ ] Process Path
  - [ ] Count
- [ ] Separador de CSV es punto y coma (;)

---

**¡Listo para detectar amenazas! 🛡️**

*Última actualización: Noviembre 2025*
*Versión: 1.0*

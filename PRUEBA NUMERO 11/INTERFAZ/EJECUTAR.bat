@echo off
title Detector de Amenazas - UNAP
color 0A

echo ====================================
echo   DETECTOR DE AMENAZAS - UNAP
echo   Sistema de Machine Learning
echo ====================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

echo [INFO] Verificando instalacion de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH
    echo [INFO] Descarga Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python encontrado
echo.

echo [INFO] Verificando librerías necesarias...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Streamlit no esta instalado
    echo [INFO] Instalando dependencias...
    pip install streamlit pandas numpy scikit-learn xgboost plotly openpyxl
    if errorlevel 1 (
        echo [ERROR] Error al instalar dependencias
        pause
        exit /b 1
    )
    echo [OK] Dependencias instaladas correctamente
) else (
    echo [OK] Librerías encontradas
)
echo.

echo [INFO] Verificando archivos necesarios...
if not exist "app_prediccion_mejorada.py" (
    echo [ERROR] No se encuentra app_prediccion_mejorada.py
    echo [INFO] Asegurate de que el archivo este en esta carpeta
    pause
    exit /b 1
)
if not exist "mejor_modelo.pkl" (
    echo [ERROR] No se encuentra mejor_modelo.pkl
    echo [INFO] Asegurate de que el modelo este en esta carpeta
    pause
    exit /b 1
)
echo [OK] Archivos encontrados
echo.

echo ====================================
echo   INICIANDO APLICACION...
echo ====================================
echo.
echo [INFO] La aplicacion se abrira en tu navegador
echo [INFO] URL: http://localhost:8501
echo.
echo [IMPORTANTE] NO CIERRES ESTA VENTANA
echo Para detener la aplicacion, presiona Ctrl+C
echo.

REM Ejecutar Streamlit
streamlit run app_prediccion_mejorada.py

pause

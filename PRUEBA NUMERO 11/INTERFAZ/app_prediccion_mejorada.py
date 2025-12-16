import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import base64

# Configuración de la página
st.set_page_config(
    page_title="Detector de Amenazas - UNAP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f4788;
        font-weight: 700;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .dataframe {
        font-size: 12px;
    }
    .stDownloadButton > button {
        background-color: #1f4788;
        color: white;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #163661;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    try:
        with open('mejor_modelo.pkl', 'rb') as file:
            modelo_data = pickle.load(file)
        return modelo_data
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'mejor_modelo.pkl'. Asegúrate de que esté en la misma carpeta que la aplicación.")
        return None

# Función para extraer características
def extraer_caracteristicas(df):
    """Extrae todas las características necesarias para el modelo"""
    
    # ✅ Conversión segura: evita errores si hay valores numéricos o nulos
    df = df.astype({
        'Activity Name': 'string',
        'Process Name': 'string',
        'Process Path': 'string'
    })
    
    df_features = pd.DataFrame()
    
    # 1. Características de Activity Name
    df_features['is_comm_blocked'] = df['Activity Name'].str.contains('Communication Blocked', case=False, na=False).astype(int)
    df_features['is_file_blocked'] = df['Activity Name'].str.contains('File Blocked', case=False, na=False).astype(int)
    df_features['is_file_quarantined'] = df['Activity Name'].str.contains('File Quarantined', case=False, na=False).astype(int)
    df_features['is_exfiltration'] = df['Activity Name'].str.contains('Exfiltration', case=False, na=False).astype(int)
    
    # 2. Características de Process Name
    df_features['is_powershell'] = df['Process Name'].str.contains('powershell', case=False, na=False).astype(int)
    df_features['is_cmd'] = df['Process Name'].str.contains('cmd', case=False, na=False).astype(int)
    df_features['is_rundll32'] = df['Process Name'].str.contains('rundll32', case=False, na=False).astype(int)
    df_features['is_wscript'] = df['Process Name'].str.contains('wscript|cscript', case=False, na=False).astype(int)
    df_features['is_regsvr32'] = df['Process Name'].str.contains('regsvr32', case=False, na=False).astype(int)
    
    # 3. Características de rutas sospechosas
    df_features['path_temp'] = df['Process Path'].str.contains('temp|tmp', case=False, na=False).astype(int)
    df_features['path_appdata'] = df['Process Path'].str.contains('appdata', case=False, na=False).astype(int)
    df_features['path_roaming'] = df['Process Path'].str.contains('roaming', case=False, na=False).astype(int)
    df_features['path_downloads'] = df['Process Path'].str.contains('downloads', case=False, na=False).astype(int)
    df_features['path_desktop'] = df['Process Path'].str.contains('desktop', case=False, na=False).astype(int)
    df_features['path_system32'] = df['Process Path'].str.contains('system32', case=False, na=False).astype(int)
    
    # 4. Características numéricas
    df_features['count'] = df['Count'].fillna(1)
    df_features['count_log'] = np.log1p(df_features['count'])
    
    # 5. Características de longitud de rutas
    df_features['path_length'] = df['Process Path'].fillna('').str.len()
    df_features['path_depth'] = df['Process Path'].fillna('').str.count('\\\\')
    
    # 6. Características combinadas
    df_features['suspicious_path_score'] = (
        df_features['path_temp'] + 
        df_features['path_appdata'] + 
        df_features['path_roaming'] + 
        df_features['path_downloads']
    )
    
    return df_features


# Función para realizar predicciones
def predecir_amenazas(df, modelo_data):
    """Realiza predicciones sobre el dataframe"""
    # El pkl contiene directamente el modelo (XGBClassifier/otro)
    modelo = modelo_data

    # Extraer características
    X = extraer_caracteristicas(df)
    
    # ✅ Asegurar columnas según el modelo, pero sin romper si no hay feature_names
    try:
        booster = modelo.get_booster()
        feature_names = getattr(booster, "feature_names", None)
    except Exception:
        feature_names = None

    if not feature_names:
        feature_names = list(X.columns)
    else:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

    X = X[feature_names]
    
    # ==========================
    # Probabilidades y Predicción
    # ==========================
    prob_peligro = None
    try:
        probas = modelo.predict_proba(X)
        if probas.ndim == 1:
            # Caso raro: devuelve vector 1D (ya positivo)
            prob_peligro = probas
        else:
            # Elegir la columna de la clase "peligrosa"
            idx = None
            if hasattr(modelo, "classes_"):
                clases = list(modelo.classes_)
                if 1 in clases:
                    idx = clases.index(1)
                elif "Peligroso" in clases:
                    idx = clases.index("Peligroso")
            # Si no encontramos índice, usar la última columna (suele ser la clase positiva)
            if idx is None:
                idx = probas.shape[1] - 1
            prob_peligro = probas[:, idx]
    except Exception:
        # Si no hay predict_proba, caer a predicciones directas y 0.5 como marcador
        pred = modelo.predict(X)
        prob_peligro = (pred == 1).astype(float)

    # Predicción binaria con umbral 0.5 (consistente)
    # Ajustar umbral de decisión (más alto para reducir falsos positivos)
    umbral = 0.53
    predicciones = (prob_peligro >= umbral).astype(int)


    # Agregar resultados al dataframe original
    df_resultado = df.copy()
    df_resultado['Prediccion'] = predicciones
    df_resultado['Prediccion_Texto'] = df_resultado['Prediccion'].map({1: '🔴 PELIGROSO', 0: '🟢 NO PELIGROSO'})
    df_resultado['Probabilidad_Peligro'] = prob_peligro
    df_resultado['Probabilidad_Texto'] = (df_resultado['Probabilidad_Peligro'] * 100).round(2).astype(str) + '%'
    
    # Clasificar por nivel de riesgo
    def clasificar_riesgo(prob):
        if prob >= 0.8:
            return '🔴 CRÍTICO'
        elif prob >= 0.6:
            return '🟠 ALTO'
        elif prob >= 0.4:
            return '🟡 MEDIO'
        else:
            return '🟢 BAJO'
    
    df_resultado['Nivel_Riesgo'] = df_resultado['Probabilidad_Peligro'].apply(clasificar_riesgo)
    
    return df_resultado

# Función para crear matriz de confusión
def crear_matriz_confusion(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Peligroso', 'Peligroso'],
        y=['No Peligroso', 'Peligroso'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title='Matriz de Confusión',
        xaxis_title='Predicción',
        yaxis_title='Real',
        height=400,
        width=500
    )
    
    return fig

# Función para crear gráfico de distribución
def crear_grafico_distribucion(df_resultado):
    counts = df_resultado['Prediccion_Texto'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,
        marker=dict(colors=['#28a745', '#dc3545']),
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title='Distribución de Amenazas Detectadas',
        height=400,
        showlegend=True
    )
    
    return fig

# Función para crear gráfico de niveles de riesgo
def crear_grafico_riesgo(df_resultado):
    counts = df_resultado['Nivel_Riesgo'].value_counts()
    order = ['🔴 CRÍTICO', '🟠 ALTO', '🟡 MEDIO', '🟢 BAJO']
    counts = counts.reindex(order, fill_value=0)
    
    colors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
    
    fig = go.Figure(data=[go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Distribución por Nivel de Riesgo',
        xaxis_title='Nivel de Riesgo',
        yaxis_title='Cantidad',
        height=400,
        showlegend=False
    )
    
    return fig

# Función para descargar CSV
def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False, sep=';')
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Descargar resultados CSV</a>'

# ==================== APLICACIÓN PRINCIPAL ====================

def main():
    st.markdown("<h1 style='text-align: center;'>🛡️ DETECTOR DE AMENAZAS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>Sistema de Clasificación de Logs con Machine Learning</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Cargar el modelo entrenado
    modelo_data = cargar_modelo()
    if modelo_data is None:
        st.stop()

    # 📁 Subir archivo CSV o Excel
    uploaded_file = st.file_uploader(
        "📂 Sube tu archivo CSV o Excel con los logs a analizar",
        type=['csv', 'xlsx', 'xls'],
        help="Debe tener columnas: Activity Name, Process Name, Process Path, Count"
    )

    if uploaded_file is not None:
        try:
            # Leer el archivo subido
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=';')
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Archivo cargado correctamente: **{uploaded_file.name}** ({len(df)} registros)")

            # Verificar columnas necesarias
            columnas_requeridas = ['Activity Name', 'Process Name', 'Process Path', 'Count']
            faltantes = [col for col in columnas_requeridas if col not in df.columns]

            if faltantes:
                st.error(f"❌ Faltan las siguientes columnas: {', '.join(faltantes)}")
                st.stop()

            # Botón para analizar
            if st.button("🚀 ANALIZAR AMENAZAS", use_container_width=True, type="primary"):
                with st.spinner("🔍 Analizando con el modelo de IA..."):
                    df_resultado = predecir_amenazas(df, modelo_data)

                # Mostrar resultados simples
                st.markdown("---")
                st.markdown("## 📊 RESULTADOS DEL ANÁLISIS")

                total = len(df_resultado)
                peligrosos = (df_resultado['Prediccion'] == 1).sum()
                no_peligrosos = total - peligrosos

                st.write(f"**Total de registros:** {total}")
                st.write(f"**Peligrosos:** {peligrosos} ({peligrosos/total*100:.1f}%)")
                st.write(f"**No peligrosos:** {no_peligrosos} ({no_peligrosos/total*100:.1f}%)")

                # Mostrar tabla de resultados
                columnas_mostrar = [
                    'Prediccion_Texto',
                    'Probabilidad_Texto',
                    'Activity Name',
                    'Process Name',
                    'Process Path',
                    'Count'
                ]
                st.dataframe(df_resultado[columnas_mostrar], use_container_width=True, height=400)

                # Botón para descargar resultados
                st.markdown("### 💾 DESCARGAR RESULTADOS")
                csv_result = df_resultado.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    label="📥 Descargar archivo con resultados",
                    data=csv_result,
                    file_name=f"analisis_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")
            st.exception(e)
    else:
        st.info("👆 Sube tu archivo CSV o Excel para comenzar el análisis.")

if __name__ == "__main__":
    main()

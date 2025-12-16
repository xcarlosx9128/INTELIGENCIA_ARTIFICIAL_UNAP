import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ================================================
# CONFIGURACIÓN GENERAL
# ================================================
st.set_page_config(page_title="Predicción de Actividades Peligrosas", page_icon="⚠️", layout="wide")

st.title("🧠 Sistema de Detección de Actividades Peligrosas")
st.markdown("""
Esta interfaz permite **subir un archivo CSV** con datos de eventos o procesos
y analizarlo con el modelo de inteligencia artificial previamente entrenado.
""")

# ================================================
# CARGAR MODELO (el modelo está en la carpeta superior)
# ================================================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("../mejor_modelo.pkl")  # 👈 Ruta ajustada
        st.success("✅ Modelo cargado correctamente desde '../mejor_modelo.pkl'")
        return modelo
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo: {e}")
        return None

modelo = cargar_modelo()

# ================================================
# SUBIR ARCHIVO CSV
# ================================================
st.subheader("📤 Subir archivo CSV para analizar")
archivo = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if archivo is not None and modelo is not None:
    try:
        # Leer archivo
        df = pd.read_csv(archivo, sep=";")
        st.write(f"**Filas cargadas:** {len(df)} | **Columnas:** {len(df.columns)}")
        st.dataframe(df.head())

        # ================================================
        # PREPROCESAMIENTO BÁSICO (similar al entrenamiento)
        # ================================================
        df = df.dropna(subset=["Host Name", "Host IP", "Process Name"])
        df["Count"] = pd.to_numeric(df.get("Count", 0), errors="coerce").fillna(0)

        df["frecuencia_riesgo"] = 0
        df.loc[df["Count"] >= 5, "frecuencia_riesgo"] += 1
        df.loc[df["Count"] >= 10, "frecuencia_riesgo"] += 2

        df["actividad_riesgo"] = df["Activity Name"].astype(str).str.lower().apply(
            lambda x: 5 if "malicious" in x else (3 if "suspicious" in x else (1 if "pup" in x else 0))
        )

        df["riesgo_total"] = df["frecuencia_riesgo"] + df["actividad_riesgo"]

        # ================================================
        # PREDICCIÓN CON EL MODELO
        # ================================================
        X = df.select_dtypes(include=["number"])
        predicciones = modelo.predict(X)

        df["Resultado"] = ["⚠️ PELIGROSO" if p == 1 else "✅ NO PELIGROSO" for p in predicciones]

        # ================================================
        # RESULTADOS
        # ================================================
        st.subheader("📊 Resultados de la Predicción")
        st.dataframe(df[["Host Name", "Activity Name", "riesgo_total", "Resultado"]])

        # Contador
        total_peligrosos = (df["Resultado"] == "⚠️ PELIGROSO").sum()
        total_normales = (df["Resultado"] == "✅ NO PELIGROSO").sum()
        st.write(f"**Total peligrosos:** {total_peligrosos} | **No peligrosos:** {total_normales}")

        # ================================================
        # VISUALIZACIÓN
        # ================================================
        fig, ax = plt.subplots()
        ax.bar(["No Peligrosos", "Peligrosos"], [total_normales, total_peligrosos], color=["#4CAF50", "#F44336"])
        ax.set_ylabel("Cantidad de Registros")
        ax.set_title("Distribución de Resultados del Modelo")
        st.pyplot(fig)

        # ================================================
        # DESCARGAR RESULTADOS
        # ================================================
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar resultados en CSV",
            data=csv_bytes,
            file_name="predicciones_resultado.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error procesando el archivo: {e}")

elif modelo is None:
    st.warning("⚠️ No se pudo cargar el modelo. Verifica que el archivo 'mejor_modelo.pkl' esté en la carpeta superior (../).")
else:
    st.info("Sube un archivo CSV para comenzar el análisis.")

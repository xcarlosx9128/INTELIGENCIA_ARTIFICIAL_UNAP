"""
═══════════════════════════════════════════════════════════════════════════════
PASO 1 MEJORADO: PREPARAR Y SEPARAR DATOS CON SPLIT 80-20
═══════════════════════════════════════════════════════════════════════════════

Cambios en esta versión:
- 50 datos para PRUEBA FINAL (25 peligrosos + 25 no peligrosos) ✓
- Del resto: 80% ENTRENAMIENTO - 20% VALIDACIÓN (balanceado)
- Manejo de desbalance con undersampling

Autor: Sistema de Clasificación de Logs
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("    PASO 1 MEJORADO: PREPARACIÓN CON SPLIT 80-20")
print("="*80 + "\n")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("📁 Cargando archivo: report1761949414801.csv...")
df = pd.read_csv('report1761949414801.csv', sep=';')
print(f"✅ Datos cargados: {len(df):,} registros\n")

# ============================================================================
# 2. CREAR ETIQUETAS DE PELIGROSIDAD (BINARIO: 0 o 1)
# ============================================================================
print("🏷️  Creando etiquetas de peligrosidad...\n")

def clasificar_peligrosidad(activity_name):
    """
    Clasificar según Activity Name de FortiEDR
    PELIGROSO (1): Malicious, Suspicious
    NO PELIGROSO (0): PUP, otros
    """
    if pd.isna(activity_name):
        return 0
    
    activity_name = str(activity_name).lower()
    
    # PELIGROSO
    if 'malicious' in activity_name or 'suspicious' in activity_name:
        return 1
    # NO PELIGROSO (PUP no es tan crítico)
    else:
        return 0

df['Peligroso'] = df['Activity Name'].apply(clasificar_peligrosidad)
df['Etiqueta_Texto'] = df['Peligroso'].map({0: 'NO_PELIGROSO', 1: 'PELIGROSO'})

# Estadísticas iniciales
total = len(df)
peligrosos = (df['Peligroso'] == 1).sum()
no_peligrosos = (df['Peligroso'] == 0).sum()

print(f"📊 DISTRIBUCIÓN ORIGINAL:")
print(f"   🔴 PELIGROSOS:     {peligrosos:5,} ({peligrosos/total*100:5.1f}%)")
print(f"   🟢 NO PELIGROSOS:  {no_peligrosos:5,} ({no_peligrosos/total*100:5.1f}%)")
print(f"   📈 RATIO: 1:{no_peligrosos/peligrosos:.2f}\n")

# ============================================================================
# 3. EXTRAER CARACTERÍSTICAS (FEATURES)
# ============================================================================
print("🔧 Extrayendo 20 características...")

# Convertir a string
df['Process Path'] = df['Process Path'].astype(str)
df['Process Name'] = df['Process Name'].astype(str)

# 1. Características del nombre del proceso (4)
df['proceso_longitud'] = df['Process Name'].str.len()
df['tiene_numeros_nombre'] = df['Process Name'].str.contains(r'\d', regex=True, na=False).astype(int)
df['nombre_aleatorio'] = df['Process Name'].str.contains(r'^[a-z]{8,}\.exe$', case=False, regex=True, na=False).astype(int)
df['nombre_muy_corto'] = (df['proceso_longitud'] <= 5).astype(int)

# 2. Características de la ruta (7)
df['ruta_appdata'] = df['Process Path'].str.contains('AppData', case=False, na=False).astype(int)
df['ruta_roaming'] = df['Process Path'].str.contains('Roaming', case=False, na=False).astype(int)
df['ruta_temp'] = df['Process Path'].str.contains('Temp|tmp', case=False, regex=True, na=False).astype(int)
df['ruta_downloads'] = df['Process Path'].str.contains('Downloads|Descargas', case=False, regex=True, na=False).astype(int)
df['ruta_desktop'] = df['Process Path'].str.contains('Desktop|Escritorio', case=False, regex=True, na=False).astype(int)
df['ruta_system32'] = df['Process Path'].str.contains('System32', case=False, na=False).astype(int)
df['ruta_program_files'] = df['Process Path'].str.contains('Program Files', case=False, na=False).astype(int)

# 3. Tipo de archivo (3)
df['es_exe'] = df['Process Name'].str.endswith('.exe', na=False).astype(int)
df['es_dll'] = df['Process Name'].str.endswith('.dll', na=False).astype(int)
df['es_script'] = df['Process Name'].str.contains('powershell|cmd|cscript|wscript', case=False, regex=True, na=False).astype(int)

# 4. Procesos conocidos peligrosos (1)
procesos_peligrosos = ['powershell.exe', 'cmd.exe', 'cscript.exe', 'wscript.exe', 
                       'rundll32.exe', 'regsvr32.exe', 'mshta.exe']
df['proceso_peligroso_conocido'] = df['Process Name'].str.lower().isin(procesos_peligrosos).astype(int)

# 5. Frecuencia (2)
df['frecuencia_alta'] = (df['Count'] > df['Count'].median()).astype(int)
df['log_count'] = np.log1p(df['Count'])

# 6. Características temporales (3)
df['Event Receive Time'] = pd.to_datetime(df['Event Receive Time'], format='%d/%m/%Y %H:%M', errors='coerce')
df['hora'] = df['Event Receive Time'].dt.hour.fillna(12).astype(int)
df['dia_semana'] = df['Event Receive Time'].dt.dayofweek.fillna(0).astype(int)
df['es_horario_nocturno'] = ((df['hora'] >= 22) | (df['hora'] <= 6)).astype(int)

print(f"✅ 20 características extraídas\n")

# Lista de features
FEATURES = [
    'proceso_longitud', 'tiene_numeros_nombre', 'nombre_aleatorio', 'nombre_muy_corto',
    'ruta_appdata', 'ruta_roaming', 'ruta_temp', 'ruta_downloads', 'ruta_desktop',
    'ruta_system32', 'ruta_program_files', 'es_exe', 'es_dll', 'es_script',
    'proceso_peligroso_conocido', 'frecuencia_alta', 'log_count',
    'hora', 'dia_semana', 'es_horario_nocturno'
]

# ============================================================================
# 4. SEPARAR DATOS
# ============================================================================
print("="*80)
print("📊 SEPARANDO DATOS (NUEVA ESTRATEGIA 80-20)")
print("="*80 + "\n")

# Separar por clase
df_peligrosos = df[df['Peligroso'] == 1].copy()
df_no_peligrosos = df[df['Peligroso'] == 0].copy()

print(f"Disponibles:")
print(f"  🔴 Peligrosos: {len(df_peligrosos):,}")
print(f"  🟢 No Peligrosos: {len(df_no_peligrosos):,}\n")

# ----------------------------------------------------------------------------
# PASO 1: PRUEBA FINAL - 50 datos (25 peligrosos + 25 no peligrosos)
# ----------------------------------------------------------------------------
print("📦 PASO 1: Separando 50 datos para PRUEBA FINAL (25+25)...")

prueba_final_peligrosos = df_peligrosos.sample(n=25, random_state=42)
prueba_final_no_peligrosos = df_no_peligrosos.sample(n=25, random_state=42)

df_prueba_final = pd.concat([prueba_final_peligrosos, prueba_final_no_peligrosos])
df_prueba_final = df_prueba_final.sample(frac=1, random_state=42)

print(f"✅ Prueba final: {len(df_prueba_final)} registros")
print(f"   - Peligrosos: {(df_prueba_final['Peligroso']==1).sum()}")
print(f"   - No Peligrosos: {(df_prueba_final['Peligroso']==0).sum()}\n")

# Remover de los datasets originales
df_peligrosos = df_peligrosos.drop(prueba_final_peligrosos.index)
df_no_peligrosos = df_no_peligrosos.drop(prueba_final_no_peligrosos.index)

print(f"Datos restantes después de separar prueba final:")
print(f"  🔴 Peligrosos: {len(df_peligrosos):,}")
print(f"  🟢 No Peligrosos: {len(df_no_peligrosos):,}\n")

# =============================================================================
# 💾 FUNCIÓN NUEVA: GUARDAR FILAS CRUDAS DE LA MUESTRA FINAL (SIN COLUMNAS NUEVAS)
# =============================================================================
def guardar_prueba_final_cruda(df_original, df_prueba_final, filename='datos_prueba_final_crudo.csv'):
    """
    Guarda las filas crudas correspondientes a los índices seleccionados en df_prueba_final,
    incluyendo solo las columnas originales del CSV de entrada (sin columnas derivadas).
    """
    # Columnas originales al momento de cargar el CSV
    columnas_originales = [
        col for col in df_original.columns
        if col not in [
            'Peligroso', 'Etiqueta_Texto',
            'proceso_longitud', 'tiene_numeros_nombre', 'nombre_aleatorio', 'nombre_muy_corto',
            'ruta_appdata', 'ruta_roaming', 'ruta_temp', 'ruta_downloads', 'ruta_desktop',
            'ruta_system32', 'ruta_program_files', 'es_exe', 'es_dll', 'es_script',
            'proceso_peligroso_conocido', 'frecuencia_alta', 'log_count',
            'hora', 'dia_semana', 'es_horario_nocturno'
        ]
    ]

    # Recuperar filas crudas (sin columnas nuevas)
    df_crudo = df_original.loc[df_prueba_final.index, columnas_originales].copy()

    # Guardar CSV crudo
    df_crudo.to_csv(filename, index=False, sep=';')
    print(f"✅ {filename} ({len(df_crudo):,} registros) - guardado solo con columnas originales.")

# 📢 LLAMADA FINAL
guardar_prueba_final_cruda(df, df_prueba_final)



# ----------------------------------------------------------------------------
# PASO 2: BALANCEAR DATOS (Undersampling si es necesario)
# ----------------------------------------------------------------------------
print("📦 PASO 2: Balanceando datos antes del split...")

n_peligrosos = len(df_peligrosos)
n_no_peligrosos = len(df_no_peligrosos)
ratio_actual = max(n_peligrosos, n_no_peligrosos) / min(n_peligrosos, n_no_peligrosos)

print(f"Ratio actual: 1:{ratio_actual:.2f}")

# Si ratio > 2, aplicar undersampling
if ratio_actual > 2:
    print(f"⚠️  Desbalance detectado. Aplicando undersampling...\n")
    
    if n_peligrosos > n_no_peligrosos:
        # Reducir peligrosos
        n_target = int(n_no_peligrosos * 2)  # Máximo ratio 2:1
        df_peligrosos = df_peligrosos.sample(n=n_target, random_state=42)
        print(f"   Peligrosos reducidos: {len(df_peligrosos):,}")
    else:
        # Reducir no peligrosos
        n_target = int(n_peligrosos * 2)
        df_no_peligrosos = df_no_peligrosos.sample(n=n_target, random_state=42)
        print(f"   No Peligrosos reducidos: {len(df_no_peligrosos):,}")
    
    nuevo_ratio = max(len(df_peligrosos), len(df_no_peligrosos)) / min(len(df_peligrosos), len(df_no_peligrosos))
    print(f"   Nuevo ratio: 1:{nuevo_ratio:.2f}\n")

# Combinar datos balanceados
df_resto = pd.concat([df_peligrosos, df_no_peligrosos])
df_resto = df_resto.sample(frac=1, random_state=42)

print(f"✅ Datos balanceados:")
print(f"   Total: {len(df_resto):,}")
print(f"   - Peligrosos: {(df_resto['Peligroso']==1).sum():,}")
print(f"   - No Peligrosos: {(df_resto['Peligroso']==0).sum():,}\n")

# ----------------------------------------------------------------------------
# PASO 3: SPLIT 80-20 para ENTRENAMIENTO y VALIDACIÓN
# ----------------------------------------------------------------------------
print("📦 PASO 3: Split 80-20 para Entrenamiento y Validación...")

# Separar X e y
X_resto = df_resto[FEATURES].fillna(0)
y_resto = df_resto['Peligroso']

# Split estratificado (mantiene proporciones)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_resto, y_resto, df_resto.index,
    test_size=0.20,  # 20% para validación
    random_state=42,
    stratify=y_resto  # Mantiene proporciones
)

# Reconstruir dataframes
df_entrenamiento = df_resto.loc[idx_train].copy()
df_validacion = df_resto.loc[idx_val].copy()

print(f"\n✅ ENTRENAMIENTO (80%):")
print(f"   Total: {len(df_entrenamiento):,}")
print(f"   - Peligrosos: {(df_entrenamiento['Peligroso']==1).sum():,} ({(df_entrenamiento['Peligroso']==1).sum()/len(df_entrenamiento)*100:.1f}%)")
print(f"   - No Peligrosos: {(df_entrenamiento['Peligroso']==0).sum():,} ({(df_entrenamiento['Peligroso']==0).sum()/len(df_entrenamiento)*100:.1f}%)")

print(f"\n✅ VALIDACIÓN (20%):")
print(f"   Total: {len(df_validacion):,}")
print(f"   - Peligrosos: {(df_validacion['Peligroso']==1).sum():,} ({(df_validacion['Peligroso']==1).sum()/len(df_validacion)*100:.1f}%)")
print(f"   - No Peligrosos: {(df_validacion['Peligroso']==0).sum():,} ({(df_validacion['Peligroso']==0).sum()/len(df_validacion)*100:.1f}%)\n")

# ============================================================================
# 5. GUARDAR ARCHIVOS
# ============================================================================
print("="*80)
print("💾 GUARDANDO ARCHIVOS")
print("="*80 + "\n")

# ENTRENAMIENTO
df_train_export = df_entrenamiento[['Process Name', 'Process Path', 'Activity Name', 
                                      'Host Name', 'Count', 'Peligroso', 'Etiqueta_Texto'] + FEATURES].copy()
df_train_export.to_csv('datos_entrenamiento.csv', index=False, sep=';')
print(f"✅ datos_entrenamiento.csv ({len(df_train_export):,} registros)")

# VALIDACIÓN
df_val_export = df_validacion[['Process Name', 'Process Path', 'Activity Name', 
                                 'Host Name', 'Count', 'Peligroso', 'Etiqueta_Texto'] + FEATURES].copy()
df_val_export.to_csv('datos_validacion.csv', index=False, sep=';')
print(f"✅ datos_validacion.csv ({len(df_val_export):,} registros)")

# PRUEBA FINAL
X_test = df_prueba_final[FEATURES].fillna(0)
y_test = df_prueba_final['Peligroso']

df_test_export = df_prueba_final[['Process Name', 'Process Path', 'Activity Name', 
                                    'Host Name', 'Count', 'Peligroso', 'Etiqueta_Texto'] + FEATURES].copy()
df_test_export.to_csv('datos_prueba_final.csv', index=False, sep=';')
print(f"✅ datos_prueba_final.csv ({len(df_test_export):,} registros) 🔒\n")

# Guardar arrays NumPy
np.savez('datos_preparados.npz',
         X_train=X_train.values, y_train=y_train.values,
         X_val=X_val.values, y_val=y_val.values,
         X_test=X_test.values, y_test=y_test.values,
         feature_names=FEATURES)
print(f"✅ datos_preparados.npz")

# ============================================================================
# 6. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("✅ PREPARACIÓN COMPLETADA CON SPLIT 80-20")
print("="*80 + "\n")

print("📊 RESUMEN DE DATOS PREPARADOS:\n")
print(f"{'Dataset':<20} {'Total':<10} {'Peligrosos':<15} {'No Peligrosos':<15} {'% Dataset'}")
print("-" * 75)
print(f"{'Entrenamiento':<20} {len(X_train):<10} {(y_train==1).sum():<15} {(y_train==0).sum():<15} 80%")
print(f"{'Validación':<20} {len(X_val):<10} {(y_val==1).sum():<15} {(y_val==0).sum():<15} 20%")
print(f"{'Prueba Final':<20} {len(X_test):<10} {(y_test==1).sum():<15} {(y_test==0).sum():<15} ---")
print("-" * 75)
total_todos = len(X_train) + len(X_val) + len(X_test)
print(f"{'TOTAL':<20} {total_todos:<10}")

print("\n📁 ARCHIVOS GENERADOS:\n")
print("  1. datos_entrenamiento.csv    - Para entrenar modelos (80%)")
print("  2. datos_validacion.csv        - Para evaluar durante entrenamiento (20%)")
print("  3. datos_prueba_final.csv      - Para prueba final (25+25)")
print("  4. datos_preparados.npz        - Arrays NumPy listos")

print("\n🎯 MEJORAS EN ESTA VERSIÓN:")
print("  ✅ Validación con más datos (20% en vez de 40 fijos)")
print("  ✅ Split estratificado (mantiene proporciones)")
print("  ✅ Menos overfitting esperado")

print("\n🚀 SIGUIENTE PASO:")
print("   Ejecuta: python paso2_entrenar_modelos.py")
print("\n" + "="*80 + "\n")
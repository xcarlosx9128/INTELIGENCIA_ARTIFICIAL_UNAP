"""
═══════════════════════════════════════════════════════════════════════════════
PASO 2: ENTRENAR MODELOS DE MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════

Este script:
1. Carga datos preparados (datos_preparados.npz)
2. Entrena 4 modelos:
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
3. Evalúa cada modelo en:
   - Datos de ENTRENAMIENTO (matriz confusión)
   - Datos de VALIDACIÓN (matriz confusión)
4. Muestra métricas completas: Accuracy, Precision, Recall, F1
5. Guarda el mejor modelo

Autor: Sistema de Clasificación de Logs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("         PASO 2: ENTRENAMIENTO DE MODELOS DE MACHINE LEARNING")
print("="*80 + "\n")

# ============================================================================
# 1. CARGAR DATOS PREPARADOS
# ============================================================================
print("📁 Cargando datos preparados...")

data = np.load('datos_preparados.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"✅ Datos cargados:")
print(f"   Entrenamiento: {len(X_train):,} registros")
print(f"   Validación: {len(X_val):,} registros")
print(f"   Prueba Final: {len(X_test):,} registros (🔒 NO usaremos ahora)")
print(f"   Features: {len(feature_names)}\n")

# ============================================================================
# 2. DEFINIR MODELOS A ENTRENAR
# ============================================================================
print("="*80)
print("🤖 DEFINIENDO MODELOS")
print("="*80 + "\n")

modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
}

print("Modelos a entrenar:")
for i, nombre in enumerate(modelos.keys(), 1):
    print(f"  {i}. {nombre}")
print()

# ============================================================================
# 3. FUNCIÓN PARA EVALUAR MODELOS
# ============================================================================
def evaluar_modelo(nombre, modelo, X_train, y_train, X_val, y_val):
    """Evaluar modelo y retornar métricas completas"""
    
    print("="*80)
    print(f"🔥 ENTRENANDO: {nombre}")
    print("="*80 + "\n")
    
    # Entrenar
    print(f"⏳ Entrenando {nombre}...")
    modelo.fit(X_train, y_train)
    print(f"✅ Entrenamiento completado\n")
    
    # Predicciones
    y_train_pred = modelo.predict(X_train)
    y_val_pred = modelo.predict(X_val)
    
    # Probabilidades (para ROC-AUC)
    try:
        y_train_proba = modelo.predict_proba(X_train)[:, 1]
        y_val_proba = modelo.predict_proba(X_val)[:, 1]
    except:
        y_train_proba = None
        y_val_proba = None
    
    # ========================================================================
    # MÉTRICAS DE ENTRENAMIENTO
    # ========================================================================
    print("📊 MÉTRICAS EN ENTRENAMIENTO:")
    print("-" * 80)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    print(f"  Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    
    if y_train_proba is not None:
        train_auc = roc_auc_score(y_train, y_train_proba)
        print(f"  ROC-AUC:   {train_auc:.4f}")
    else:
        train_auc = None
    
    print("\n📊 MATRIZ DE CONFUSIÓN - ENTRENAMIENTO:")
    cm_train = confusion_matrix(y_train, y_train_pred)
    print(f"""
    Predicho:    No Peligroso  |  Peligroso
    Real:
    No Peligroso      {cm_train[0,0]:5d}      |    {cm_train[0,1]:5d}
    Peligroso         {cm_train[1,0]:5d}      |    {cm_train[1,1]:5d}
    """)
    
    # ========================================================================
    # MÉTRICAS DE VALIDACIÓN
    # ========================================================================
    print("-" * 80)
    print("📊 MÉTRICAS EN VALIDACIÓN:")
    print("-" * 80)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"  Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1-Score:  {val_f1:.4f}")
    
    if y_val_proba is not None:
        val_auc = roc_auc_score(y_val, y_val_proba)
        print(f"  ROC-AUC:   {val_auc:.4f}")
    else:
        val_auc = None
    
    print("\n📊 MATRIZ DE CONFUSIÓN - VALIDACIÓN:")
    cm_val = confusion_matrix(y_val, y_val_pred)
    print(f"""
    Predicho:    No Peligroso  |  Peligroso
    Real:
    No Peligroso      {cm_val[0,0]:5d}      |    {cm_val[0,1]:5d}
    Peligroso         {cm_val[1,0]:5d}      |    {cm_val[1,1]:5d}
    """)
    
    # ========================================================================
    # REPORTE DETALLADO DE VALIDACIÓN
    # ========================================================================
    print("-" * 80)
    print("📋 REPORTE DETALLADO - VALIDACIÓN:")
    print("-" * 80)
    print(classification_report(y_val, y_val_pred, 
                                target_names=['No Peligroso', 'Peligroso'],
                                digits=4))
    
    # ========================================================================
    # FEATURE IMPORTANCE (si está disponible)
    # ========================================================================
    if hasattr(modelo, 'feature_importances_'):
        print("-" * 80)
        print("🔍 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
        print("-" * 80)
        
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': modelo.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importances.head(10).to_string(index=False))
        print()
    
    # Retornar métricas
    return {
        'modelo': modelo,
        'train_acc': train_acc,
        'train_prec': train_prec,
        'train_rec': train_rec,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'val_acc': val_acc,
        'val_prec': val_prec,
        'val_rec': val_rec,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'cm_train': cm_train,
        'cm_val': cm_val
    }

# ============================================================================
# 4. ENTRENAR TODOS LOS MODELOS
# ============================================================================
resultados = {}

for nombre, modelo in modelos.items():
    resultado = evaluar_modelo(nombre, modelo, X_train, y_train, X_val, y_val)
    resultados[nombre] = resultado
    
    # Guardar modelo
    filename = f'modelo_{nombre.replace(" ", "_").lower()}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(resultado['modelo'], f)
    print(f"💾 Modelo guardado: {filename}\n")

# ============================================================================
# 5. COMPARACIÓN FINAL DE MODELOS
# ============================================================================
print("="*80)
print("🏆 COMPARACIÓN FINAL DE MODELOS")
print("="*80 + "\n")

print("📊 MÉTRICAS EN VALIDACIÓN (lo más importante):\n")
print(f"{'Modelo':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 90)

for nombre, resultado in resultados.items():
    auc_str = f"{resultado['val_auc']:.4f}" if resultado['val_auc'] else "N/A"
    print(f"{nombre:<20} {resultado['val_acc']:<12.4f} {resultado['val_prec']:<12.4f} "
          f"{resultado['val_rec']:<12.4f} {resultado['val_f1']:<12.4f} {auc_str:<12}")

# Encontrar mejor modelo por F1-Score
mejor_modelo_nombre = max(resultados.items(), key=lambda x: x[1]['val_f1'])[0]
mejor_modelo = resultados[mejor_modelo_nombre]

print("\n" + "="*80)
print(f"🥇 MEJOR MODELO: {mejor_modelo_nombre}")
print("="*80)
print(f"\n  Accuracy:  {mejor_modelo['val_acc']:.4f} ({mejor_modelo['val_acc']*100:.2f}%)")
print(f"  Precision: {mejor_modelo['val_prec']:.4f}")
print(f"  Recall:    {mejor_modelo['val_rec']:.4f}")
print(f"  F1-Score:  {mejor_modelo['val_f1']:.4f}")
if mejor_modelo['val_auc']:
    print(f"  ROC-AUC:   {mejor_modelo['val_auc']:.4f}")

print(f"\n💾 Guardado como: modelo_{mejor_modelo_nombre.replace(' ', '_').lower()}.pkl")

# Guardar también como "mejor_modelo.pkl"
with open('mejor_modelo.pkl', 'wb') as f:
    pickle.dump(mejor_modelo['modelo'], f)
print(f"💾 También guardado como: mejor_modelo.pkl")

# ============================================================================
# 6. GUARDAR RESUMEN EN CSV
# ============================================================================
print("\n" + "="*80)
print("💾 GUARDANDO RESUMEN")
print("="*80 + "\n")

resumen_df = pd.DataFrame([
    {
        'Modelo': nombre,
        'Train_Accuracy': res['train_acc'],
        'Train_Precision': res['train_prec'],
        'Train_Recall': res['train_rec'],
        'Train_F1': res['train_f1'],
        'Train_AUC': res['train_auc'] if res['train_auc'] else 0,
        'Val_Accuracy': res['val_acc'],
        'Val_Precision': res['val_prec'],
        'Val_Recall': res['val_rec'],
        'Val_F1': res['val_f1'],
        'Val_AUC': res['val_auc'] if res['val_auc'] else 0
    }
    for nombre, res in resultados.items()
])

resumen_df.to_csv('resumen_modelos.csv', index=False)
print("✅ resumen_modelos.csv guardado")

# ============================================================================
# 7. ANÁLISIS DE OVERFITTING
# ============================================================================
print("\n" + "="*80)
print("🔍 ANÁLISIS DE OVERFITTING")
print("="*80 + "\n")

print(f"{'Modelo':<20} {'Train Acc':<12} {'Val Acc':<12} {'Diferencia':<12} {'Overfitting?':<15}")
print("-" * 75)

for nombre, resultado in resultados.items():
    diff = resultado['train_acc'] - resultado['val_acc']
    overfit = "⚠️ SÍ" if diff > 0.10 else "✅ NO"
    print(f"{nombre:<20} {resultado['train_acc']:<12.4f} {resultado['val_acc']:<12.4f} "
          f"{diff:<12.4f} {overfit:<15}")

print("\n💡 Si diferencia > 0.10, hay overfitting (modelo memoriza en vez de aprender)")

# ============================================================================
# 8. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80 + "\n")

print("📁 ARCHIVOS GENERADOS:\n")
print("  1. modelo_random_forest.pkl")
print("  2. modelo_gradient_boosting.pkl")
print("  3. modelo_xgboost.pkl")
print("  4. modelo_lightgbm.pkl")
print("  5. mejor_modelo.pkl (el mejor de todos)")
print("  6. resumen_modelos.csv")

print("\n🎯 SIGUIENTE PASO:")
print("   Ejecuta: python paso3_probar_modelo.py")
print("   (Usará los 50 datos separados: datos_prueba_final.csv)")

print("\n💡 PARA USAR EL MEJOR MODELO:")
print("""
import pickle
import pandas as pd

# Cargar modelo
with open('mejor_modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Cargar datos nuevos (los 50 separados)
df = pd.read_csv('datos_prueba_final.csv', sep=';')

# Extraer features (las mismas 20 que usamos)
FEATURES = ['proceso_longitud', 'tiene_numeros_nombre', ...]
X_nuevo = df[FEATURES].fillna(0)

# Predecir
predicciones = modelo.predict(X_nuevo)
# 0 = NO PELIGROSO, 1 = PELIGROSO
""")

print("\n" + "="*80 + "\n")
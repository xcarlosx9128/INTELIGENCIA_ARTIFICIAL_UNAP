"""
═══════════════════════════════════════════════════════════════════════════════
PASO 2.5: GENERAR VISUALIZACIONES DE LOS MODELOS
═══════════════════════════════════════════════════════════════════════════════

Este script genera:
1. Matrices de Confusión (entrenamiento + validación) para cada modelo
2. Curvas ROC comparativas
3. Gráficos de Feature Importance
4. Comparación de métricas entre modelos
5. TODO en formato PNG de alta calidad

Autor: Sistema de Clasificación de Logs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*80)
print("         GENERANDO VISUALIZACIONES DE MODELOS")
print("="*80 + "\n")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("📁 Cargando datos...")

data = np.load('datos_preparados.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
feature_names = data['feature_names']

print(f"✅ Datos cargados:")
print(f"   Entrenamiento: {len(X_train):,} registros")
print(f"   Validación: {len(X_val):,} registros\n")

# ============================================================================
# 2. CARGAR MODELOS
# ============================================================================
print("🤖 Cargando modelos entrenados...")

modelos = {}
nombres_modelos = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']

for nombre in nombres_modelos:
    with open(f'modelo_{nombre}.pkl', 'rb') as f:
        modelos[nombre] = pickle.load(f)
    print(f"   ✅ {nombre}")

print()

# ============================================================================
# 3. FUNCIÓN PARA CREAR MATRIZ DE CONFUSIÓN
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, title, filename, labels=['No Peligroso', 'Peligroso']):
    """Crear y guardar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Cantidad'}, ax=ax)
    
    # Agregar anotaciones con cantidad y porcentaje
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black',
                   fontsize=14, fontweight='bold')
    
    # Configuración
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Agregar métricas en el título
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics_text = f'Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}'
    ax.text(1, -0.15, metrics_text, transform=ax.transAxes,
           fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

# ============================================================================
# 4. GENERAR MATRICES DE CONFUSIÓN PARA CADA MODELO
# ============================================================================
print("="*80)
print("📊 GENERANDO MATRICES DE CONFUSIÓN")
print("="*80 + "\n")

nombres_legibles = {
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

for nombre_archivo, modelo in modelos.items():
    nombre_legible = nombres_legibles[nombre_archivo]
    
    print(f"📈 Generando gráficos para: {nombre_legible}")
    
    # Predicciones
    y_train_pred = modelo.predict(X_train)
    y_val_pred = modelo.predict(X_val)
    
    # Matriz de confusión - ENTRENAMIENTO
    filename_train = f'cm_train_{nombre_archivo}.png'
    plot_confusion_matrix(
        y_train, y_train_pred,
        f'{nombre_legible} - Entrenamiento ({len(y_train)} registros)',
        filename_train
    )
    print(f"   ✅ {filename_train}")
    
    # Matriz de confusión - VALIDACIÓN
    filename_val = f'cm_val_{nombre_archivo}.png'
    plot_confusion_matrix(
        y_val, y_val_pred,
        f'{nombre_legible} - Validación ({len(y_val)} registros)',
        filename_val
    )
    print(f"   ✅ {filename_val}")

print()

# ============================================================================
# 5. GENERAR CURVAS ROC COMPARATIVAS
# ============================================================================
print("="*80)
print("📊 GENERANDO CURVAS ROC")
print("="*80 + "\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Curva ROC - ENTRENAMIENTO
for i, (nombre_archivo, modelo) in enumerate(modelos.items()):
    nombre_legible = nombres_legibles[nombre_archivo]
    
    try:
        y_train_proba = modelo.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_train_proba)
        auc = roc_auc_score(y_train, y_train_proba)
        
        ax1.plot(fpr, tpr, color=colores[i], linewidth=2,
                label=f'{nombre_legible} (AUC = {auc:.3f})')
    except:
        pass

ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar (AUC = 0.500)')
ax1.set_xlabel('Tasa de Falsos Positivos', fontsize=12, fontweight='bold')
ax1.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12, fontweight='bold')
ax1.set_title('Curvas ROC - Entrenamiento', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Curva ROC - VALIDACIÓN
for i, (nombre_archivo, modelo) in enumerate(modelos.items()):
    nombre_legible = nombres_legibles[nombre_archivo]
    
    try:
        y_val_proba = modelo.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_val_proba)
        auc = roc_auc_score(y_val, y_val_proba)
        
        ax2.plot(fpr, tpr, color=colores[i], linewidth=2,
                label=f'{nombre_legible} (AUC = {auc:.3f})')
    except:
        pass

ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar (AUC = 0.500)')
ax2.set_xlabel('Tasa de Falsos Positivos', fontsize=12, fontweight='bold')
ax2.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12, fontweight='bold')
ax2.set_title('Curvas ROC - Validación', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_roc_comparacion.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ curvas_roc_comparacion.png\n")

# ============================================================================
# 6. FEATURE IMPORTANCE - MEJOR MODELO
# ============================================================================
print("="*80)
print("📊 GENERANDO FEATURE IMPORTANCE")
print("="*80 + "\n")

# Cargar mejor modelo (XGBoost)
with open('mejor_modelo.pkl', 'rb') as f:
    mejor_modelo = pickle.load(f)

if hasattr(mejor_modelo, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mejor_modelo.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Top 15 features
    top_features = importances.head(15)
    
    bars = ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Características Más Importantes (XGBoost)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Agregar valores en las barras
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['Importance'], i, f' {row["Importance"]:.4f}',
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ feature_importance.png\n")

# ============================================================================
# 7. COMPARACIÓN DE MÉTRICAS ENTRE MODELOS
# ============================================================================
print("="*80)
print("📊 GENERANDO COMPARACIÓN DE MÉTRICAS")
print("="*80 + "\n")

# Cargar resumen
df_resumen = pd.read_csv('resumen_modelos.csv')

# Crear gráfico de barras comparativo
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metricas = ['Val_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_F1']
titulos = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
    ax = axes[idx // 2, idx % 2]
    
    # Datos
    modelos_nombres = df_resumen['Modelo'].values
    valores = df_resumen[metrica].values
    
    # Colores (destacar el mejor)
    colors = ['#ff7f0e' if v == max(valores) else '#1f77b4' for v in valores]
    
    # Crear barras
    bars = ax.bar(range(len(modelos_nombres)), valores, color=colors, alpha=0.8, edgecolor='black')
    
    # Configuración
    ax.set_xticks(range(len(modelos_nombres)))
    ax.set_xticklabels(modelos_nombres, rotation=45, ha='right')
    ax.set_ylabel(titulo, fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación de {titulo} en Validación', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores encima de las barras
    for i, (bar, valor) in enumerate(zip(bars, valores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{valor:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Comparación de Métricas entre Modelos', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('comparacion_metricas.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ comparacion_metricas.png\n")

# ============================================================================
# 8. GRÁFICO DE OVERFITTING
# ============================================================================
print("="*80)
print("📊 GENERANDO ANÁLISIS DE OVERFITTING")
print("="*80 + "\n")

fig, ax = plt.subplots(figsize=(12, 6))

modelos_nombres = df_resumen['Modelo'].values
train_acc = df_resumen['Train_Accuracy'].values
val_acc = df_resumen['Val_Accuracy'].values

x = np.arange(len(modelos_nombres))
width = 0.35

bars1 = ax.bar(x - width/2, train_acc, width, label='Entrenamiento', color='#2ca02c', alpha=0.8)
bars2 = ax.bar(x + width/2, val_acc, width, label='Validación', color='#d62728', alpha=0.8)

# Líneas conectando train y val
for i in range(len(modelos_nombres)):
    ax.plot([i - width/2, i + width/2], [train_acc[i], val_acc[i]], 
           'k--', alpha=0.5, linewidth=1)
    
    # Mostrar diferencia
    diff = train_acc[i] - val_acc[i]
    mid_y = (train_acc[i] + val_acc[i]) / 2
    ax.text(i, mid_y, f'Δ {diff:.3f}', ha='center', va='center',
           fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Análisis de Overfitting: Entrenamiento vs Validación', 
            fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(modelos_nombres, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(0.7, 1.0)
ax.grid(axis='y', alpha=0.3)

# Línea de referencia de 10% de diferencia
ax.axhline(y=0.85, color='red', linestyle=':', alpha=0.5, linewidth=2)
ax.text(len(modelos_nombres)-0.5, 0.855, 'Umbral 10% diferencia', 
       fontsize=9, color='red', ha='right')

plt.tight_layout()
plt.savefig('analisis_overfitting.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ analisis_overfitting.png\n")

# ============================================================================
# 9. MATRIZ DE CONFUSIÓN CONSOLIDADA (4 MODELOS)
# ============================================================================
print("="*80)
print("📊 GENERANDO MATRIZ CONSOLIDADA")
print("="*80 + "\n")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, (nombre_archivo, modelo) in enumerate(modelos.items()):
    nombre_legible = nombres_legibles[nombre_archivo]
    
    # Predicciones
    y_train_pred = modelo.predict(X_train)
    y_val_pred = modelo.predict(X_val)
    
    # Matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    
    # Entrenamiento (fila superior)
    ax_train = axes[0, idx]
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Peligroso', 'Peligroso'],
               yticklabels=['No Peligroso', 'Peligroso'],
               cbar=False, ax=ax_train)
    ax_train.set_title(f'{nombre_legible}\nEntrenamiento', fontweight='bold')
    
    if idx == 0:
        ax_train.set_ylabel('Valor Real', fontweight='bold')
    
    # Validación (fila inferior)
    ax_val = axes[1, idx]
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges',
               xticklabels=['No Peligroso', 'Peligroso'],
               yticklabels=['No Peligroso', 'Peligroso'],
               cbar=False, ax=ax_val)
    ax_val.set_title('Validación', fontweight='bold')
    ax_val.set_xlabel('Predicción', fontweight='bold')
    
    if idx == 0:
        ax_val.set_ylabel('Valor Real', fontweight='bold')

plt.suptitle('Matrices de Confusión - Todos los Modelos', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('matrices_consolidadas.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ matrices_consolidadas.png\n")

# ============================================================================
# 10. RESUMEN FINAL
# ============================================================================
print("="*80)
print("✅ VISUALIZACIONES COMPLETADAS")
print("="*80 + "\n")

print("📁 ARCHIVOS PNG GENERADOS:\n")
print("MATRICES DE CONFUSIÓN INDIVIDUALES:")
print("  1. cm_train_random_forest.png")
print("  2. cm_val_random_forest.png")
print("  3. cm_train_gradient_boosting.png")
print("  4. cm_val_gradient_boosting.png")
print("  5. cm_train_xgboost.png")
print("  6. cm_val_xgboost.png")
print("  7. cm_train_lightgbm.png")
print("  8. cm_val_lightgbm.png")
print("\nGRÁFICOS COMPARATIVOS:")
print("  9. curvas_roc_comparacion.png")
print(" 10. comparacion_metricas.png")
print(" 11. analisis_overfitting.png")
print(" 12. matrices_consolidadas.png")
print("\nOTROS:")
print(" 13. feature_importance.png")

print("\n💡 TODOS los gráficos están en resolución 300 DPI")
print("   Listos para incluir en presentaciones y reportes\n")
print("="*80 + "\n")

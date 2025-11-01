import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, roc_curve, auc, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

ruta_imagenes = r"Proyecto_1\train"
ruta_csv = r"ISIC_2020_Training_GroundTruth_v2.csv"
output_dir = r"results"
os.makedirs(output_dir, exist_ok=True)
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# CARGA DEL CSV
df = pd.read_csv(ruta_csv)
df_simple = df[["image_name", "target"]].rename(columns={"target": "label"})
df_simple.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
print("Archivo metadata.csv creado con", len(df_simple), "registros.")

df = df_simple.copy()
df['image_path'] = df['image_name'].apply(lambda x: os.path.join(ruta_imagenes, x+ ".jpg"))
df = df[df['image_path'].apply(os.path.exists)]
print("Imágenes encontradas:", df.shape[0])

# DIVISION DEL ENTRENAMIENTO 70%, VALIDACIÓN 15% Y PRUEBA 15%
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# GENERADORES Y TAMAÑO DE IMAGENES
IMG_SIZE = (244, 244)
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.2,
                               horizontal_flip=True, vertical_flip=True)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(train_df, x_col='image_path', y_col='label',
                                           target_size=IMG_SIZE, class_mode='raw', batch_size=32)
val_data = val_test_gen.flow_from_dataframe(val_df, x_col='image_path', y_col='label',
                                            target_size=IMG_SIZE, class_mode='raw', batch_size=32)
test_data = val_test_gen.flow_from_dataframe(test_df, x_col='image_path', y_col='label',
                                             target_size=IMG_SIZE, class_mode='raw', batch_size=32, shuffle=False)

# MODELO EFFICIENTNET Y TRANSFER LEARNING
def crear_modelo(lr, optimizador, capas_descongeladas):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    if capas_descongeladas > 0:
        for layer in base.layers[-capas_descongeladas:]:
            layer.trainable = True

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)

    if optimizador == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# EXPERIMENTOS CON TASA DE APRENDIZAJE, OPTIMIZADORES, ETC
tasas_aprendizaje = [0.01, 0.001]
optimizadores = ["adam", "sgd"]
capas_descongeladas = [0, 10]
epochs = 6

experimentos_resultados = []

for lr in tasas_aprendizaje:
    for opt in optimizadores:
        for capas in capas_descongeladas:
            print(f"\nEntrenando EfficientNetB0 | lr={lr} | opt={opt} | capas={capas}")
            model = crear_modelo(lr, opt, capas)

            early = tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=2, restore_best_weights=True)
            hist = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early], verbose=1)

            preds_val = model.predict(val_data).ravel()
            y_val = val_data.labels

            auc_val = roc_auc_score(y_val, preds_val)
            f1_val = f1_score(y_val, preds_val > 0.5)
            recall_val = recall_score(y_val, preds_val > 0.5)
            precision_val = precision_score(y_val, preds_val > 0.5)
            cm = confusion_matrix(y_val, preds_val > 0.5)

            exp_result = {
                "lr": lr,
                "opt": opt,
                "capas_descongeladas": capas,
                "auc": auc_val,
                "f1": f1_val,
                "recall": recall_val,
                "precision": precision_val,
                "hist": hist.history,
                "cm": cm
            }
            experimentos_resultados.append(exp_result)
            
# RESULTADOS 
res_df = pd.DataFrame([{
    "lr": e["lr"], "opt": e["opt"], "capas_descongeladas": e["capas_descongeladas"],
    "auc": e["auc"], "f1": e["f1"], "recall": e["recall"], "precision": e["precision"]
} for e in experimentos_resultados])
res_df.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)

# GRAFICAS
for i, e in enumerate(experimentos_resultados):
    hist = e['hist']
    cm = e['cm']
    nombre_base = f"EfficientNet_lr{e['lr']}_opt{e['opt']}_capas{e['capas_descongeladas']}"

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist['accuracy'], label='train_acc')
    plt.plot(hist['val_accuracy'], label='val_acc')
    plt.title(f"Accuracy {nombre_base}")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist['auc'], label='train_auc')
    plt.plot(hist['val_auc'], label='val_auc')
    plt.title(f"AUC {nombre_base}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{nombre_base}_acc_auc.png"))
    plt.close()

#SELECCIÓN DEL MEJOR MODELO Y EVALUAR EN TEST
mejor_idx = res_df['auc'].idxmax()
mejor = res_df.loc[mejor_idx]
print("\nMejor combinación encontrada:\n", mejor)

modelo_final = crear_modelo(mejor["lr"], mejor["opt"], mejor["capas_descongeladas"])
modelo_final.fit(train_data, epochs=epochs, verbose=1)

preds_test = modelo_final.predict(test_data).ravel()
y_test = test_data.labels

print("\nResultados en Test:")
print(f"AUC: {roc_auc_score(y_test, preds_test):.4f}")
print(f"F1: {f1_score(y_test, preds_test > 0.5):.4f}")
print(f"Recall: {recall_score(y_test, preds_test > 0.5):.4f}")
print(f"Precision: {precision_score(y_test, preds_test > 0.5):.4f}")

# CURVA ROC FINAL
fpr, tpr, _ = roc_curve(y_test, preds_test)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Mejor modelo')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "roc_mejor_modelo.png"))
plt.close()

print("\nTodos los resultados guardados en:", plots_dir)

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
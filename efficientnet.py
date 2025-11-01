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
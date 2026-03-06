import gradio as gr
import pandas as pd
import numpy as np
import os
import shutil
import time 
import matplotlib.pyplot as plt
from PIL import Image
import PyPDF2

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- STEP 1 & 2: FEATURE EXTRACTION ---
def extract_features(files):
    features = []
    filenames = []
    file_types = []
    fixed_feature_length = 32

    for f in files:
        current_filename = os.path.basename(f.name)
        filenames.append(current_filename)
        file_ext = os.path.splitext(f.name)[1].lower()
        feat = np.zeros(fixed_feature_length)

        type_label = "Other"
        if file_ext in ['.png', '.jpg', '.jpeg']:
            type_label = "Image"
            try:
                img = Image.open(f.name).convert('L').resize((8, 4))
                feat = np.array(img).flatten() / 255.0
            except Exception as e:
                print(f"Feature extraction error (Image): {e}")
                feat = np.zeros(fixed_feature_length)

        elif file_ext in ['.csv', '.xlsx']:
            type_label = "Tabular"
            try:
                df = pd.read_csv(f.name) if file_ext == '.csv' else pd.read_excel(f.name)
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    raw_features = numeric_df.mean().values
                    feat[:min(len(raw_features), fixed_feature_length)] = raw_features[:fixed_feature_length]
            except Exception as e:
                print(f"Feature extraction error (Tabular): {e}")
                feat = np.zeros(fixed_feature_length)

        elif file_ext == '.pdf':
            type_label = "PDF"
            try:
                with open(f.name, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    feat[0] = len(reader.pages) / 100.0
                print(f"Feature extraction success (PDF): {current_filename}")
            except Exception as e:
                print(f"Feature extraction error (PDF): {e}")
                feat = np.zeros(fixed_feature_length)

        file_types.append(type_label)
        feat = np.nan_to_num(feat, nan=0.0)
        features.append(feat)

    return np.array(features), filenames, file_types




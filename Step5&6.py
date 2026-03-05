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

# --- STEP 3 & 4: CLUSTERING & EVALUATION ---
def cluster_and_evaluate(files, num_clusters, model_type):
    if not files or len(files) < 2:
        return pd.DataFrame(), "Needs at least 2 files", "N/A", None, None, None

    X_raw, filenames, file_types = extract_features(files)
    n_samples = len(X_raw)

    # Scaling Features 
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Auto-Adjust k Logic
    # Requirements: k must be < n_samples for valid silhouette/db metrics
    k = num_clusters
    adjustment_note = ""

    if k >= n_samples:
        k = n_samples - 1   # Auto-reduce k to satisfy k < n_samples
        adjustment_note = f" (Auto-adjusted from {num_clusters} to {k})"

    if k < 2: k = 2 # Ensure minimum 2 clusters

    # Variance Check
    if np.all(np.isclose(X, X[0, :])):
        return pd.DataFrame({"Filename": filenames, "Cluster": [0]*n_samples}), "Variance Error", "Variance Error", None, None, None

    # Clustering Model with fixed random_state for scientific practice
    if model_type == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=k)

    labels = model.fit_predict(X)

    # Metric Calculation 
    metric_info = f"Model: {model_type} | k={k}{adjustment_note} | "
    sil_score = ""
    db_score = ""
    
    # Needs at least 3 samples and k < samples for meaningful metrics
    if n_samples < 3:
        status_msg = "N/A (insufficient samples; needs >=3)"
        sil_score = f"{metric_info}{status_msg}"
        db_score = f"{metric_info}{status_msg}"
    elif k < n_samples and len(set(labels)) > 1:
        sil_score = f"{metric_info}Score: {silhouette_score(X, labels):.4f}"
        db_score = f"{metric_info}Index: {davies_bouldin_score(X, labels):.4f}"
    else:
        status_msg = "N/A (k must be < samples)"
        sil_score = f"{metric_info}{status_msg}"
        db_score = f"{metric_info}{status_msg}"

    # PCA Plot with reproducibility
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    fig_pca, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10")
    
    # Figure Labels and Grid for report correctness
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f"PCA Visualization ({model_type})")

    # Filename annotation (Truncated for long names)
    if n_samples <= 10:
        for i, txt in enumerate(filenames):
            short = (txt[:25] + "...") if len(txt) > 28 else txt
            ax.annotate(short, (X_2d[i, 0], X_2d[i, 1]), size=8, alpha=0.7)

    plt.colorbar(scatter)
    plt.tight_layout()

    # EDA Analysis Plots
    fig_eda1, ax_eda1 = plt.subplots(figsize=(5, 3))
    pd.Series(file_types).value_counts().plot(kind='bar', ax=ax_eda1)
    ax_eda1.set_title("EDA: Input Data Distribution")
    plt.tight_layout()

    fig_eda2, ax_eda2 = plt.subplots(figsize=(5, 3))
    pd.Series(labels).value_counts().sort_index().plot(kind='bar', ax=ax_eda2)
    ax_eda2.set_title("EDA: Cluster Size Distribution")
    ax_eda2.set_xlabel("Cluster ID")
    plt.tight_layout()

    df = pd.DataFrame({"Filename": filenames, "Type": file_types, "Cluster": labels})
    return df, sil_score, db_score, fig_pca, fig_eda1, fig_eda2

# --- STEP 5: FILE MANAGEMENT (INTEGRATED WITH WINDOWS FIX) ---
def save_results_and_organize(files, df):
    # Safety Check: Return None if dataframe is invalid or missing required columns
    if df is None or df.empty or 'Cluster' not in df.columns:
        return None, None

    # UNIQUE NAMING SOLUTION: Create unique folder names using Timestamp
    # This prevents PermissionError caused by locked folders on Windows
    timestamp = int(time.time())
    base_folder = f"smart_sort_output_{timestamp}"
    csv_path = os.path.join(base_folder, f"report_{timestamp}.csv")
    zip_output_name = f"sorted_files_{timestamp}"

    # PERMISSION SOLUTION: Attempt to clean up old 'smart_sort_output' if it exists
    if os.path.exists("smart_sort_output"):
        try:
            # Brief delay to allow Windows OS to release potential file locks
            time.sleep(0.5) 
            shutil.rmtree("smart_sort_output")
        except PermissionError:
            # Catch error if folder is locked by File Explorer or another application
            print("Warning: Folder 'smart_sort_output' is locked, skipping deletion.")

    # Create the new base directory for the current run
    os.makedirs(base_folder, exist_ok=True)

    # Save CSV Report
    df.to_csv(csv_path, index=False)

    # Organize original files into cluster sub-folders
    for file_obj in files:
        fname = os.path.basename(file_obj.name)
        row = df[df['Filename'] == fname]
        if not row.empty:
            cluster_id = row.iloc[0]['Cluster']
            cluster_dir = os.path.join(base_folder, f"Cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
            shutil.copy2(file_obj.name, os.path.join(cluster_dir, fname))

    # Archive the organized folder into a ZIP file for download
    zip_path = shutil.make_archive(zip_output_name, 'zip', base_folder)
    return csv_path, zip_path

# --- STEP 6: MAIN RUNNER ---
def run_full_app(files, num_clusters, model_type):
    df, sil, db, fig_pca, fig_eda1, fig_eda2 = cluster_and_evaluate(files, num_clusters, model_type)
    csv_p, zip_p = save_results_and_organize(files, df)
    return df, sil, db, fig_pca, fig_eda1, fig_eda2, csv_p, zip_p

# --- UI LAYOUT ---
with gr.Blocks(title="SMARTSORT Advanced") as app:
    gr.Markdown("# SMARTSORT: AI Organizer & Data Analysis")

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(file_count="multiple", label="1. Upload Files")
            model_selector = gr.Dropdown(choices=["K-Means", "Agglomerative"], value="K-Means", label="2. Model")
            cluster_slider = gr.Slider(minimum=2, maximum=10, value=3, step=1, label="3. Clusters")
            submit_btn = gr.Button("Analyze & Sort", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Cluster Plot"):
                    plot_output = gr.Plot(label="PCA")
                with gr.TabItem("EDA Analysis"):
                    eda_output1 = gr.Plot(label="Type Distribution")
                    eda_output2 = gr.Plot(label="Cluster Distribution")

    with gr.Row():
        sil_metric = gr.Textbox(label="Silhouette Analysis", interactive=False)
        db_metric = gr.Textbox(label="Davies-Bouldin Analysis", interactive=False)

    with gr.Row():
        result_table = gr.Dataframe(label="Final Sorting Report")
        with gr.Column():
            dl_csv = gr.File(label="CSV Report")
            dl_zip = gr.File(label="Organized Zip")

    submit_btn.click(
        fn=run_full_app,
        inputs=[file_uploader, cluster_slider, model_selector],
        outputs=[result_table, sil_metric, db_metric, plot_output, eda_output1, eda_output2, dl_csv, dl_zip]
    )

app.launch()
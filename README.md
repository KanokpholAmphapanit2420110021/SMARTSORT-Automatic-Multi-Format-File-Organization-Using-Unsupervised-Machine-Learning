# SMARTSORT: AI-Powered File Organizer

SMARTSORT is an automated data sorting application that utilizes Machine Learning clustering algorithms to organize diverse file formats — including **Images, CSV, Excel, and PDF** — into structured folders based on their extracted features.

The system demonstrates how **unsupervised learning techniques** can automatically group heterogeneous files without predefined labels.

---

# Project Overview

SMARTSORT is designed as a practical **AI-powered file organization system**.

It accepts multiple file types from users, extracts numerical features, applies clustering algorithms, evaluates clustering quality, and organizes similar files into grouped folders automatically.

---

# AI Workflow

SMARTSORT processes uploaded files using the following Machine Learning pipeline.

## 1. Feature Extraction

- **Images** → grayscale statistics and pixel features  
- **CSV / Excel** → numeric column statistics  
- **PDF** → page count analysis  

## 2. Feature Scaling

Features are normalized using **StandardScaler**.

## 3. Dimensionality Reduction

For visualization, the system applies **PCA (Principal Component Analysis)**.

## 4. Clustering Algorithms

Two clustering models are supported:

- **K-Means**
- **Agglomerative Clustering**

## 5. Cluster Evaluation

Clustering performance is evaluated using:

- **Silhouette Score**
- **Davies-Bouldin Index**

## 6. File Organization

After clustering:

- Files are grouped into **cluster folders automatically**
- Results are exported as a **ZIP archive**

---

# Requirements

To run this application, ensure **Python** is installed along with the following libraries.

## Core Libraries

### Gradio
Used to build the **interactive web interface**.

### Pandas & NumPy
Used for **data processing and numerical computation**.

### Scikit-Learn
Used for:
- K-Means clustering
- Agglomerative clustering
- Feature scaling
- PCA visualization
- Evaluation metrics

### Matplotlib
Used to generate **EDA and clustering plots**.

### Pillow (PIL)
Used for **image preprocessing and feature extraction**.

### PyPDF2
Used for **PDF metadata and page count analysis**.

## Install Dependencies

Install required packages using:


pip install -r requirements.txt


---

# How to Run

Follow these steps to launch the SMARTSORT application.

## 1. Open Terminal

Navigate to the project directory:


cd SMARTSORT
cd src


## 2. Run the Application

Execute the main application file:


python Step5&6.py


## 3. Access the Web Interface

Open your browser and go to:


http://127.0.0.1:7860


## 4. Upload Files

Use the **Upload Files** box to select multiple files such as:

- Images
- CSV files
- Excel files
- PDF documents

## 5. Configure Settings

Select:

- AI Model (**K-Means** or **Agglomerative**)
- Number of clusters

## 6. Analyze and Download

Click **Run AI Sort** to:

- visualize clustering results
- organize files automatically
- download the sorted ZIP file

---

# Dataset Location

This project **does not require a fixed dataset**.

Users can upload their own files directly through the interface.

## Supported File Formats

### Images


.jpg
.jpeg
.png


Processed using **grayscale normalization and pixel statistics**.

### Tabular Data


.csv
.xlsx


Processed using **numeric column statistics**.

### Documents


.pdf


Processed using **page count analysis**.

## Example Dataset

Users may test the system with:

- academic files
- personal documents
- mixed file collections

to observe how the AI groups files automatically.

---

# Results

SMARTSORT generates several outputs to evaluate clustering performance.

## Output Visualizations

- Cluster Size Distribution
- Silhouette Score Analysis
- Davies-Bouldin Index Analysis
- PCA Cluster Visualization

Example result files:


Cluster Size Distribution.png
Davies-Bouldin Analysis.png
Silhouette Score Analysis.png
Visualization K-Means.png
Visualization Agglomerative.png


All results are stored in:


results/


---

# Project Structure


SMARTSORT
│
├── data/ # Input datasets
├── results/ # Generated clustering outputs
├── src/ # Source code
│ ├── Step1&2.py # Feature extraction
│ ├── Step3&4.py # Clustering and evaluation
│ └── Step5&6.py # Full application (UI + sorting)
│
├── README.md
└── requirements.txt


---

# Key Features

- AI-powered automatic file organization
- Support for multiple file formats
- Multiple clustering algorithms
- Visualization of clustering results
- Interactive interface using **Gradio**
- Automatic generation of sorted folders

---

# Technologies Used

- Python
- Scikit-learn
- Gradio
- Pandas
- NumPy
- Matplotlib
- Pillow
- PyPDF2

---

# Project Purpose

This project was developed as part of a **Machine Learning course project**.

SMARTSORT demonstrates how **unsupervised learning techniques** can automatically categorize heterogeneous files.

The system integrates:

- feature extraction
- machine learning clustering
- evaluation metrics
- interactive user interface

to build a practical **AI-powered file organization system**.

---

# Notes

- This project uses **unsupervised learning**, so no predefined labels are required.
- Clustering quality depends on the **quality of extracted features**.
- Users can experiment with **different clustering models and cluster numbers** through the UI.

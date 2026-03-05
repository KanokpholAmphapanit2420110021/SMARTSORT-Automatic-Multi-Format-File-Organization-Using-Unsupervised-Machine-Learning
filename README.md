SMARTSORT: AI-Powered File Organizer
SMARTSORT is an automated data sorting application that utilizes Machine Learning clustering algorithms to organize diverse file formats—including Images, CSV, Excel, and PDF—into structured folders based on their extracted features.

🛠 Requirements
To run this application, ensure you have Python installed along with the following libraries:

Gradio: For the web-based user interface.

Pandas & Numpy: For data manipulation and feature processing.

Scikit-Learn: For clustering models (K-Means, Agglomerative), scaling, and PCA visualization.

Matplotlib: For generating EDA and cluster plots.

Pillow (PIL): For image preprocessing and feature extraction.

PyPDF2: For PDF metadata and page analysis.

You can install all dependencies using the provided requirements.txt:

Bash
pip install -r requirements.txt
🚀 How to Run
Follow these steps to launch the SMARTSORT application:

Open Terminal: Navigate to the project directory SMARTSORT/.

Execute the Script: Run the main application file:

Bash
python "Step5&6.py"
Access the UI: Open your web browser and go to the local URL provided (default is http://127.0.0.1:7860).

Upload Files: Use the "Upload Files" box to select multiple Images, CSV, or PDF files.

Configure Settings: Select your preferred AI Model (K-Means or Agglomerative) and set the desired number of clusters.

Analyze & Download: Click "Run AI Sort" to visualize the clusters and download the organized ZIP file containing your sorted folders.

📂 Dataset Information
Source: This project is designed for user-provided datasets. It does not require a pre-installed database.

Supported Formats:

Images: .jpg, .jpeg, .png (Processed via grayscale normalization).

Tabular: .csv, .xlsx (Processed via numeric mean calculation).

Documents: .pdf (Processed via page count analysis).

Sample Data: Users can test the system using any collection of miscellaneous academic or personal files to see real-time grouping.

📊 Project Structure
Step1&2.py: Initial feature extraction logic.

Step3&4.py: Clustering implementation and evaluation metrics.

Step5&6.py: Final integrated application with file organization and UI.

requirements.txt: List of necessary Python packages.

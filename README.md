# 🌱 Soil Organic Carbon Prediction System (SOC-AI)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange.svg)](https://scikit-learn.org/)

A high-performance **Agricultural AI** application designed to predict **Soil Organic Carbon (SOC)** using **VNIR Spectroscopy** (350nm—2500nm). This system transitions traditional 7-day laboratory diagnostics into a near-instantaneous digital reporting tool.

---

## 🚀 Key Innovation
Traditional SOC testing is expensive and slow. This project leverages **Machine Learning** to identify the "Light Signature" of soil samples, providing real-time health data for farmers, researchers, and climate scientists.

### 🧠 The "Brain" (Model Architecture)
*   **Dimensionality Reduction (PCA):** Compresses 2,151 high-dimensional spectral wavelengths into 50 principal components to eliminate noise and prevent overfitting.
*   **ElasticNet Regression:** A hybrid model combining **Lasso (Feature Selection)** and **Ridge (Multicollinearity management)** to ensure stable predictions across variable soil types.
*   **3-Step Hybrid Calibration:** A custom post-processing pipeline that ensures physically realistic SOC outputs (0.0%—1.0%) and maintains visible trends even in low-signal datasets.

---

## 🛠️ Technical Features
-   **Universal Resampling Engine:** Automatically standardizes input data from any spectrometer (detecting 1nm/2nm intervals or generic headers).
-   **Automated Data Cleaning:** Built-in whitespace stripping, NaN handling, and metadata filtering for robust "fail-safe" operation.
-   **Scientific Visualization:** Real-time plotting of spectral signatures and Predicted vs. Actual performance charts.
-   **Reporting:** Professional high-precision CSV export (4 decimal places) for research documentation.

---
**Live Link:** https://jalynatmucetgtplmgqf22.streamlit.app/
---

## 📖 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`.

### Installation
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Swatantraraj19/Soil-Analysis-and-Prediction.git
    cd Soil-Analysis-and-Prediction
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
```bash
streamlit run app.py
```

---

## 📄 Supported File Format
-   **Type:** `.csv` or `.xlsx`
-   **Requirement:** The file must contain numeric spectral columns (Wavelengths from 350 to 2500).
-   **Note:** The app automatically ignores non-numeric metadata (IDs, pH, GPS) to ensure clean inference.

---

## 📦 Project Structure
-   `app.py`: The Main Production UI.
-   `model.pkl`: Pre-trained ElasticNet Model.
-   `pca.pkl`: PCA Dimensionality reduction artifacts.
-   `scaler.pkl`: StandardScaler for feature normalization.
-   `wavelengths.pkl`: Master list of target wavelengths (350-2500nm).

---
Built by Swatantra Raj Kumar Singh for the future of **Sustainable Agriculture** and **Carbon Sequestration.** 🌍🏜️🌱

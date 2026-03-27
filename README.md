# 🌍 Soil Organic Carbon Prediction & Analysis

A comprehensive system for predicting **Soil Organic Carbon (SOC)** using VNIR spectroscopy and Machine Learning. This repository contains both the **Interactive Web Application** and the **original Research Notebooks**.

---

## 🚀 1. Interactive Soil Analyzer (Web App)
Predict SOC in real-time by uploading spectral data (350nm - 2500nm).

### ✨ Features
- **Hindi & English Support:** Fully localized for regional use.
- **Spectral Prediction:** Powered by a pre-trained **Elastic Net** model.
- **Smart Filtering:** Filter results by carbon threshold.
- **Visual Trends:** Integrated charts for SOC trends and spectral signatures.
- **Exportable:** Download predictions as CSV.

### 🛠️ Getting Started
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the App:**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 2. Research & Model Training (Colab)
This project originated from a study using Visible Near Infrared (VNIR) spectroscopy to estimate SOC content.

### 📉 Algorithms Studied
- **Elastic Net Regression** (Best Performing Model)
- **Support Vector Regression (SVR)**
- **Polynomial Regression**

### 📁 Research Files
- `Project_ElasticNet.ipynb`: Primary training notebook.
- `Project_Polynomial.ipynb`: Polynomial experimentation.
- `Project_SVR.ipynb`: SVR experimentation.

### 📊 Performance Summary
Elastic Net Regression outperformed other models by effectively balancing feature selection and regularization, achieving the lowest RMSE and the highest R² Score.

---

## 📦 Project Structure
- `app.py`: Main interactive interface.
- `model.pkl`: Pre-trained Elastic Net model used by the app.
- `scaler.pkl` & `features.pkl`: Preprocessing artifacts.
- `requirements.txt`: Python dependencies.
- `*.ipynb`: Original research and training notebooks.

---
Built with ❤️ for Soil Science.

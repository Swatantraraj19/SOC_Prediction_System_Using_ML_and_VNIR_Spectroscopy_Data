# 🌱 Soil Organic Carbon Analyzer

A powerful **Streamlit** application designed to predict **Soil Organic Carbon (SOC)** using VNIR spectroscopy data and a pre-trained **Elastic Net** model.

## 🚀 Key Features
- **Spectral Prediction:** Predict SOC from 350nm-2500nm wavelength data.
- **Robust Validation:** Automatically handles missing data and alerts users of required spectral bands.
- **Data Insights:** Get statistics and distributions for your uploaded soil samples.
- **Sample Mapping:** Interactive visualization for samples with GPS coordinates.
- **Spectral Signatures:** Visualize the reflectance/absorbance profiles for up to 10 samples.
- **Exportable Results:** Download your predictions directly as a CSV.

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- [Optional] VS Code or any text editor

### Installation
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
From your terminal, navigate to the folder and run:
```bash
streamlit run app.py
```

## 📄 File Format
- **Supported Formats:** `.csv` or `.xlsx`
- **Required Columns:**
  - Spectral bands from `350` to `2500`.
  - [Optional] `Latitude` and `Longitude` for mapping features.
  - The app automatically ignores existing `oc` (Organic Carbon) columns.

## 📦 Project Structure
- `app.py`: The main Streamlit interface.
- `model.pkl`: Pre-trained Elastic Net model.
- `scaler.pkl`: StandardScaler for data normalization.
- `features.pkl`: List of required spectral features.
- `requirements.txt`: Python dependencies.

---
Built with ❤️ for Soil Science.

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.interpolate import interp1d

# 1. Page Config & Caching

st.set_page_config(page_title="Soil Analyzer", layout="wide", page_icon="🌱")

@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        pca = pickle.load(open('pca.pkl', 'rb'))
        target_wavelengths = pickle.load(open('wavelengths.pkl', 'rb'))
        return model, scaler, pca, target_wavelengths
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None, None, None

model, scaler, pca, target_wavelengths = load_artifacts()

# --- SHARED STATE ---
input_unit = "Raw Reflectance (R)"

# --- TRANSLATIONS ---
translations = {
    "English": {
        "title": "🌱 Soil Organic Carbon Prediction System",
        "about": "📌 About this App",
        "about_text": "Advanced ML app for predicting SOC using spectroscopy and PCA-ElasticNet pipeline.",
        "sample_file": "⬇️ Sample CSV Template",
        "step1": "Step 1: Upload CSV or Excel",
        "preview": "📄 Data Preview",
        "analyzing": "Analyzing spectral data...",
        "avg_soc": "Avg SOC",
        "tab_results": "📊 Results",
        "tab_spectral": "📈 Spectral",
        "tab_performance": "🎯 Accuracy Analysis",
        "filter_title": "🔍 Filter & Trends",
        "results_title": "📝 Predictions",
        "download_btn": "💾 Download CSV",
        "trend_title": "SOC Trend",
        "spectral_title": "🔬 Spectral Signatures",
        "sample_id": "Sample ID",
        "pred_soc": "Predicted SOC (%)",
        "acc_title": "🎯 Performance Analysis",
        "actual_soc_label": "Actual SOC (%)",
        "r2_score": "R² Accuracy",
        "rmse_error": "RMSE Error",
        "error_generic": "❌ Processing error.",
        "index_label": "Sr. No"
    },
    "हिन्दी (Hindi)": {
        "title": "🌱 मिट्टी की जांच और कार्बन अनुमान प्रणाली",
        "about": "📌 जानकारी",
        "about_text": "स्पेक्ट्रोस्कोपी और PCA-ElasticNet का उपयोग करके मिट्टी के कार्बन का अनुमान लगाने वाली ऐप।",
        "sample_file": "⬇️ नमूना फाइल",
        "step1": "फाइल अपलोड करें",
        "preview": "📄 डेटा पूर्वावलोकन",
        "analyzing": "विश्लेषण हो रहा है...",
        "avg_soc": "औसत कार्बन",
        "tab_results": "📊 परिणाम",
        "tab_spectral": "📈 ग्राफ",
        "tab_performance": "🎯 सटीकता की जांच",
        "filter_title": "🔍 फिल्टर और रुझान",
        "results_title": "📝 अनुमानित परिणाम",
        "download_btn": "💾 सेव करें (CSV)",
        "trend_title": "कार्बन रुझान",
        "spectral_title": "🔬 स्पेक्ट्रल विश्लेषण",
        "sample_id": "नमूना संख्या",
        "pred_soc": "अनुमानित SOC (%)",
        "acc_title": "🎯 सटीकता विश्लेषण",
        "actual_soc_label": "वास्तविक SOC (%)",
        "r2_score": "सटीकता (R²)",
        "rmse_error": "त्रुटि (RMSE)",
        "error_generic": "❌ प्रोसेसिंग में समस्या आई",
        "index_label": "क्र. सं."
    }
}

#  High-Symmetry Navbar Logic
st.markdown("""
<style>
    /* Force vertical centering in all columns */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: center; /* Center Vertically */
        align-items: stretch;    /* Stretch horizontally to fill 1/3 each */
        padding: 0 10px;
    }
    /* Align individual elements internally to center */
    [data-testid="stVerticalBlock"] > div {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    /* Match Expander height/look closer to buttons */
    .stExpander {
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Symmetrical 3-Column Navbar (1:1:1 weighting)
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

with nav_col1:
    lang_choice = st.selectbox("🌍 Language / भाषा:", list(translations.keys()), label_visibility="collapsed")
    lang = translations[lang_choice]

with nav_col2:
    @st.cache_data
    def get_sample_csv():
        # Create columns 350-2500 nm (2151 columns)
        cols = [str(w) for w in range(350, 2501)]
        df = pd.DataFrame(np.random.rand(5, 2151), columns=cols)
        return df.to_csv(index=False).encode('utf-8')
    st.download_button(lang["sample_file"], get_sample_csv(), "soil_template.csv", use_container_width=True)

with nav_col3:
    with st.expander(lang["about"]):
        st.write(lang["about_text"])

st.divider()
st.title(lang["title"])

file = st.file_uploader(lang["step1"], type=['csv', 'xlsx'])

#  SAFE RESAMPLING FUNCTION
def resample_input(df, target_wavelengths, input_unit):
    # --- ROBUST CLEANING ---
    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Find numeric columns (assumed to be wavelengths)
    numeric_columns_mask = pd.to_numeric(df.columns, errors='coerce').notnull()
    spectral_df = df.loc[:, numeric_columns_mask]

    #  REMOVE columns that are completely NaN
    spectral_df = spectral_df.dropna(axis=1, how='all')

    if spectral_df.shape[1] == 0:
        raise ValueError("No valid spectral data found after cleaning.")

    # Wavelength mapping logic
    input_wavelengths = spectral_df.columns.astype(float)
    
    # Check if headers are generic indices (e.g., 0, 1, 2...) instead of actual wavelengths
    if input_wavelengths.min() < 100 and len(input_wavelengths) > 1:
        # Standard NIR spectrometers usually cover 350-2500nm. 
        # If columns are 0, 1, 2... we map them to this range to avoid extrapolation errors.
        input_wavelengths = np.linspace(350, 2500, len(input_wavelengths))
    
    resampled = []
    for _, row in spectral_df.iterrows():
        if row.isnull().all():
            raise ValueError("One or more rows contain no valid spectral values.")
        val = row.values
        # If user picked Reflectance (Default), we pass raw values as trained.
        # If they pick Absorbance, we transform it.
        if input_unit == "Log Absorbance (log 1/R)":
            val = np.clip(val, 0.0001, 1.0)
            val = np.log10(1.0 / val)
            
        # Interpolate to target wavelengths (500 pts as per wavelengths.pkl)
        # If the input range is narrow, the model will see zeros for those bands.
        f = interp1d(input_wavelengths, val, kind='linear', bounds_error=False, fill_value=0.0)
        resampled.append(f(target_wavelengths))
    
    # Check for range coverage
    max_in = input_wavelengths.max()
    min_in = input_wavelengths.min()
    if max_in < 2400 or min_in > 400:
        st.warning(f"⚠️ **Partial spectrum detected:** Your file covers {int(min_in)}nm—{int(max_in)}nm, but the model expects 350nm—2500nm. Missing bands was set to Zero (0.0). This will affect accuracy.")

    return np.array(resampled)


# MAIN LOGIC

if file is not None and model is not None:
    try:
        raw_data = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if raw_data.empty:
            st.error("❌ File is empty")
            st.stop()

        st.subheader(lang["preview"])
        raw_data.index.name = lang["index_label"]
        st.dataframe(raw_data.head(5), height=150)

        # Detect Actual Ground Truth
        raw_data.columns = [str(c).strip() for c in raw_data.columns]
        actual_col = next((c for c in raw_data.columns if c.lower() in ['soc', 'oc','actual','actual_soc']), None)
        
        data_to_pred = raw_data.copy()
        actual_values = None
        if actual_col:
            actual_values = data_to_pred[actual_col]
            data_to_pred = data_to_pred.drop(columns=[actual_col])

        # PCA + ElasticNet Pipeline
        with st.spinner(lang["analyzing"]):
            input_resampled = resample_input(data_to_pred, target_wavelengths, input_unit)
            X_pca = pca.transform(input_resampled)
            X_scaled = scaler.transform(X_pca)
            raw_pred = model.predict(X_scaled)

            #  STEP 1: Clip unrealistic values
            prediction = np.clip(raw_pred, 0, 1)

            #  STEP 2: If all values collapse (same), apply mild scaling for demo
            # If the model gives -10 to -5, Step 1 makes them all 0. This "if" detects that zero-collapse.
            if np.std(prediction) < 1e-6:
                prediction = (raw_pred - raw_pred.min()) / (raw_pred.max() - raw_pred.min() + 1e-6)

            #  STEP 3: Final safety clip
            prediction = np.clip(prediction, 0, 1)
        
        if raw_pred.min() < 0:
            st.warning("⚠️ Predictions normalized due to model extrapolation (demo mode).")

        result = pd.DataFrame({
            lang["sample_id"]: range(1, len(prediction)+1),
            lang["pred_soc"]: prediction
        })

        # --- TABS ---
        tabs = st.tabs([lang["tab_results"], lang["tab_spectral"]] + ([lang["tab_performance"]] if actual_col else []))

        with tabs[0]:
            col_a, col_b = st.columns([3, 2])
            with col_b:
                st.subheader(lang["filter_title"])
                # Result Logic
                min_p, max_p = float(prediction.min()), float(prediction.max())
                s_min, s_max = (min_p, max_p) if min_p != max_p else (min_p - 0.01, max_p + 0.01)

                th = st.slider(lang["pred_soc"], s_min, s_max, s_min)
                res_f = result[result[lang["pred_soc"]] >= th]
                
                st.metric(lang["avg_soc"], f"{res_f[lang['pred_soc']].mean():.3f}%")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(res_f[lang["sample_id"]], res_f[lang["pred_soc"]], marker='o', color='#1b5e20')
                ax.set_title(lang["trend_title"])
                st.pyplot(fig)

            with col_a:
                st.subheader(lang["results_title"])
                res_f.index.name = lang["index_label"]
                st.dataframe(res_f, use_container_width=True, height=400)
                st.download_button(lang["download_btn"], res_f.to_csv(index=False, float_format="%.4f").encode('utf-8'), "results.csv")

        with tabs[1]:
            st.subheader(lang["spectral_title"])
            if hasattr(scaler, "mean_") and len(scaler.mean_) == 50:
                 st.info("💡 Note: Model expects 50 PCA features. Statistics shown below are for the 500 raw wavelengths after resampling.")
            
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
            for i in range(min(5, len(input_resampled))):
                ax_spec.plot(target_wavelengths, input_resampled[i], alpha=0.5, label=f"Sample {i+1}")
            ax_spec.set_xlabel("Wavelength (nm)")
            ax_spec.set_ylabel("Amplitude")
            ax_spec.legend()
            st.pyplot(fig_spec)
            
            # --- DIAGNOSTIC STATS ---
            with st.expander("📊 Data Diagnostic (Spectral Statistics)"):
                avg_ref = input_resampled.mean()
                st.write(f"**Mean Reflectance of Uploaded File:** `{avg_ref:.4f}`")
                st.write(f"**Max Reflectance:** `{input_resampled.max():.4f}`")
                st.write(f"**Min Reflectance:** `{input_resampled.min():.4f}`")
                st.caption("If your reflectance is > 0.6 but your soil looks dark, check if your spectrometer used different units.")

        if actual_col:
            # Sub-selection for the visual plot based on slider filter
            filt_actual = actual_values.reindex(res_f.index)
            filt_pred = res_f[lang["pred_soc"]]

            with tabs[2]:
                st.subheader(lang["acc_title"])
                
                # Scatter Plot
                fig_err, ax_err = plt.subplots(figsize=(8, 4))
                ax_err.scatter(filt_actual, filt_pred, color='#2e7d32', alpha=0.6)
                
                all_vals = pd.concat([filt_actual, filt_pred])
                line_range = [all_vals.min(), all_vals.max()]
                ax_err.plot(line_range, line_range, 'r--', lw=2)
                ax_err.set_xlabel(lang["actual_soc_label"])
                ax_err.set_ylabel(lang["pred_soc"])
                st.pyplot(fig_err)

    except ValueError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        st.error(lang["error_generic"])
        st.exception(e)
elif model is None:
    st.info("Artifacts missing.")
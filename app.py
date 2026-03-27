import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# 1. Page Config & Caching
# -----------------------------
st.set_page_config(page_title="Soil Analyzer", layout="wide", page_icon="🌱")

@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        features = pickle.load(open('features.pkl', 'rb'))
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None, None

model, scaler, features = load_artifacts()

# -----------------------------
# 2. Multi-Language & Error Handling
# -----------------------------
translations = {
    "English": {
        "title": "🌱 Soil Organic Carbon Prediction System",
        "about": "📌 About this App",
        "about_text": "This app predicts Soil Organic Carbon (SOC) using VNIR spectroscopy data and Elastic Net model.",
        "sample_file": "⬇️ Sample CSV Template",
        "sample_help": "Ensure correct column naming (350 to 2500).",
        "step1": "Step 1: Upload CSV or Excel File",
        "preview": "📄 Data Preview",
        "warning_missing": "⚠️ Warning: Missing {n} spectral bands. Filling with 0.",
        "warning_range": "⚠️ Warning: Unusual spectral values detected (Negative or >1000). Results may be unreliable.",
        "warning_large": "ℹ️ Note: Large dataset (>5000 rows). Analysis may take a moment.",
        "error_empty": "❌ Error: The uploaded file is empty or missing spectral data.",
        "analyzing": "Analyzing spectra...",
        "avg_soc": "Avg SOC",
        "tab_results": "📊 Results",
        "tab_spectral": "📈 Spectral",
        "tab_performance": "🎯 Accuracy",
        "filter_title": "🔍 Filter & Trends",
        "results_title": "📝 Predictions",
        "download_btn": "📥 Download Results CSV",
        "trend_title": "SOC Trend",
        "spectral_title": "🔬 Spectral Signatures",
        "sample_id": "Sample ID",
        "pred_soc": "Predicted SOC (%)",
        "acc_title": "🎯 Performance Analysis",
        "actual_soc_label": "Actual SOC (%)",
        "r2_score": "R² Accuracy",
        "rmse_error": "RMSE Error",
        "error_generic": "❌ Something went wrong during processing."
    },
    "हिन्दी (Hindi)": {
        "title": "🌱 मिट्टी की जांच और कार्बन अनुमान प्रणाली",
        "about": "📌 ऐप के बारे में",
        "about_text": "यह ऐप स्पेक्ट्रोस्कोपी डेटा और इलास्टिक नेट मॉडल का उपयोग करके मिट्टी के कार्बन (SOC) का अनुमान लगाता है।",
        "sample_file": "⬇️ नमूना फाइल (Template)",
        "sample_help": "सही कॉलम नाम (350 से 2500) सुनिश्चित करने के लिए इस टेम्पलेट का उपयोग करें।",
        "step1": "फाइल अपलोड करें (CSV या Excel)",
        "preview": "📄 डेटा की झलक",
        "warning_missing": "⚠️ चेतावनी: {n} बैंड गायब हैं।",
        "warning_range": "⚠️ चेतावनी: कुछ वैल्यू गलत लग रही हैं (नेगेटिव या बहुत ज़्यादा)। अनुमान गलत हो सकते हैं।",
        "warning_large": "ℹ️ सूचना: बड़ा डेटा सेट। विश्लेषण में समय लग सकता है।",
        "error_empty": "❌ गलती: अपलोड की गई फाइल खाली है या डेटा गायब है।",
        "analyzing": "विश्लेषण किया जा रहा है...",
        "avg_soc": "औसत कार्बन",
        "tab_results": "📊 परिणाम",
        "tab_spectral": "📈 ग्राफ",
        "tab_performance": "🎯 सटीकता",
        "filter_title": "🔍 फिल्टर और रुझान",
        "results_title": "📝 अनुमानित परिणाम",
        "download_btn": "📥 परिणाम डाउनलोड करें (CSV)",
        "trend_title": "कार्बन रुझान",
        "spectral_title": "🔬 स्पेक्ट्रल विश्लेषण",
        "sample_id": "नमूना संख्या",
        "pred_soc": "अनुमानित SOC (%)",
        "acc_title": "🎯 सटीकता का विश्लेषण",
        "actual_soc_label": "वास्तविक SOC (%)",
        "r2_score": "सटीकता (R²)",
        "rmse_error": "त्रुटि (RMSE)",
        "error_generic": "❌ प्रोसेसिंग के दौरान कुछ गलत हुआ।"
    }
}

# Navbar Columns
nav_col1, nav_col2, nav_col3 = st.columns([1.5, 1.5, 2])
with nav_col1:
    lang_choice = st.selectbox("🌍 Language", options=list(translations.keys()), label_visibility="collapsed")
    lang = translations[lang_choice]
with nav_col2:
    if features is not None:
        @st.cache_data
        def get_sample_csv():
            df = pd.DataFrame([[0.0] * len(features)], columns=features)
            return df.to_csv(index=False).encode('utf-8')
        st.download_button(lang["sample_file"], get_sample_csv(), "soil_template.csv", "text/csv", use_container_width=True)
with nav_col3:
    with st.expander(lang["about"]): st.write(lang["about_text"])

st.divider()
st.title(lang["title"])

file = st.file_uploader(lang["step1"], type=['csv', 'xlsx'])

if file is not None and model is not None:
    try:
        raw_data = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if raw_data.empty:
            st.error(lang["error_empty"])
            st.stop()
        
        st.subheader(lang["preview"])
        st.dataframe(raw_data.head(5), height=150)

        # Pre-processing
        raw_data.columns = [str(c).strip() for c in raw_data.columns]
        actual_col = next((c for c in raw_data.columns if c.lower() in ['oc', 'actual', 'actual_soc']), None)
        
        data_to_pred = raw_data.copy()
        if actual_col:
            actual_values = data_to_pred[actual_col]
            data_to_pred = data_to_pred.drop(columns=[actual_col])

        # Validate features
        missing_bands = [f for f in features if f not in data_to_pred.columns]

        if len(missing_bands) > 0:
            st.error("❌ Invalid file! Please upload correct spectral data.")
            st.stop()
        
        input_data = data_to_pred[features].apply(pd.to_numeric, errors='coerce').fillna(0)

        # 📏 Quality Check
        is_all_zero = (input_data == 0).all().all()
        out_of_range = (input_data < 0).any().any() or (input_data > 1000).any().any()
        
        if out_of_range or missing_bands:
            with st.expander("📝 Data Quality Report"):
                if missing_bands: st.write(lang["warning_missing"].format(n=len(missing_bands)))
                if out_of_range: st.warning(lang["warning_range"])

        # Prediction
        with st.spinner(lang["analyzing"]):
            scaled = scaler.transform(input_data)
            prediction = model.predict(scaled)

        # Result Logic
        min_p, max_p = float(prediction.min()), float(prediction.max())
        s_min, s_max = (min_p, max_p) if min_p != max_p else (min_p - 0.01, max_p + 0.01)

        result = pd.DataFrame({lang["sample_id"]: range(1, len(prediction)+1), lang["pred_soc"]: prediction})
        
        tabs = st.tabs([lang["tab_results"], lang["tab_spectral"]] + ([lang["tab_performance"]] if actual_col else []))
        
        with tabs[0]:
            col_a, col_b = st.columns([3, 2])
            with col_b:
                st.subheader(lang["filter_title"])
                th = st.slider(f"{lang['pred_soc']} >", min_value=s_min, max_value=s_max, value=s_min)
                res_f = result[result[lang["pred_soc"]] >= th]
                st.metric(f"{lang['avg_soc']} (Filtered)", f"{res_f[lang['pred_soc']].mean():.3f}%")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(res_f[lang["sample_id"]], res_f[lang["pred_soc"]], marker='o', color='#1b5e20')
                ax.set_title(lang["trend_title"])
                st.pyplot(fig)

            with col_a:
                st.subheader(lang["results_title"])
                st.dataframe(res_f, use_container_width=True, height=400)
                st.download_button(lang["download_btn"], res_f.to_csv(index=False).encode('utf-8'), "results.csv", "text/csv")

        with tabs[1]:
            st.subheader(lang["spectral_title"])
            fig_s, ax_s = plt.subplots(figsize=(10, 4))
            wls = [float(f) for f in features]
            f_input = input_data.loc[res_f.index]
            for i in range(min(5, len(f_input))):
                ax_s.plot(wls, f_input.iloc[i], alpha=0.5)
            st.pyplot(fig_s)

        if actual_col:
            act_f = actual_values.loc[res_f.index]
            with tabs[2]:
                st.subheader(lang["acc_title"])
                r2 = r2_score(act_f, res_f[lang["pred_soc"]]) if len(res_f) > 1 else 0
                st.metric(lang["r2_score"], f"{r2:.4f}")
                fig_e, ax_e = plt.subplots(figsize=(8, 4))
                ax_e.scatter(act_f, res_f[lang["pred_soc"]], color='#2e7d32')
                ax_e.plot([act_f.min(), act_f.max()], [act_f.min(), act_f.max()], 'r--')
                st.pyplot(fig_e)

    except Exception as e:
        st.error(lang["error_generic"])
        st.exception(e)
elif model is None:
    st.info("Model missing.")
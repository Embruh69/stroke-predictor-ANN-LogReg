import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr        = joblib.load('lr_stroke_model.pkl')
    rf        = joblib.load('rf_stroke_model.pkl')
    features  = joblib.load('model_features.pkl')
    thresholds = joblib.load('model_thresholds.pkl')
    return lr, rf, features, thresholds

lr_model, rf_model, FEATURES, THRESHOLDS = load_models()

# ── Feature engineering (must mirror notebook) ─────────────────────────────────
def build_features(raw: dict) -> pd.DataFrame:
    row = pd.DataFrame([raw])
    row['age_glucose'] = row['age'] * row['avg_glucose_level']
    row['age_bmi']     = row['age'] * row['bmi']
    row['age_squared'] = row['age'] ** 2
    row['risk_score']  = (
        row['hypertension'] +
        row['heart_disease'] +
        (row['avg_glucose_level'] > 140).astype(int) +
        (row['bmi'] > 30).astype(int)
    )
    row['age_bin']     = pd.cut(row['age'], bins=[0, 40, 55, 65, 100],
                                 labels=[0, 1, 2, 3]).astype(int)
    row['glucose_bmi'] = row['avg_glucose_level'] * row['bmi']
    row['cardio_risk'] = (
        row['hypertension'] +
        row['heart_disease'] +
        row['smoking_status'].clip(lower=0)
    )
    return row[FEATURES]

def predict_risk(model, threshold, input_df):
    prob = model.predict_proba(input_df)[0][1]
    pct  = round(prob * 100, 2)
    pred = int(prob >= threshold)
    if pct < 10:
        label, color = "Low Risk",        "#2ecc71"
    elif pct < 25:
        label, color = "Moderate Risk",   "#f39c12"
    elif pct < 50:
        label, color = "High Risk",       "#e67e22"
    else:
        label, color = "Very High Risk",  "#e74c3c"
    return pct, label, color, pred

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🧠 Stroke Risk Predictor")
st.markdown("Fill in the patient information below to estimate stroke risk.")

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Government Job", "Children", "Never Worked"]
    )
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])

with col2:
    st.subheader("Health Indicators")
    hypertension  = st.selectbox("Hypertension",   ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease",   ["No", "Yes"])
    avg_glucose   = st.slider("Avg Glucose Level (mg/dL)", 50, 400, 100)
    bmi           = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    smoking       = st.selectbox(
        "Smoking Status",
        ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"]
    )

st.divider()
model_choice = st.radio(
    "Select prediction model",
    ["Logistic Regression", "Random Forest", "Both"],
    horizontal=True
)
st.divider()

# ── Encode inputs ──────────────────────────────────────────────────────────────
def encode_inputs():
    return {
        'gender':            0 if gender == "Male" else 1,
        'age':               age,
        'hypertension':      1 if hypertension == "Yes" else 0,
        'heart_disease':     1 if heart_disease == "Yes" else 0,
        'ever_married':      1 if ever_married == "Yes" else 0,
        'work_type':         {"Private": 0, "Self-employed": 1,
                              "Government Job": 2, "Children": -1, "Never Worked": -2}[work_type],
        'Residence_type':    1 if residence == "Urban" else 0,
        'avg_glucose_level': avg_glucose,
        'bmi':               bmi,
        'smoking_status':    {"Never Smoked": 0, "Formerly Smoked": 1,
                              "Smokes": 2, "Unknown": -1}[smoking],
    }

# ── Risk gauge helper ──────────────────────────────────────────────────────────
def show_result(name: str, pct: float, label: str, color: str):
    st.markdown(f"#### {name}")
    bar_pct = min(pct, 100)
    st.markdown(
        f"""
        <div style='background:#eee;border-radius:8px;overflow:hidden;height:28px;margin-bottom:6px'>
          <div style='width:{bar_pct}%;background:{color};height:100%;
                      display:flex;align-items:center;padding-left:10px;
                      color:white;font-weight:bold;font-size:14px;border-radius:8px;
                      min-width:60px'>
            {pct:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='color:{color};font-weight:bold;font-size:18px'>{label}</span>",
        unsafe_allow_html=True
    )
    st.caption(f"Raw probability: {pct:.2f}%")

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Calculate Stroke Risk", type="primary", use_container_width=True):
    raw     = encode_inputs()
    inp_df  = build_features(raw)

    st.subheader("Prediction Results")

    if model_choice == "Both":
        c1, c2 = st.columns(2)
        with c1:
            pct, label, color, _ = predict_risk(lr_model, THRESHOLDS['lr'], inp_df)
            show_result("Logistic Regression", pct, label, color)
        with c2:
            pct, label, color, _ = predict_risk(rf_model, THRESHOLDS['rf'], inp_df)
            show_result("Random Forest", pct, label, color)

    elif model_choice == "Logistic Regression":
        pct, label, color, _ = predict_risk(lr_model, THRESHOLDS['lr'], inp_df)
        show_result("Logistic Regression", pct, label, color)

    else:
        pct, label, color, _ = predict_risk(rf_model, THRESHOLDS['rf'], inp_df)
        show_result("Random Forest", pct, label, color)

    st.divider()
    st.markdown(
        "⚠️ **Disclaimer:** This tool is for informational purposes only and does "
        "not constitute medical advice. Please consult a qualified healthcare "
        "professional for diagnosis and treatment."
    )

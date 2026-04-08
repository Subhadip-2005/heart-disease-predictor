import pickle
import time
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS = {
    'Logistic Regression':    'LogisticRegression.pkl',
    'SVM':                    'svm.pkl',
    'Decision Tree':          'DecisionTreeClassifier.pkl',
    'Random Forest':          'RandomForestClassifier.pkl',
    'XGBoost':                'XGBClassifier.pkl',
    'Gradient Boosting':      'GradientBoostingClassifier.pkl',
}

FEATURES = ['Age','Sex','ChestPainType','RestingBP','Cholesterol',
            'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

PERFORMANCE = {
    'Model':     ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
                  'SVM', 'XGBoost', 'Decision Tree'],
    'Accuracy':  [86.41, 88.58, 85.86, 83.69, 85.32, 80.97],
    'Precision': [84.5,  83.2,  83.8,  83.0,  82.9,  79.5],
    'Recall':    [85.0,  83.5,  84.0,  83.2,  83.1,  80.0],
    'F1 Score':  [84.7,  83.3,  83.9,  83.1,  83.0,  79.7],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        return None

def load_scaler():
    try:
        return pickle.load(open('scaler.pkl', 'rb'))
    except FileNotFoundError:
        return None

def encode_inputs(age, sex, chest_pain, resting_bp, cholesterol,
                  fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    return pd.DataFrame([{
        'Age':             age,
        'Sex':             0 if sex == "Male" else 1,
        'ChestPainType':   ["Atypical Angina","Non-Anginal Pain","Asymptomatic","Typical Angina"].index(chest_pain),
        'RestingBP':       resting_bp,
        'Cholesterol':     cholesterol,
        'FastingBS':       1 if fasting_bs == "> 120 mg/dl" else 0,
        'RestingECG':      ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg),
        'MaxHR':           max_hr,
        'ExerciseAngina':  1 if exercise_angina == "Yes" else 0,
        'Oldpeak':         oldpeak,
        'ST_Slope':        ["Upsloping","Flat","Downsloping"].index(st_slope),
    }])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ❤️ Heart Disease Predictor")
    st.info("""
    **Models available:**
    - Decision Tree (80.97%)
    - Logistic Regression (86.41%)
    - Random Forest (88.58%)
    - SVM (83.69%)
    - XGBoost (85.32%)
    - Gradient Boosting (85.86%)
    """)
    st.metric("Total Samples", "918")
    st.metric("Features", "11")
    st.markdown("---")
    st.caption("⚠️ For educational use only. Always consult a doctor.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Prediction System")
st.caption("ML-powered early detection — 6 models, 11 clinical features")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Bulk Prediction", "📈 Model Performance"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Prediction
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Personal & Clinical**")
        age           = st.number_input("Age (years)", 1, 120, 50)
        sex           = st.selectbox("Sex", ["Male", "Female"])
        chest_pain    = st.selectbox("Chest Pain Type",
                                     ["Atypical Angina","Non-Anginal Pain","Asymptomatic","Typical Angina"])
        resting_bp    = st.number_input("Resting Blood Pressure (mm Hg)", 60, 250, 120)
        cholesterol   = st.number_input("Serum Cholesterol (mg/dl)", 0, 700, 200)
        fasting_bs    = st.selectbox("Fasting Blood Sugar", ["<=120 mg/dl", "> 120 mg/dl"])

    with col2:
        st.markdown("**Cardiac Parameters**")
        resting_ecg   = st.selectbox("Resting ECG",
                                      ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"])
        max_hr        = st.number_input("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak       = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0, step=0.1)
        st_slope      = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    if st.button("🚀 Predict", use_container_width=True, type="primary"):
        input_df = encode_inputs(age, sex, chest_pain, resting_bp, cholesterol,
                                 fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)

        scaler = load_scaler()
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values  # fallback if scaler not found

        with st.spinner("Analysing patient data..."):
            time.sleep(1)

        results, names = [], []
        for name, filename in MODELS.items():
            model = load_model(filename)
            if model:
                pred = model.predict(input_scaled)[0]
                results.append(pred)
                names.append(name)
            else:
                st.warning(f"Model file `{filename}` not found — skipped.")

        if results:
            st.divider()
            st.subheader("Results")
            cols = st.columns(3)
            for i, (name, pred) in enumerate(zip(names, results)):
                with cols[i % 3]:
                    if pred == 1:
                        st.error(f"**{name}**\n\n⚠️ Heart Disease Detected")
                    else:
                        st.success(f"**{name}**\n\n✅ No Heart Disease")

            risk_pct = sum(results) / len(results) * 100
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Models predicting risk", f"{sum(results)}/{len(results)}")
            c2.metric("Risk consensus", f"{risk_pct:.0f}%")
            c3.metric("Overall", "⚠️ High Risk" if risk_pct > 50 else "✅ Low Risk")

            if risk_pct > 50:
                st.warning("**High risk detected.** Please consult a cardiologist promptly.")
            else:
                st.success("**Low risk detected.** Continue healthy lifestyle habits.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bulk Prediction
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Bulk Prediction from CSV")

    with st.expander("📖 Required CSV format"):
        st.markdown("Your CSV must have these columns in any order:")
        st.code(", ".join(FEATURES))
        st.markdown("""
        | Column | Values |
        |---|---|
        | Sex | 0 = Male, 1 = Female |
        | ChestPainType | 0–3 |
        | FastingBS | 0 or 1 |
        | RestingECG | 0–2 |
        | ExerciseAngina | 0 or 1 |
        | ST_Slope | 0–2 |
        """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.dataframe(df_in.head(), use_container_width=True)

        if st.button("🔮 Generate Predictions", use_container_width=True):
            model = load_model('LogisticRegression.pkl')
            scaler = load_scaler()

            if model is None:
                st.error("LogisticRegression.pkl not found. Run train_model.py first.")
            else:
                X = df_in[FEATURES].values
                if scaler:
                    X = scaler.transform(X)

                df_in['Prediction'] = model.predict(X)
                df_in['Risk Level'] = df_in['Prediction'].map({1: 'High Risk', 0: 'Low Risk'})

                high = df_in['Prediction'].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Patients", len(df_in))
                c2.metric("High Risk", int(high))
                c3.metric("Low Risk", int(len(df_in) - high))

                st.dataframe(df_in, use_container_width=True)

                csv_bytes = df_in.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv_bytes,
                                   "predictions.csv", "text/csv")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Performance
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance")

    df_perf = pd.DataFrame(PERFORMANCE)

    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        fig.add_trace(go.Bar(name=metric, x=df_perf['Model'], y=df_perf[metric],
                             text=df_perf[metric], textposition='auto'))
    fig.update_layout(barmode='group', yaxis_title="Score (%)",
                      height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.info("""
    **Key insights:**
    - Random Forest gives the best accuracy (88.58%)
    - Logistic Regression is fastest and most interpretable
    - All models consistently exceed 80% accuracy
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ Medical Disclaimer: This tool is for educational purposes only. "
           "Always consult a qualified healthcare professional for medical advice.")

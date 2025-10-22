import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
from pathlib import Path

st.set_page_config(page_title="Credit Risk PD Demo", layout="centered")
st.title("💳 Credit Risk — PD Scoring Demo")

# lazy train-on-first-run fallback
def ensure_models():
    if not (Path("models/logreg_pd.joblib").exists() and Path("models/scaler.joblib").exists()):
        st.warning("Models not found — training quickly...")
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from joblib import dump
        df = pd.read_csv("data/credit_data.csv")
        X = df.drop(columns=["default"]); y = df["default"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        sc = StandardScaler().fit(X_tr)
        model = LogisticRegression(max_iter=2000).fit(sc.transform(X_tr), y_tr)
        Path("models").mkdir(exist_ok=True)
        dump(model, "models/logreg_pd.joblib"); dump(sc, "models/scaler.joblib")

ensure_models()
model = load("models/logreg_pd.joblib")
scaler = load("models/scaler.joblib")

st.subheader("Applicant Inputs")
age = st.number_input("Age", 18, 80, 35)
income = st.number_input("Annual Income ($)", 10000, 300000, 60000, step=1000)
loan_amount = st.number_input("Loan Amount ($)", 500, 100000, 15000, step=500)
credit_score = st.number_input("Credit Score", 300, 850, 650)
employment_length = st.number_input("Employment Length (years)", 0, 40, 5)
delinquencies = st.number_input("Past Delinquencies", 0, 10, 0)

X = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_length": employment_length,
    "delinquencies": delinquencies
}])

if st.button("Score Applicant"):
    proba = model.predict_proba(scaler.transform(X))[:, 1][0]
    st.metric("Predicted PD", f"{proba:.2%}")
    st.progress(min(1.0, proba))
    st.caption("Synthetic demo only — not for production decisions.")

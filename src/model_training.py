import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from joblib import dump
from pathlib import Path

df = pd.read_csv("data/credit_data.csv")
X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_s, y_train)

proba = model.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, proba)
brier = brier_score_loss(y_test, proba)
print(f"ROC AUC: {auc:.3f}")
print(f"Brier score: {brier:.4f}")

Path("models").mkdir(exist_ok=True)
dump(model, "models/logreg_pd.joblib")
dump(scaler, "models/scaler.joblib")
print("✅ Saved models/logreg_pd.joblib & models/scaler.joblib")

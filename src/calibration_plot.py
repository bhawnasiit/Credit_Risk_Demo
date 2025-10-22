calibration_plot.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from pathlib import Path

df = pd.read_csv("data/credit_data.csv")
X = df.drop(columns=["default"])
y = df["default"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

model = LogisticRegression(max_iter=2000).fit(X_tr_s, y_tr)
proba = model.predict_proba(X_te_s)[:, 1]

frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")

Path("artifacts").mkdir(exist_ok=True)
plt.figure()
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.plot(mean_pred, frac_pos, marker="o", linewidth=1, label="Model")
plt.xlabel("Predicted probability"); plt.ylabel("Observed default rate")
plt.title("Reliability Diagram (Calibration)")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/calibration_reliability.png")
print("✅ Saved artifacts/calibration_reliability.png")

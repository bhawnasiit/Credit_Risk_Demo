import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 5000  # number of applicants

def truncated_normal(mean, sd, low, high, size):
    vals = rng.normal(mean, sd, size)
    return np.clip(vals, low, high)

credit_data = pd.DataFrame({
    "age": truncated_normal(35, 10, 18, 70, N).astype(int),
    "income": truncated_normal(60000, 20000, 20000, 150000, N).astype(int),
    "loan_amount": truncated_normal(15000, 8000, 1000, 50000, N).astype(int),
    "credit_score": truncated_normal(650, 70, 300, 850, N).astype(int),
    "employment_length": rng.integers(0, 20, N),
    "delinquencies": rng.integers(0, 5, N),
})

linear = (
    5
    - 0.005 * credit_data["credit_score"]
    - 0.00003 * credit_data["income"]
    + 0.0001 * credit_data["loan_amount"]
    + 0.25 * credit_data["delinquencies"]
)
p_default = 1 / (1 + np.exp(-linear))
credit_data["default"] = rng.binomial(1, p_default)

Path("data").mkdir(parents=True, exist_ok=True)
credit_data.to_csv("data/credit_data.csv", index=False)
print("✅ Wrote data/credit_data.csv with", len(credit_data), "rows")


# example.py
import os
from dotenv import load_dotenv
load_dotenv()                      # looks for .env in project root
print("MY_SETTING:", os.getenv("MY_SETTING"))

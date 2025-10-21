import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

def ks_score(y_true, y_proba, n_bins: int = 20):
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    grouped = df.groupby("bin")
    cum_good = (grouped["y"].apply(lambda x: (1 - x).sum()).cumsum()) / (1 - df["y"]).sum()
    cum_bad = (grouped["y"].sum().cumsum()) / df["y"].sum()
    return np.max(np.abs(cum_bad.values - cum_good.values))

def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10):
    e_bins = pd.qcut(expected, n_bins, duplicates="drop")
    a_bins = pd.cut(actual, pd.IntervalIndex(e_bins.cat.categories))
    e_pct = e_bins.value_counts(normalize=True).sort_index()
    a_pct = a_bins.value_counts(normalize=True).sort_index()
    return np.sum((a_pct - e_pct) * np.log((a_pct + 1e-9) / (e_pct + 1e-9)))

if __name__ == "__main__":
    holdout = pd.read_csv("data/credit_data.csv")
    rng = np.random.default_rng(123)
    score = 0.5 * (holdout["credit_score"] - holdout["credit_score"].min()) / (
        holdout["credit_score"].max() - holdout["credit_score"].min()
    ) + 0.5 * rng.random(len(holdout))

    auc = roc_auc_score(holdout["default"], score)
    ks = ks_score(holdout["default"], score)
    stability = psi(
        expected=score.sample(frac=0.5, random_state=1),
        actual=score.drop(score.sample(frac=0.5, random_state=1).index)
    )

    print(f"Validation — AUC: {auc:.3f} | KS: {ks:.3f} | PSI: {stability:.3f}")

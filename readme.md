# Credit Risk Model Demo (Cursor AI)

A clean PD (probability of default) demo with synthetic data:
- Data generation
- Logistic Regression training
- Validation (AUC, Brier, KS, PSI)

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/make_credit_data.py
python src/model_training.py
python src/model_validation.py

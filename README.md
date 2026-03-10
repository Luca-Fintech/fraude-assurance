# FraudGuard — Healthcare Provider Fraud Detection

A machine learning system to detect fraudulent healthcare providers in Medicare claims data. Built as a full end-to-end ML project: from raw data to a production-ready scoring API with a professional web interface.

---

## Overview

Healthcare fraud costs the U.S. Medicare system an estimated **$60 billion per year**. This project trains a classification model on real Medicare claims data to identify suspicious providers, explains predictions using SHAP, and exposes the model through a REST API with a modern dashboard.

**Model performance:**
| Metric | Value |
|---|---|
| CV AUC (5-fold) | **0.9943** |
| Test AUC | 0.9682 |
| Recall (fraud) | ~72% |
| Precision (fraud) | ~52% |
| Optimal threshold | ~11.6% |
| Estimated annual ROI | ~$50M |

---

## Project Structure

```
.
├── main.ipynb              # Full ML pipeline (11 sections)
├── api.py                  # Flask REST API
├── templates/
│   └── index.html          # Web dashboard (FraudGuard)
├── models/
│   ├── fraud_detection_xgb.pkl       # Trained model artifact
│   └── scoring_providers_test.csv    # Scored test providers
└── figures/                # All generated visualizations
```

---

## Notebook — `main.ipynb`

The notebook covers the complete ML lifecycle across 11 sections:

1. **Business Context** — fraud typology, regulatory framework, ROI framing
2. **Exploratory Data Analysis** — distributions, Mann-Whitney tests, outlier detection
3. **Preprocessing** — beneficiary, inpatient, outpatient table cleaning and merge
4. **Feature Engineering** — 40+ provider-level features across 5 families (volume, financial, temporal, demographic, network)
5. **Train/Test Split + SMOTE** — stratified 80/20, class imbalance handling
6. **Model Comparison** — Logistic Regression, Random Forest, XGBoost, LightGBM
7. **Hyperparameter Optimization** — RandomizedSearchCV (40 iterations) on XGBoost
8. **SHAP Interpretability** — feature importance, beeswarm plot, fraud case analysis
9. **Threshold Optimization** — precision-recall curve, business ROI simulation
10. **Conclusions** — benchmark comparison vs. published Kaggle results
11. **Production Scoring** — batch scoring pipeline on 1,353 unlabeled test providers

---

## API

### Run locally

```bash
pip install flask flask-cors xgboost shap scikit-learn pandas numpy
python api.py
```

Server starts at `http://127.0.0.1:5001`

### Endpoints

#### `POST /api/score`

Score a single provider.

```bash
curl -X POST http://127.0.0.1:5001/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "provider_id": "PRV0001",
    "n_claims": 420,
    "n_patients": 80,
    "avg_reimb": 1200,
    "total_reimb": 504000,
    "pct_high_claims": 42,
    "avg_hosp_dur": 8.5,
    "pct_long_stays": 35,
    "pct_deceased": 12,
    "avg_n_chronic": 3.1,
    "n_unique_physicians": 2,
    "diag_entropy": 1.8
  }'
```

**Response:**
```json
{
  "provider_id": "PRV0001",
  "score": 0.9134,
  "score_pct": 91.3,
  "risk": "CRITIQUE",
  "alert": true,
  "threshold": 0.1162,
  "factors": [
    { "feature": "reimb_per_patient", "label": "Remboursement par patient très élevé", "shap": 0.842, "direction": "increase" },
    ...
  ]
}
```

Risk levels: `CRITIQUE` (≥80%) · `ELEVE` (≥60%) · `MODERE` (≥40%) · `FAIBLE` (≥20%) · `TRES_FAIBLE`

#### `POST /api/score-batch`

Score a CSV file of providers.

```bash
curl -X POST http://127.0.0.1:5001/api/score-batch \
  -F "file=@providers.csv"
```

CSV must include columns: `Provider`, `n_claims`, `n_patients`, `avg_reimb`, `total_reimb`, `pct_high_claims`, `avg_hosp_dur`, `pct_long_stays`, `pct_deceased`, `avg_n_chronic`, `n_unique_physicians`, `diag_entropy`

---

## Web Dashboard

The dashboard is served at `http://127.0.0.1:5001` when the API is running.

**Single scoring view:**
- Input form for 11 provider metrics
- Animated score ring with risk level
- Top 6 SHAP factors with direction indicators
- KPI cards: threshold, risk level, alert status, fraud probability

**Batch scoring view:**
- Drag-and-drop CSV upload
- Results table sorted by fraud score
- Color-coded risk chips
- CSV export

---

## Dataset

[Healthcare Provider Fraud Detection Analysis — Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

Three Medicare tables: `Train_Beneficiarydata`, `Train_Inpatientdata`, `Train_Outpatientdata` + labels (`Train-1542865627584.csv`).

> The raw data files are excluded from this repository (`.gitignore`). Download them from Kaggle and place them in a `data/` folder.

---

## Tech Stack

- **ML:** scikit-learn, XGBoost, imbalanced-learn, SHAP
- **API:** Flask, flask-cors
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **Data:** pandas, numpy

"""
Healthcare Provider Fraud Detection — Flask API
"""

import io
import pickle
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
with open("./models/fraud_detection_xgb.pkl", "rb") as f:
    artifact = pickle.load(f)

MODEL        = artifact["model"]
SCALER       = artifact["scaler"]
THRESHOLD    = artifact["threshold"]
FEATURE_COLS = artifact["feature_cols"]
EXPLAINER    = shap.TreeExplainer(MODEL)

DEFAULTS = {
    "n_inpatient": 5, "n_outpatient": 95, "ratio_inout": 0.05,
    "avg_claims_per_pat": 3.2, "std_reimb": 1800, "median_reimb": 366,
    "avg_deductible": 120, "cv_reimb": 1.1, "reimb_per_patient": 2200,
    "std_hosp_dur": 1.5, "avg_claim_dur": 14, "pct_long_stays": 2.5,
    "avg_age": 72, "avg_n_physicians": 1.8, "n_unique_diag": 45,
    "avg_n_diag_codes": 3.2, "avg_n_proc_codes": 0.8, "n_states": 3,
    "pct_ChronicCond_Alzheimer": 12, "pct_ChronicCond_Heartfailure": 22,
    "pct_ChronicCond_KidneyDisease": 18, "pct_ChronicCond_Cancer": 8,
    "pct_ChronicCond_ObstrPulmonary": 19, "pct_ChronicCond_Depression": 24,
    "pct_ChronicCond_Diabetes": 35, "pct_ChronicCond_IschemicHeart": 30,
    "pct_ChronicCond_Osteoporasis": 20,
    "pct_ChronicCond_rheumatoidarthritis": 14,
    "pct_ChronicCond_stroke": 9, "physician_patient_ratio": 0.45,
}

FACTOR_LABELS = {
    "total_reimb":           "Montant total remboursé anormalement élevé",
    "avg_reimb":             "Montant moyen par claim élevé (upcoding)",
    "pct_high_claims":       "Forte concentration de claims à montant élevé",
    "pct_deceased":          "Taux de patients décédés élevé (identity fraud)",
    "avg_hosp_dur":          "Durée de séjour anormalement longue",
    "n_unique_physicians":   "Nombre de médecins associés élevé (kickback)",
    "diag_entropy":          "Diversité atypique des codes diagnostic",
    "avg_claims_per_pat":    "Nombre de claims par patient élevé",
    "n_claims":              "Volume total de claims élevé",
    "reimb_per_patient":     "Remboursement par patient très élevé",
    "pct_long_stays":        "Proportion de séjours prolongés suspecte",
    "physician_patient_ratio": "Ratio médecins/patients anormal",
    "cv_reimb":              "Variabilité des montants suspecte",
    "std_reimb":             "Écart-type des montants élevé",
}


def build_vector(inputs: dict) -> pd.DataFrame:
    row = {**DEFAULTS, **inputs}
    row["ratio_inout"]        = row["n_inpatient"] / (row["n_outpatient"] + 1)
    row["avg_claims_per_pat"] = row["n_claims"] / max(row["n_patients"], 1)
    row["reimb_per_patient"]  = row["total_reimb"] / max(row["n_patients"], 1)
    row["cv_reimb"]           = row["std_reimb"] / (row["avg_reimb"] + 1)
    row["median_reimb"]       = row["avg_reimb"] * 0.7
    row["n_inpatient"]        = int(row["n_claims"] * 0.08)
    row["n_outpatient"]       = int(row["n_claims"] * 0.92)
    row["physician_patient_ratio"] = row["n_unique_physicians"] / max(row["n_patients"], 1)
    return pd.DataFrame([{f: row.get(f, 0) for f in FEATURE_COLS}])


def risk_level(score: float) -> str:
    if score >= 0.8: return "CRITIQUE"
    if score >= 0.6: return "ELEVE"
    if score >= 0.4: return "MODERE"
    if score >= 0.2: return "FAIBLE"
    return "TRES_FAIBLE"


def score_one(inputs: dict) -> dict:
    X        = build_vector(inputs)
    X_scaled = SCALER.transform(X)
    proba    = float(MODEL.predict_proba(X_scaled)[0, 1])
    sv       = EXPLAINER.shap_values(pd.DataFrame(X_scaled, columns=FEATURE_COLS))[0]

    shap_series = pd.Series(sv, index=FEATURE_COLS)
    top_factors = []
    for feat, val in shap_series.abs().nlargest(6).items():
        direction = "increase" if shap_series[feat] > 0 else "decrease"
        top_factors.append({
            "feature":   feat,
            "label":     FACTOR_LABELS.get(feat, feat.replace("_", " ").title()),
            "shap":      round(float(shap_series[feat]), 4),
            "direction": direction,
        })

    return {
        "score":      round(proba, 4),
        "score_pct":  round(proba * 100, 1),
        "risk":       risk_level(proba),
        "alert":      bool(proba >= THRESHOLD),
        "threshold":  round(THRESHOLD, 4),
        "factors":    top_factors,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", threshold=round(THRESHOLD * 100, 1))


@app.route("/api/score", methods=["POST"])
def score_single():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    try:
        inputs = {
            "n_claims":            float(data.get("n_claims", 50)),
            "n_patients":          float(data.get("n_patients", 20)),
            "avg_reimb":           float(data.get("avg_reimb", 500)),
            "total_reimb":         float(data.get("total_reimb", 25000)),
            "std_reimb":           float(data.get("avg_reimb", 500)) * 0.9,
            "pct_high_claims":     float(data.get("pct_high_claims", 10)),
            "avg_hosp_dur":        float(data.get("avg_hosp_dur", 1.5)),
            "pct_long_stays":      float(data.get("pct_long_stays", 2.0)),
            "pct_deceased":        float(data.get("pct_deceased", 0.5)),
            "avg_n_chronic":       float(data.get("avg_n_chronic", 2.5)),
            "n_unique_physicians": float(data.get("n_unique_physicians", 30)),
            "diag_entropy":        float(data.get("diag_entropy", 3.2)),
        }
        result = score_one(inputs)
        result["provider_id"] = data.get("provider_id", "")
        result["alert"] = bool(result["alert"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/score-batch", methods=["POST"])
def score_batch():
    if "file" not in request.files:
        return jsonify({"error": "CSV file required (field: file)"}), 400
    try:
        df = pd.read_csv(io.StringIO(request.files["file"].read().decode("utf-8")))
        results = []
        for _, row in df.iterrows():
            inputs = {
                "n_claims":            float(row.get("n_claims", 50)),
                "n_patients":          float(row.get("n_patients", 20)),
                "avg_reimb":           float(row.get("avg_reimb", 500)),
                "total_reimb":         float(row.get("total_reimb", 25000)),
                "std_reimb":           float(row.get("avg_reimb", 500)) * 0.9,
                "pct_high_claims":     float(row.get("pct_high_claims", 10)),
                "avg_hosp_dur":        float(row.get("avg_hosp_dur", 1.5)),
                "pct_long_stays":      float(row.get("pct_long_stays", 2.0)),
                "pct_deceased":        float(row.get("pct_deceased", 0.5)),
                "avg_n_chronic":       float(row.get("avg_n_chronic", 2.5)),
                "n_unique_physicians": float(row.get("n_unique_physicians", 30)),
                "diag_entropy":        float(row.get("diag_entropy", 3.2)),
            }
            r = score_one(inputs)
            results.append({
                "provider_id":  str(row.get("Provider", "")),
                "score":        r["score"],
                "score_pct":    r["score_pct"],
                "risk":         r["risk"],
                "alert":        r["alert"],
                "top_factor":   r["factors"][0]["label"] if r["factors"] else "",
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODELS_DIR = "models"


def load_bundle(name):
    files = [f"{name}_best_model.pkl", f"{name}_scaler.pkl", f"{name}_feature_names.pkl", f"{name}_model_info.json"]
    missing = [f for f in files if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing) +
            f"\nRun python train_{name}_model.py first."
        )

    model = joblib.load(os.path.join(MODELS_DIR, files[0]))
    scaler = joblib.load(os.path.join(MODELS_DIR, files[1]))
    features = joblib.load(os.path.join(MODELS_DIR, files[2]))
    with open(os.path.join(MODELS_DIR, files[3])) as f:
        info = json.load(f)
    return model, scaler, features, info


diabetes_model, diabetes_scaler, diabetes_features, diabetes_info = load_bundle("diabetes")
heart_model, heart_scaler, heart_features, heart_info = load_bundle("heart")

print(f"diabetes model: {diabetes_info['best_model_name']} ({diabetes_info['cv_accuracy']*100:.1f}% cv)")
print(f"heart model: {heart_info['best_model_name']} ({heart_info['cv_accuracy']*100:.1f}% cv)")


def fill_missing(row, info):
    # 0 in these columns means "not measured", training used the median for them
    for col, median in info.get("fill_values", {}).items():
        if row.get(col) == 0:
            row[col] = median
    return row


def risk_bucket(prob):
    if prob < 0.30:
        return "Low Risk", "low"
    if prob < 0.60:
        return "Moderate Risk", "moderate"
    return "High Risk", "high"


def predict(model, scaler, features, row):
    X = scaler.transform(pd.DataFrame([row])[features])
    prob = float(model.predict_proba(X)[0][1])
    label, level = risk_bucket(prob)
    return {
        "success": True,
        "probability": round(prob * 100, 1),
        "risk_label": label,
        "risk_level": level,
    }


@app.route("/")
def home():
    return render_template("index.html", diabetes_info=diabetes_info, heart_info=heart_info)


@app.route("/api/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json(silent=True)
        row = {col: float(data[col]) for col in diabetes_features}
        row = fill_missing(row, diabetes_info)
        return jsonify(predict(diabetes_model, diabetes_scaler, diabetes_features, row))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"success": False, "error": f"Invalid input: {e}"}), 400


@app.route("/api/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json(silent=True)

        row = {col: 0 for col in heart_features}
        for col in ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]:
            row[col] = float(data[col])
        row = fill_missing(row, heart_info)

        # categorical fields come in as raw text, map them onto the dummy columns
        if data["Sex"] == "M" and "Sex_M" in row:
            row["Sex_M"] = 1
        if data["ExerciseAngina"] == "Y" and "ExerciseAngina_Y" in row:
            row["ExerciseAngina_Y"] = 1
        for field in ["ChestPainType", "RestingECG", "ST_Slope"]:
            col = f"{field}_{data[field]}"
            if col in row:
                row[col] = 1

        return jsonify(predict(heart_model, heart_scaler, heart_features, row))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"success": False, "error": f"Invalid input: {e}"}), 400


@app.route("/api/model-info")
def model_info():
    return jsonify({"diabetes": diabetes_info, "heart": heart_info})


if __name__ == "__main__":
    app.run(debug=False, port=5000)

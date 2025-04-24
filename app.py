from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
import pandas as pd
import pprint
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load all models into memory (only once at startup)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {}

for file in os.listdir(MODEL_DIR):
    if file.endswith('_bundle.pkl'):
        model_name = file.replace('_model.pkl', '').replace('_', ' ').title()
        with open(os.path.join(MODEL_DIR, file), 'rb') as f:
            loaded = pickle.load(f)
            MODELS[model_name] = loaded  # {'model': ..., 'scaler': ...}

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": list(MODELS.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_name = data.get("model")
    features = data.get("features", {})

    if model_name not in MODELS:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    model_bundle = MODELS[model_name]
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    raw_features = model_bundle["features"]  # <-- the exact ordering you trained with

    # Build a DataFrame from the incoming dict
    input_df = pd.DataFrame([features])\
                   .reindex(columns=raw_features, fill_value=0)

    df_dummies = pd.get_dummies(input_df, drop_first=True)

    
    # 3) Reindex those dummies to match what scaler saw at fit time
    #    `scaler.feature_names_in_` holds the exact dummy‐column names
    expected_cols = scaler.feature_names_in_
    df_dummies    = df_dummies.reindex(columns=expected_cols, fill_value=0)

    # Scale + predict
    X_scaled = scaler.transform(df_dummies)
    raw_pred = model.predict(X_scaled)[0]
    prediction = round(float(raw_pred), 2)

    return jsonify({"prediction": prediction})

@app.route("/correlation", methods=["GET"])
def get_correlation():
    model_name = request.args.get("model")
    bundle     = MODELS.get(model_name)
    if not bundle:
        return jsonify(error="Model not found"), 404

    corr = bundle.get("corr_matrix")
    if corr is None:
        return jsonify(error="No correlation matrix stored"), 500

    # corr is already a nested dict of { feature1: {feature2: value, …}, … }
    return jsonify(matrix=corr)

@app.route("/evaluation", methods=["POST"])
def evaluation():
    data = request.get_json() or {}
    bundle = MODELS.get(data.get("model"))
    if not bundle:
        return jsonify(error="Model not found"), 404
    actual    = bundle.get("eval_actual")
    predicted = bundle.get("eval_predicted")
    if actual is None or predicted is None:
        return jsonify(error="No evaluation data stored"), 500
    
    if hasattr(actual, "tolist"):
        actual = actual.tolist()
    if hasattr(predicted, "tolist"):
        predicted = predicted.tolist()
        
    return jsonify(actual=actual, predicted=predicted)

@app.route("/feature-weights", methods=["POST"])
def get_feature_weights():
    data = request.get_json()
    model_name = data.get("model")

    if model_name not in MODELS:
        return jsonify({"error": "Model not found"}), 404

    bundle = MODELS[model_name]
    model  = bundle["model"]
    features = bundle["features"]  # the exact list you trained on

    # pick up linear or tree‐based importances
    if hasattr(model, "coef_"):
        raw = model.coef_
    elif hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    else:
        return jsonify({"error": "Model has no feature‐importance attribute"}), 400

    # pair them up
    weights = [
      {"feature": f, "weight": round(float(w), 4)}
      for f, w in zip(features, raw)
    ]

    return jsonify({"weights": weights})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
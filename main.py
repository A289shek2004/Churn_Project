# main.py
import os
import pickle
import traceback
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Basic logging
logging.basicConfig(level=logging.INFO)

def load_model():
    basedir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(basedir, 'churn_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully")
    return model

# Load model at startup (fail fast)
try:
    model = load_model()
except Exception as e:
    logging.error("Failed to load model: %s", e)
    model = None

@app.route('/')
def index():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500
    try:
        payload = request.get_json()
        # Expect payload to contain a list/array or a single record dict
        # Validate & convert to proper input format here
        if payload is None:
            return jsonify({"error": "missing json body"}), 400

        # Example: expecting payload {"features": [ ... ]}
        features = payload.get('features')
        if features is None:
            return jsonify({"error": "missing 'features' key"}), 400

        # If single record, wrap it to 2D
        import numpy as np
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = model.predict(X)
        probs = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1].tolist()

        return jsonify({"predictions": preds.tolist(), "probabilities": probs})
    except Exception:
        logging.error("Prediction error: %s", traceback.format_exc())
        return jsonify({"error": "prediction failed", "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

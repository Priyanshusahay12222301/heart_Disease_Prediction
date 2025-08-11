# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import joblib
import numpy as np
import os

# Load the saved model using a path relative to this file so it works when launched from repo root
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
try:
    model = joblib.load(_MODEL_PATH)
    load_error = None
except Exception as e:  # pragma: no cover
    model = None
    load_error = str(e)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-insecure-key')  # override in prod

@app.route('/')
def home():
    # Always just render the form page; no prediction shown here.
    return render_template('index.html')

def _compute_risk(prob: float) -> str:
    if prob < 0.3:
        return 'Low'
    if prob < 0.6:
        return 'Moderate'
    return 'High'

def _feature_importance(row_df: pd.DataFrame):
    """Return raw coefficients and simple contribution = coef * value (log-odds) for each feature."""
    if not hasattr(model, 'coef_'):
        return []
    coefs = model.coef_[0]
    features = row_df.columns
    values = row_df.iloc[0].values
    contributions = coefs * values  # log-odds space
    items = []
    for f, v, c, contr in zip(features, values, coefs, contributions):
        items.append({
            'feature': f,
            'value': float(v),
            'coefficient': float(c),
            'contribution': float(contr)
        })
    # sort by absolute contribution desc
    items.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return items

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])
    except (KeyError, ValueError):
        return redirect(url_for('home'))
    # DataFrame to keep feature names (avoid warnings & for importance)
    row_df = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
    proba = float(model.predict_proba(row_df)[0][1]) if hasattr(model, 'predict_proba') else None
    prediction = model.predict(row_df)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    risk_level = _compute_risk(proba) if proba is not None else None
    importance = _feature_importance(row_df)
    session['prediction_payload'] = {
        'result': result,
        'probability': proba,
        'risk_level': risk_level,
        'inputs': {'age': age, 'cp': cp, 'thalach': thalach},
        'importance': importance
    }
    return redirect(url_for('result'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint: expects JSON {age:int, cp:int, thalach:int}. Returns prediction, probability, risk_level, feature_importance."""
    data = request.get_json(silent=True) or {}
    missing = [k for k in ['age', 'cp', 'thalach'] if k not in data]
    if missing:
        return jsonify({'error': 'Missing fields', 'missing': missing}), 400
    try:
        age = int(data['age'])
        cp = int(data['cp'])
        thalach = int(data['thalach'])
    except ValueError:
        return jsonify({'error': 'Invalid value types; must be integers'}), 400
    row_df = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
    proba = float(model.predict_proba(row_df)[0][1]) if hasattr(model, 'predict_proba') else None
    pred = model.predict(row_df)[0]
    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    risk_level = _compute_risk(proba) if proba is not None else None
    importance = _feature_importance(row_df)
    return jsonify({
        'result': result,
        'probability': proba,
        'risk_level': risk_level,
        'importance': importance,
        'inputs': {'age': age, 'cp': cp, 'thalach': thalach}
    })

@app.route('/result')
def result():
    payload = session.get('prediction_payload')
    if not payload:
        return redirect(url_for('home'))
    return render_template('result.html', **payload)

@app.route('/health')
def health():
    if model is None:
        return jsonify(status='error', model_loaded=False, error=load_error), 500
    return jsonify(status='ok', model_loaded=True), 200

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG') == '1'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug)

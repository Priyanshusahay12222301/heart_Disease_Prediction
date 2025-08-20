# enhanced_app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
from werkzeug.exceptions import BadRequest
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the enhanced model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'enhanced_heart_disease_model.pkl')
try:
    model_package = joblib.load(_MODEL_PATH)
    model = model_package['model']
    scaler = model_package.get('scaler')
    feature_columns = model_package['feature_columns']
    model_type = model_package['model_type']
    feature_names_mapping = model_package.get('feature_names_mapping', {})
    load_error = None
    logger.info(f"Loaded {model_type} model successfully")
except Exception as e:
    model_package = None
    model = None
    scaler = None
    feature_columns = []
    model_type = None
    feature_names_mapping = {}
    load_error = str(e)
    logger.error(f"Failed to load model: {e}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-insecure-key-change-in-production')

# Input validation ranges
VALIDATION_RANGES = {
    'age': (1, 120),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (80, 250),
    'chol': (100, 600),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (50, 220),
    'exang': (0, 1),
    'oldpeak': (0, 10),
    'slope': (0, 2),
    'ca': (0, 3),
    'thal': (0, 2)
}

def validate_input(data):
    """Validate input data against expected ranges"""
    errors = []
    
    for field, (min_val, max_val) in VALIDATION_RANGES.items():
        if field in data:
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    errors.append(f"{field}: must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{field}: must be a valid number")
    
    return errors

def require_model(f):
    """Decorator to ensure model is loaded"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if model is None:
            return jsonify({'error': 'Model not available', 'details': load_error}), 500
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    """Enhanced home page with better user guidance"""
    return render_template('enhanced_index.html', 
                         feature_ranges=VALIDATION_RANGES,
                         feature_names=feature_names_mapping)

def _compute_risk(prob: float) -> dict:
    """Enhanced risk computation with more detailed analysis"""
    if prob < 0.3:
        level = 'Low'
        color = '#4caf50'
        description = 'Low risk - Continue healthy lifestyle habits'
    elif prob < 0.6:
        level = 'Moderate'
        color = '#ff9800'
        description = 'Moderate risk - Consider lifestyle modifications and regular check-ups'
    else:
        level = 'High'
        color = '#f44336'
        description = 'High risk - Strongly recommend medical consultation'
    
    return {
        'level': level,
        'color': color,
        'description': description,
        'percentage': round(prob * 100, 1)
    }

def _get_feature_importance(input_data: pd.DataFrame):
    """Enhanced feature importance analysis"""
    if model_type == 'logistic_regression' and hasattr(model, 'coef_'):
        # For logistic regression, use coefficients
        coefs = model.coef_[0]
        values = input_data.iloc[0].values
        
        # If we have a scaler, we need to work with scaled values for contribution calculation
        if scaler is not None:
            scaled_values = scaler.transform(input_data)[0]
            contributions = coefs * scaled_values
        else:
            contributions = coefs * values
        
        importance_data = []
        for i, (feature, coef, contrib) in enumerate(zip(feature_columns, coefs, contributions)):
            importance_data.append({
                'feature': feature,
                'display_name': feature_names_mapping.get(feature, feature),
                'value': float(values[i]),
                'coefficient': float(coef),
                'contribution': float(contrib),
                'abs_contribution': abs(float(contrib))
            })
    
    elif model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
        # For random forest, use feature importances
        importances = model.feature_importances_
        values = input_data.iloc[0].values
        
        importance_data = []
        for i, (feature, importance) in enumerate(zip(feature_columns, importances)):
            importance_data.append({
                'feature': feature,
                'display_name': feature_names_mapping.get(feature, feature),
                'value': float(values[i]),
                'importance': float(importance),
                'contribution': float(importance * values[i]),  # Simplified contribution
                'abs_contribution': float(importance)
            })
    
    else:
        return []
    
    # Sort by absolute contribution/importance
    importance_data.sort(key=lambda x: x['abs_contribution'], reverse=True)
    return importance_data[:5]  # Return top 5 most important features

@app.route('/predict', methods=['POST'])
@require_model
def predict():
    """Enhanced prediction with comprehensive validation"""
    try:
        # Extract and validate all required features
        form_data = {}
        missing_fields = []
        
        for field in feature_columns:
            if field in request.form and request.form[field].strip():
                try:
                    if field in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                        # Integer fields
                        form_data[field] = int(request.form[field])
                    else:
                        # Float fields
                        form_data[field] = float(request.form[field])
                except ValueError:
                    flash(f"Invalid value for {feature_names_mapping.get(field, field)}", 'error')
                    return redirect(url_for('home'))
            else:
                missing_fields.append(feature_names_mapping.get(field, field))
        
        if missing_fields:
            flash(f"Missing required fields: {', '.join(missing_fields)}", 'error')
            return redirect(url_for('home'))
        
        # Validate input ranges
        validation_errors = validate_input(form_data)
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
            return redirect(url_for('home'))
        
        # Create DataFrame with proper feature order
        input_df = pd.DataFrame([form_data], columns=feature_columns)
        
        # Make prediction
        if scaler is not None:
            # Scale features for logistic regression
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(input_scaled)[0][1])
            else:
                probability = None
        else:
            # No scaling needed for random forest
            prediction = model.predict(input_df)
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(input_df)[0][1])
            else:
                probability = None
        
        # Prepare results
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        risk_info = _compute_risk(probability) if probability is not None else None
        importance = _get_feature_importance(input_df)
        
        # Store in session
        session['prediction_payload'] = {
            'result': result,
            'probability': probability,
            'risk_info': risk_info,
            'inputs': form_data,
            'importance': importance,
            'model_type': model_type,
            'prediction_time': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {result}, Probability: {probability}")
        return redirect(url_for('result'))
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash("An error occurred during prediction. Please try again.", 'error')
        return redirect(url_for('home'))

@app.route('/api/predict', methods=['POST'])
@require_model
def api_predict():
    """Enhanced JSON API endpoint with comprehensive validation"""
    try:
        data = request.get_json(silent=True) or {}
        
        # Check for required fields
        missing_fields = [field for field in feature_columns if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': feature_columns
            }), 400
        
        # Validate input ranges
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'validation_errors': validation_errors
            }), 400
        
        # Convert to proper types and create DataFrame
        processed_data = {}
        for field in feature_columns:
            try:
                if field in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                    processed_data[field] = int(data[field])
                else:
                    processed_data[field] = float(data[field])
            except ValueError:
                return jsonify({
                    'error': f'Invalid data type for field: {field}',
                    'expected': 'integer' if field in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'] else 'float'
                }), 400
        
        input_df = pd.DataFrame([processed_data], columns=feature_columns)
        
        # Make prediction
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probability = float(model.predict_proba(input_scaled)[0][1]) if hasattr(model, 'predict_proba') else None
        else:
            prediction = model.predict(input_df)
            probability = float(model.predict_proba(input_df)[0][1]) if hasattr(model, 'predict_proba') else None
        
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        risk_info = _compute_risk(probability) if probability is not None else None
        importance = _get_feature_importance(input_df)
        
        return jsonify({
            'result': result,
            'prediction': int(prediction[0]),
            'probability': probability,
            'risk_info': risk_info,
            'feature_importance': importance,
            'model_info': {
                'type': model_type,
                'features_used': len(feature_columns)
            },
            'inputs': processed_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/result')
def result():
    """Enhanced result page"""
    payload = session.get('prediction_payload')
    if not payload:
        flash("No prediction data found. Please make a prediction first.", 'warning')
        return redirect(url_for('home'))
    return render_template('enhanced_result.html', **payload)

@app.route('/health')
def health():
    """Enhanced health check endpoint"""
    if model is None:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'error': load_error,
            'timestamp': datetime.now().isoformat()
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': model_type,
        'features_count': len(feature_columns),
        'model_info': {
            'training_date': model_package.get('training_date'),
            'test_accuracy': model_package.get('test_accuracy'),
            'test_auc': model_package.get('test_auc')
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/model-info')
@app.route('/api/model-info')
@require_model
def model_info():
    """Endpoint to get detailed model information"""
    return jsonify({
        'model_type': model_type,
        'features': feature_columns,
        'feature_descriptions': feature_names_mapping,
        'validation_ranges': VALIDATION_RANGES,
        'accuracy': model_package.get('test_accuracy'),
        'model_performance': {
            'test_accuracy': model_package.get('test_accuracy'),
            'test_auc': model_package.get('test_auc'),
            'training_date': model_package.get('training_date')
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)

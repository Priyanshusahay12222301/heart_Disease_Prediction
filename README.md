# Heart Disease Prediction (Flask Web App)

Interactive web application that predicts the likelihood of heart disease using a simple machine learning model (Logistic Regression) and a clean, modern UI.

## ✨ Features
- Web form to input health parameters (currently: `age`, `cp`, `thalach`)
- Separate result page with styled risk outcome & entered values
- Session-based Post/Redirect/Get pattern (no accidental resubmission)
- Responsive dark UI with gradient & subtle animations
- Easily retrainable model via `retrain_model.py`

## 🧠 Current Model
Trained using a tiny sample dataset (demo). It is **not** medically reliable. For real use you should:
1. Use the full UCI Heart Disease dataset (or similar)
2. Include more features (sex, trestbps, chol, fbs, exang, oldpeak, slope, ca, thal, etc.)
3. Perform train/test split + cross validation
4. Calibrate probabilities & evaluate metrics (AUC, precision/recall)

## 🗂 Project Structure
```
heart disease predictor/
	app.py                # Flask application
	retrain_model.py      # Simple script to (re)train and save model
	heart_disease_model.pkl
	templates/
		index.html          # Input form
		result.html         # Prediction result page
README.md
```

## 🚀 Quick Start
PowerShell (Windows):
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install flask joblib numpy pandas scikit-learn
python "heart disease predictor/app.py"
```
Visit: http://127.0.0.1:5000

## 🔁 Retrain the Model
Edit / extend the sample data in `retrain_model.py`, then run:
```powershell
python "heart disease predictor/retrain_model.py"
```
This overwrites `heart_disease_model.pkl`.

## 📝 Routes
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Show input form |
| `/predict` | POST | Accept form, run model, redirect to `/result` |
| `/result` | GET | Display prediction + inputs |

## 📦 Suggested `.gitignore`
```
.venv/
__pycache__/
*.pyc
*.pkl
```
(Commit the model file only if you want a fixed snapshot; otherwise ignore it.)

## 🧪 Future Improvements
- Add more clinical features & validation
- Show probability (model.predict_proba)
- Explanations (SHAP/LIME) per prediction
- REST JSON API endpoint (`/api/predict`)
- Persistence (SQLite) for prediction history
- Dockerization & deployment (Render / Railway / Azure / Heroku)

## ⚠ Disclaimer
This app is **for educational/demo purposes only** and must **not** be used for medical diagnosis or treatment decisions.



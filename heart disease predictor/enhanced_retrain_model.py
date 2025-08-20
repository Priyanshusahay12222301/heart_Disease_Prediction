# enhanced_retrain_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_enhanced_dataset():
    """Create a more comprehensive and realistic heart disease dataset"""
    np.random.seed(42)  # For reproducibility
    
    # Generate 500 synthetic patients with realistic correlations
    n_samples = 500
    
    # Age distribution (realistic)
    age = np.random.normal(54, 9, n_samples).astype(int)
    age = np.clip(age, 29, 77)
    
    # Sex (0: female, 1: male) - slightly more males in heart disease studies
    sex = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Chest pain type (0-3) - correlated with heart disease
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.25, 0.35, 0.25])
    
    # Resting blood pressure (94-200 mmHg)
    trestbps = np.random.normal(132, 18, n_samples)
    trestbps = np.clip(trestbps, 94, 200)
    
    # Cholesterol (126-564 mg/dl)
    chol = np.random.normal(246, 52, n_samples)
    chol = np.clip(chol, 126, 564)
    
    # Fasting blood sugar > 120 mg/dl (0: false, 1: true)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    # Resting ECG (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])
    
    # Maximum heart rate (71-202 bpm) - inversely correlated with age
    base_thalach = 220 - age + np.random.normal(0, 15, n_samples)
    thalach = np.clip(base_thalach, 71, 202)
    
    # Exercise induced angina (0: no, 1: yes)
    exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])
    
    # ST depression induced by exercise (0-6.2)
    oldpeak = np.random.exponential(0.8, n_samples)
    oldpeak = np.clip(oldpeak, 0, 6.2)
    
    # Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.4, 0.15])
    
    # Number of major vessels colored by fluoroscopy (0-3)
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05])
    
    # Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)
    thal = np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.15, 0.3])
    
    # Create target variable with realistic correlations
    # Higher risk factors: older age, male, certain cp types, high bp, high chol, etc.
    risk_score = (
        (age - 40) * 0.02 +
        sex * 0.3 +
        (cp == 0) * 0.4 +  # Typical angina
        (trestbps > 140) * 0.3 +
        (chol > 240) * 0.2 +
        fbs * 0.15 +
        (restecg > 0) * 0.2 +
        (thalach < 150) * 0.3 +
        exang * 0.4 +
        (oldpeak > 1) * 0.3 +
        (slope == 2) * 0.3 +  # Downsloping
        (ca > 0) * 0.4 +
        (thal == 2) * 0.3 +  # Reversible defect
        np.random.normal(0, 0.2, n_samples)  # Add some noise
    )
    
    # Convert risk score to binary target (approximately 45% positive)
    target = (risk_score > np.percentile(risk_score, 55)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    
    return data

def train_enhanced_model(data):
    """Train multiple models with proper validation"""
    print("Training enhanced heart disease prediction models...")
    
    # Prepare features and target
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = data[feature_columns]
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression with hyperparameter tuning
    print("Training Logistic Regression...")
    lr_param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_param_grid, cv=5, scoring='roc_auc')
    lr_grid.fit(X_train_scaled, y_train)
    best_lr = lr_grid.best_estimator_
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='roc_auc')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    # Evaluate models
    print("\nModel Evaluation:")
    
    # Logistic Regression
    lr_train_score = best_lr.score(X_train_scaled, y_train)
    lr_test_score = best_lr.score(X_test_scaled, y_test)
    lr_auc = roc_auc_score(y_test, best_lr.predict_proba(X_test_scaled)[:, 1])
    
    print(f"Logistic Regression - Train Accuracy: {lr_train_score:.3f}, Test Accuracy: {lr_test_score:.3f}, AUC: {lr_auc:.3f}")
    
    # Random Forest
    rf_train_score = best_rf.score(X_train, y_train)
    rf_test_score = best_rf.score(X_test, y_test)
    rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
    
    print(f"Random Forest - Train Accuracy: {rf_train_score:.3f}, Test Accuracy: {rf_test_score:.3f}, AUC: {rf_auc:.3f}")
    
    # Choose best model based on AUC
    if lr_auc >= rf_auc:
        best_model = best_lr
        best_scaler = scaler
        model_type = "logistic_regression"
        print(f"\nSelected Logistic Regression as best model (AUC: {lr_auc:.3f})")
    else:
        best_model = best_rf
        best_scaler = None  # RF doesn't need scaling
        model_type = "random_forest"
        print(f"\nSelected Random Forest as best model (AUC: {rf_auc:.3f})")
    
    # Create model package
    model_package = {
        'model': best_model,
        'scaler': best_scaler,
        'feature_columns': feature_columns,
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'test_accuracy': lr_test_score if model_type == "logistic_regression" else rf_test_score,
        'test_auc': lr_auc if model_type == "logistic_regression" else rf_auc,
        'feature_names_mapping': {
            'age': 'Age (years)',
            'sex': 'Sex (0: Female, 1: Male)',
            'cp': 'Chest Pain Type (0-3)',
            'trestbps': 'Resting Blood Pressure (mmHg)',
            'chol': 'Cholesterol (mg/dl)',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)',
            'restecg': 'Resting ECG (0: Normal, 1: ST-T abnormality, 2: LV hypertrophy)',
            'thalach': 'Maximum Heart Rate (bpm)',
            'exang': 'Exercise Induced Angina (0: No, 1: Yes)',
            'oldpeak': 'ST Depression',
            'slope': 'Slope of Peak Exercise ST (0: Up, 1: Flat, 2: Down)',
            'ca': 'Major Vessels Colored (0-3)',
            'thal': 'Thalassemia (0: Normal, 1: Fixed, 2: Reversible)'
        }
    }
    
    return model_package, X_test, y_test

def save_model_and_data(model_package, data):
    """Save the enhanced model and dataset"""
    # Save model package
    joblib.dump(model_package, 'enhanced_heart_disease_model.pkl')
    print("Enhanced model saved as 'enhanced_heart_disease_model.pkl'")
    
    # Save dataset for future use
    data.to_csv('heart_disease_dataset.csv', index=False)
    print("Dataset saved as 'heart_disease_dataset.csv'")
    
    # Also save as the original filename for backward compatibility
    joblib.dump(model_package, 'heart_disease_model.pkl')
    print("Model also saved as 'heart_disease_model.pkl' for compatibility")

if __name__ == "__main__":
    print("Creating enhanced heart disease prediction model...")
    print("=" * 60)
    
    # Create dataset
    data = create_enhanced_dataset()
    print(f"Created dataset with {len(data)} patients")
    print(f"Positive cases: {data['target'].sum()} ({data['target'].mean()*100:.1f}%)")
    
    # Train model
    model_package, X_test, y_test = train_enhanced_model(data)
    
    # Save everything
    save_model_and_data(model_package, data)
    
    print("\n" + "=" * 60)
    print("Enhanced model training completed successfully!")
    print("The model now includes 13 clinical features and uses proper validation.")

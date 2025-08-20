# retrain_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Example data (replace with your real dataset)
data = pd.DataFrame({
    'age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
    'cp': [1, 2, 1, 1, 0, 0, 1, 1, 2, 2],
    'thalach': [150, 187, 172, 178, 163, 148, 100, 120, 172, 150],
    'target': [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
})

X = data[['age', 'cp', 'thalach']]
y = data['target']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'heart_disease_model.pkl')
print('Model trained and saved as heart_disease_model.pkl')

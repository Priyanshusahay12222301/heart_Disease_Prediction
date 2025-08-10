# app.py
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

# Load the saved model
model = joblib.load('heart_disease_model.pkl')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

@app.route('/')
def home():
    # Always just render the form page; no prediction shown here.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form inputs
    age = int(request.form['age'])
    cp = int(request.form['cp'])
    thalach = int(request.form['thalach'])
    # Prepare and predict
    input_data = np.array([[age, cp, thalach]])
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    # Store details in session for the result page
    session['prediction_result'] = result
    session['input_values'] = {'age': age, 'cp': cp, 'thalach': thalach}
    return redirect(url_for('result'))

@app.route('/result')
def result():
    result_value = session.get('prediction_result')
    inputs = session.get('input_values')
    if result_value is None:
        # No prediction yet; send user to form
        return redirect(url_for('home'))
    # Do not pop immediately so user can refresh; clear only when leaving maybe.
    return render_template('result.html', result=result_value, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)

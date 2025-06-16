from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and encoder with error handling
model = None
encoder = None
model_path = 'Model/stroke_prediction_model.pkl'  # Fixed case sensitivity
encoder_path = 'Model/encoders.pkl'  # Fixed case sensitivity

if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = joblib.load(open(model_path, 'rb'))
    encoder = joblib.load(open(encoder_path, 'rb'))
else:
    print(f"Model file or encoder file not found. Please ensure '{model_path}' and '{encoder_path}' exist.")

@app.route('/')
def home():
    try:
        return render_template('index.html', bmi=None, smoking_status=None, prediction_text=None)
    except Exception as e:
        return f"Error loading template: {e}", 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoder is None:
        return render_template('index.html', prediction_text="Model files not found. Please contact the administrator.", bmi=None, smoking_status=None)
        
    if request.method == 'POST':
        # Collect data from form
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        residence_type = int(request.form['residence_type'])
        avg_glucose = float(request.form['avg_glucose'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create input array
        features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                              work_type, residence_type, avg_glucose, bmi, smoking_status]])        # Make prediction
        prediction = model.predict(features)[0]

        result = "Stroke likely" if prediction == 1 else "No Stroke likely"
        return render_template('index.html', 
                             prediction_text=result,
                             smoking_status=smoking_status,
                             bmi=bmi)

if __name__ == '__main__':
    app.run(debug=True)

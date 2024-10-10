from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Load dataset from the directory
    df = pd.read_csv('Covid Data.csv')

    # Specify the features used during model training
    required_features = [
        'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'PNEUMONIA', 
        'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 
        'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL'
    ]

    # Select only the required features
    features = df[required_features]

    # Handle non-numeric data by converting to numeric or dropping columns
    features = features.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
    features = features.fillna(0)  # Fill NaN with 0 or handle it as needed

    # Make prediction
    prediction = model.predict(features)

    return jsonify({'prediction': prediction.tolist()})

import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)



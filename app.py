from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Set the upload folder for images
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    required_fields = [
        'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'PNEUMONIA',
        'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
        'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL'
    ]

    # Check if all required fields are present
    for field in required_fields:
        if field not in request.form:
            if request.content_type == 'application/json':
                return jsonify({"error": f"Missing field: {field}"}), 400
            return render_template('result.html', error=f"Missing field: {field}")

    # Prepare the input data for prediction
    input_data = {field: request.form[field] for field in required_fields}
    input_df = pd.DataFrame([input_data])
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Make a prediction
    prediction = model.predict(input_df)

    # If the request was made via Postman (JSON), return a JSON response
    if request.content_type == 'application/json':
        return jsonify({'prediction': int(prediction[0])})

    # For browser-based access, render the result page with charts
    image_filenames = [
        'typevsage.png',
        'deathvsage.png',
        'hipervsdeath.png',
        'kde.png',
        'piedead.png',
        'class.png'
    ]

    return render_template('result.html', prediction=int(prediction[0]), images=image_filenames)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)





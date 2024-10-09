from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the CSV file from the request
    data = request.files['Covid Data.csv']
    df = pd.read_csv(data)

    # Assuming the target column is 'DEATH', drop it to get features
    features = df.drop(columns=['DEATH']).values

    # Make predictions
    predictions = model.predict(features)
    
    # Return predictions as a list
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

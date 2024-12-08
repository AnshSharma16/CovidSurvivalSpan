import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to load the dataset without caching
def load_data():
    data = pd.read_csv('Covid Data.csv')
    return data

# Load data
data = load_data()

# Check dataset loaded
st.write("Loaded Dataset:")
st.write(data.head())  # Display the first few rows to confirm it's loaded correctly

# Streamlit app title and description
st.title("Patient Survival Prediction")
st.write("This app predicts the likelihood of patient survival based on various health features.")

# Display dataset and correlation heatmap
if st.checkbox("Show Dataset"):
    st.write(data)

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Age distribution histogram
if st.checkbox("Show Age Distribution"):
    plt.figure(figsize=(6, 4))
    sns.histplot(data['AGE'], kde=True, bins=30, color='skyblue')
    plt.title("Age Distribution of Patients")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Input fields for user data
inputs = {
    'USMER': st.number_input("USMER", min_value=0, max_value=1, step=1),
    'MEDICAL_UNIT': st.number_input("Medical Unit", min_value=0, max_value=10, step=1),
    'SEX': st.number_input("Sex (0=Female, 1=Male)", min_value=0, max_value=1, step=1),
    'PATIENT_TYPE': st.number_input("Patient Type (0=Outpatient, 1=Inpatient)", min_value=0, max_value=1, step=1),
    'PNEUMONIA': st.number_input("Pneumonia (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'AGE': st.number_input("Age", min_value=0, max_value=120, step=1),
    'PREGNANT': st.number_input("Pregnant (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'DIABETES': st.number_input("Diabetes (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'COPD': st.number_input("COPD (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'ASTHMA': st.number_input("Asthma (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'INMSUPR': st.number_input("Immunosuppressed (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'HIPERTENSION': st.number_input("Hypertension (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'OTHER_DISEASE': st.number_input("Other Disease (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'CARDIOVASCULAR': st.number_input("Cardiovascular Disease (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'OBESITY': st.number_input("Obesity (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'RENAL_CHRONIC': st.number_input("Chronic Kidney Disease (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'TOBACCO': st.number_input("Tobacco Use (0=No, 1=Yes)", min_value=0, max_value=1, step=1),
    'CLASIFFICATION_FINAL': st.number_input("Classification Final", min_value=0, max_value=10, step=1)
}

# Prediction button
if st.button("Predict Survival"):
    # Prepare input features as DataFrame
    features = pd.DataFrame({
        'USMER': [inputs['USMER']],
        'MEDICAL_UNIT': [inputs['MEDICAL_UNIT']],
        'SEX': [inputs['SEX']],
        'PATIENT_TYPE': [inputs['PATIENT_TYPE']],
        'PNEUMONIA': [inputs['PNEUMONIA']],
        'AGE': [inputs['AGE']],
        'PREGNANT': [inputs['PREGNANT']],
        'DIABETES': [inputs['DIABETES']],
        'COPD': [inputs['COPD']],
        'ASTHMA': [inputs['ASTHMA']],
        'INMSUPR': [inputs['INMSUPR']],
        'HIPERTENSION': [inputs['HIPERTENSION']],
        'OTHER_DISEASE': [inputs['OTHER_DISEASE']],
        'CARDIOVASCULAR': [inputs['CARDIOVASCULAR']],
        'OBESITY': [inputs['OBESITY']],
        'RENAL_CHRONIC': [inputs['RENAL_CHRONIC']],
        'TOBACCO': [inputs['TOBACCO']],
        'CLASIFFICATION_FINAL': [inputs['CLASIFFICATION_FINAL']]
    })

    # Predict using the model
    prediction = model.predict(features)

    # Display result
    if prediction[0] == 0:
        st.success("The model predicts: Patient is alive")
    else:
        st.error("The model predicts: Patient is deceased")



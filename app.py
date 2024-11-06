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
    data = pd.read_csv(r'C:\Users\asind\CovidSurvivalSpan\Covid Data(1).csv')
    return data

# Load data
data = load_data()

# Streamlit app title and description
st.title("Patient Survival Prediction with Risk Factor Analysis")
st.write("This app predicts the likelihood of patient survival based on various health features and highlights the key risk factors.")

# Display dataset and correlation heatmap
if st.checkbox("Show Dataset"):
    st.write(data.head())

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

# Prepare input features as DataFrame
features = pd.DataFrame({k: [v] for k, v in inputs.items()})

# Prediction button
if st.button("Predict Survival"):
    # Predict using the model
    prediction = model.predict(features)

    # Display result
    if prediction[0] == 0:
        st.success("The model predicts: Patient is alive")
    else:
        st.error("The model predicts: Patient is deceased")

# Risk factor analysis
st.subheader("Risk Factor Analysis")
st.write("The following chart shows the relative importance of each feature in predicting patient survival:")

# Extract feature importance (coefficients) from the logistic regression model
coefficients = model.coef_[0]
feature_names = features.columns

# Create a DataFrame to hold the feature names and their corresponding importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(coefficients)  # Use absolute values to show magnitude
})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue=None)
plt.title("Feature Importance in Survival Prediction")
st.pyplot(plt)





import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    

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
    'USMER': st.number_input("USMER (0=First level, 1=Second level, 2=Third level)", min_value=0, max_value=2, step=1, value=None),
    'MEDICAL_UNIT': st.number_input("Medical Unit (Type of institution from National Health System)", min_value=0, max_value=10, step=1, value=None),
    'SEX': st.number_input("Sex (0=Female, 1=Male)", min_value=0, max_value=1, step=1, value=None),
    'PATIENT_TYPE': st.number_input("Patient Type (1=Returned Home, 2=Hospitalization)", min_value=1, max_value=2, step=1, value=None),
    'PNEUMONIA': st.number_input("Pneumonia (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'AGE': st.number_input("Age", min_value=0, max_value=120, step=1, value=None),
    'PREGNANT': st.number_input("Pregnant (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'DIABETES': st.number_input("Diabetes (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'COPD': st.number_input("COPD (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'ASTHMA': st.number_input("Asthma (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'INMSUPR': st.number_input("Immunosuppressed (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'HIPERTENSION': st.number_input("Hypertension (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'OTHER_DISEASE': st.number_input("Other Disease (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'CARDIOVASCULAR': st.number_input("Cardiovascular Disease (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'OBESITY': st.number_input("Obesity (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'RENAL_CHRONIC': st.number_input("Chronic Kidney Disease (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'TOBACCO': st.number_input("Tobacco Use (0=No, 1=Yes)", min_value=0, max_value=1, step=1, value=None),
    'CLASIFFICATION_FINAL': st.number_input("Classification Final", min_value=0, max_value=10, step=1, value=None)
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

    # Display patient-specific suggestions
    st.subheader("Personalized Recommendations")
    suggestions = []

    # Add recommendations based on each feature
    if inputs['SEX'] == 1:  # Male
        suggestions.append("For males, maintaining a healthy lifestyle, including regular checkups, is essential.")
    else:  # Female
        suggestions.append("Women are encouraged to monitor health parameters and consult for regular screenings.")

    if inputs['AGE'] > 60:
        suggestions.append("Elderly patients should focus on strengthening immunity and avoiding exposure to infections.")

    if inputs['TOBACCO'] == 1:
        suggestions.append("As a smoker, it’s recommended to reduce or quit smoking to improve respiratory health.")
    else:
        suggestions.append("Maintain a smoke-free lifestyle for optimal lung health.")

    if inputs['DIABETES'] == 1:
        suggestions.append("For patients with diabetes, managing blood sugar and following a balanced diet is crucial.")

    if inputs['OBESITY'] == 1:
        suggestions.append("Consider a balanced diet and regular exercise to manage weight and reduce health risks.")

    # Display all recommendations
    for suggestion in suggestions:
        st.write("- " + suggestion)

# Risk factor analysis
st.subheader("Risk Factor Analysis")
st.write("The following chart shows the relative importance of each feature in predicting patient survival:")

# Extract feature importance (coefficients) from the logistic regression model
coefficients = model.coef_[0]
feature_names = features.columns

# Create a DataFrame to hold the feature names and their corresponding importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(coefficients),  # Use absolute values to show magnitude
    'Explanation': [
        "Impact of USMER status on survival", "Medical unit's impact on survival",
        "Sex as a factor in survival (0=Female, 1=Male)", "Patient type's impact (Returned home or Hospitalized)",
        "Presence of Pneumonia", "Age as a risk factor", "Pregnancy status impact",
        "Presence of Diabetes", "Presence of COPD", "Presence of Asthma",
        "Immunosuppression status", "Presence of Hypertension", "Other diseases as risk factors",
        "Cardiovascular disease impact", "Obesity's effect", "Chronic kidney disease impact",
        "Tobacco use's effect", "Classification final category"
    ]  # Add explanations for each feature here
})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot with Plotly for hover tooltips
fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
             hover_data={'Feature': True, 'Importance': True, 'Explanation': True},
             title="Feature Importance in Survival Prediction")

# Display the plot
st.plotly_chart(fig)


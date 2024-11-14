import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    
import plotly.graph_objects as go

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to load the dataset without caching
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\asind\CovidSurvivalSpan\Covid Data(1).csv')
    return data

# Load data
data = load_data().sample(1000)  # Use only 1000 samples for quicker testing


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

# Gender Distribution Pie Chart
st.subheader("Gender Distribution")
gender_counts = data['SEX'].value_counts()
gender_labels = ['Female', 'Male']
fig_gender = px.pie(values=gender_counts, names=gender_labels, title="Gender Distribution of Patients")
st.plotly_chart(fig_gender)

# Patient Type Distribution Pie Chart
st.subheader("Patient Type Distribution")
patient_type_counts = data['PATIENT_TYPE'].value_counts()
patient_type_labels = ['Not Hospitalized', 'Hospitalized']
fig_patient_type = px.pie(values=patient_type_counts, names=patient_type_labels, title="Patient Type Distribution")
st.plotly_chart(fig_patient_type)

# Survival vs. Death Pie Chart
st.subheader("Survival vs. Death Distribution")
death_counts = data['DEATH'].value_counts()
death_labels = ['Alive', 'Deceased']
fig_death = px.pie(values=death_counts, names=death_labels, title="Survival vs. Death Distribution")
st.plotly_chart(fig_death)

# Patient Type vs. Age Bar Plot
st.subheader("Patient Type vs. Age")
fig_age_patient_type = px.bar(data, x='PATIENT_TYPE', y='AGE', 
                              color='PATIENT_TYPE', 
                              labels={'PATIENT_TYPE': 'Patient Type', 'AGE': 'Age'},
                              title="Patient Type vs. Age Distribution",
                              category_orders={'PATIENT_TYPE': [0, 1]},
                              color_discrete_map={0: 'blue', 1: 'green'})
st.plotly_chart(fig_age_patient_type)

# Age Distribution for Deceased Patients Bar Plot
st.subheader("Age Distribution for Deceased Patients")
deceased_data = data[data['DEATH'] == 1]
fig_death_age = px.histogram(deceased_data, x='AGE', nbins=30, 
                             title="Age Distribution for Deceased Patients", 
                             color_discrete_sequence=['red'])
st.plotly_chart(fig_death_age)

# Hypertension vs. Death Count Plot
st.subheader("Hypertension vs. Death Count")
fig_hypertension_death = sns.countplot(x='HIPERTENSION', hue='DEATH', data=data, palette='husl')
plt.title("Hypertension vs. Death Count")
plt.xlabel("Hypertension (0=No, 1=Yes)")
plt.ylabel("Count")
st.pyplot(plt.gcf())
plt.clf()  # Clear the plot for next visualization

# Diabetes vs. Death Count Plot
st.subheader("Diabetes vs. Death Count")
fig_diabetes_death = sns.countplot(x='DIABETES', hue='DEATH', data=data, palette='husl')
plt.title("Diabetes vs. Death Count")
plt.xlabel("Diabetes (0=No, 1=Yes)")
plt.ylabel("Count")
st.pyplot(plt.gcf())
plt.clf()  # Clear the plot for next visualization

# Sex vs. Death Count Plot
st.subheader("Sex vs. Death Count")
fig_sex_death = sns.countplot(x='SEX', hue='DEATH', data=data, palette='husl')
plt.title("Sex vs. Death Count")
plt.xlabel("Sex (0=Female, 1=Male)")
plt.ylabel("Count")
st.pyplot(plt.gcf())
plt.clf()  # Clear the plot for next visualization


# Input fields for user data with sliders and radio buttons
inputs = {
    'USMER': st.radio(
        "USMER (0=First level, 1=Second level, 2=Third level)", 
        [0, 1, 2],
        index=None  # default option can be None
    ),
    'MEDICAL_UNIT': st.slider(
        "Medical Unit (Type of institution from National Health System)", 
        0, 10, value=None
    ),
    'SEX': st.radio("Sex (0=Female, 1=Male)", [0, 1]),
    'PATIENT_TYPE': st.radio(
        "Patient Type (0=Not Hospitalized, 1=Hospitalization)", 
        [0, 1],
        index=None  # default option can be None
    ),
    'PNEUMONIA': st.radio("Pneumonia (0=No, 1=Yes)", [0, 1]),
    'AGE': st.slider("Age", 0, 120, value=None),
    'PREGNANT': st.radio("Pregnant (0=No, 1=Yes)", [0, 1]),
    'DIABETES': st.radio("Diabetes (0=No, 1=Yes)", [0, 1]),
    'COPD': st.radio("COPD (0=No, 1=Yes)", [0, 1]),
    'ASTHMA': st.radio("Asthma (0=No, 1=Yes)", [0, 1]),
    'INMSUPR': st.radio("Immunosuppressed (0=No, 1=Yes)", [0, 1]),
    'HIPERTENSION': st.radio("Hypertension (0=No, 1=Yes)", [0, 1]),
    'OTHER_DISEASE': st.radio("Other Disease (0=No, 1=Yes)", [0, 1]),
    'CARDIOVASCULAR': st.radio("Cardiovascular Disease (0=No, 1=Yes)", [0, 1]),
    'OBESITY': st.radio("Obesity (0=No, 1=Yes)", [0, 1]),
    'RENAL_CHRONIC': st.radio("Chronic Kidney Disease (0=No, 1=Yes)", [0, 1]),
    'TOBACCO': st.radio("Tobacco Use (0=No, 1=Yes)", [0, 1]),
    'CLASIFFICATION_FINAL': st.slider("Classification Final", 0, 10, value=None)
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

# Radar chart of patient profile
st.subheader("Patient Profile (Radar Chart)")
categories = list(inputs.keys())
values = list(inputs.values())

# Create radar chart
fig_radar = go.Figure()

# Add a trace for the patient profile
fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Patient Profile'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]  # Adjust range based on your input scale
        )
    ),
    showlegend=False,
    title="Radar Chart of Patient Profile"
)

# Display the radar chart
st.plotly_chart(fig_radar)
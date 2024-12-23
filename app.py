import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open('C:\\Users\\asind\\CovidSurvivalSpan\\model (1).pkl', 'rb') as file:
    model = pickle.load(file)


# Set the page title and layout
st.set_page_config(page_title="Patient Survival Prediction", layout="wide")

# Sidebar Layout with icons and labels
sidebar_options = {
    "Home": "🏠 Home",
    "Prediction Model": "📊 Prediction Model",
    "Risk Factor Analysis": "⚠️ Risk Factors",
    "Patient Data Insights": "📋 Patient Insights"
}

st.sidebar.title("Navigation")
selected_sidebar_option = st.sidebar.radio("Select an option", list(sidebar_options.values()))

# Handle sidebar navigation
if selected_sidebar_option == "🏠 Home":
    st.header("Welcome to the Patient Survival Prediction App")
    st.markdown("Explore the prediction models and risk factors for better healthcare outcomes.")
    # Add additional home page content here, e.g., an introduction to your app

elif selected_sidebar_option == "📊 Prediction Model":
    st.header("Prediction Model")
    st.markdown("Use this section to input data and predict patient survival.")
    
    # Prediction input form
    prediction_input = st.text_input("Enter patient details")
    if st.button("Predict"):
        st.write("Prediction Result: ...")
    # Add additional features for this section

elif selected_sidebar_option == "⚠️ Risk Factor Analysis":
    st.header("Risk Factor Analysis")
    st.markdown("Analyze the factors affecting patient survival.")
    # Add risk factor analysis widgets here (e.g., selection boxes, charts)

elif selected_sidebar_option == "📋 Patient Data Insights":
    st.header("Patient Data Insights")
    st.markdown("Get insights from historical patient data.")
    # Display insights from your data (e.g., tables, charts)

# Add some custom styling for hover effects on buttons
st.markdown("""
    <style>
        button:hover {
            background-color: #FF5733;
        }
    </style>
""", unsafe_allow_html=True)
                   
# Add global CSS styles
def add_custom_css():
    st.markdown("""
    <style>
        /* General Background and Font */
        body {
            background-color: #f4f4f4; /* Light gray background */
            font-family: 'Arial', sans-serif; /* Modern font */
        }

        /* Header Style */
        h1, h2, h3 {
            color: #2c3e50; /* Dark blue for headers */
        }

        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }

        h2, h3 {
            text-align: left;
            margin: 10px 0;
        }

        /* Custom Container */
        .custom-container {
            background-color: #ffffff; /* White background */
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Soft shadow */
            padding: 20px;
            margin: 20px auto;
            max-width: 800px; /* Fixed width for readability */
        }

        /* Button Style */
        button {
            background-color: #007bff; /* Primary blue color */
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3; /* Darker blue on hover */
        }

        /* Disclaimer Box */
        .disclaimer {
            background-color: #f9d6d5; /* Soft red background */
            border: 2px solid #ff5e57; /* Strong red border */
            border-radius: 10px;
            padding: 15px;
            color: #333;
            font-size: 14px;
            text-align: center;
        }

        /* Input Box */
        input[type="text"], select {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            width: calc(100% - 20px); /* Full width minus padding */
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS function at the start of your app
add_custom_css()

# Function to load the dataset without caching
@st.cache_data
def load_data():
    data = pd.read_csv('Covid Data(1).csv')
    return data

# Load data
data = load_data().sample(1000)

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
        0, 12, value=None
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
    title="Patient Profile Radar Chart"
)

# Display the radar chart
st.plotly_chart(fig_radar)

st.markdown("""
<div style="background-color: #f9d6d5; 
            padding: 15px; 
            border-radius: 10px; 
            border: 2px solid #ff5e57; 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
    <h3 style="color: #b22222; text-align: center; margin-bottom: 10px;">Disclaimer</h3>
    <p style="color: #333; font-size: 16px; line-height: 1.5;">
        This app is for educational purposes and not intended for medical diagnosis or treatment.
        Consult a healthcare provider for personalized medical advice.
    </p>
</div>
""", unsafe_allow_html=True)


from sklearn.metrics import accuracy_score
# Separate features and labels (assuming 'DEATH' is the label column)
X = data.drop(columns=['DEATH'])
y = data['DEATH']

# Calculate accuracy on the dataset (you can use a separate validation/test set)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
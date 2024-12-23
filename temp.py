import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model
with open('C:\\Users\\asind\\CovidSurvivalSpan\\model (1).pkl', 'rb') as file:
    model = pickle.load(file)

# Function to load the dataset without caching
@st.cache_data
def load_data():
    data = pd.read_csv('Covid Data(1).csv')
    return data

# Load data
data = load_data().sample(1000)

# Input form for prediction
def prediction_model():
    st.header("Prediction Model")
    st.write("Fill in the patient details below to predict survival status.")

    # Input form
    inputs = {
        'USMER': st.radio("USMER (0=First level, 1=Second level, 2=Third level)", [0, 1, 2]),
        'MEDICAL_UNIT': st.slider("Medical Unit (Type of institution from National Health System)", 0, 12),
        'SEX': st.radio("Sex (0=Female, 1=Male)", [0, 1]),
        'PATIENT_TYPE': st.radio("Patient Type (0=Not Hospitalized, 1=Hospitalized)", [0, 1]),
        'PNEUMONIA': st.radio("Pneumonia (0=No, 1=Yes)", [0, 1]),
        'AGE': st.slider("Age", 0, 120),
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
        'CLASIFFICATION_FINAL': st.slider("Classification Final", 0, 10)
    }

    # Save inputs to session state
    st.session_state.inputs = inputs

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

def eda_section():
    st.header("Exploratory Data Analysis (EDA)")

    # Age Distribution Histogram
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
    fig_age_patient_type = px.bar(
        data, 
        x='PATIENT_TYPE', 
        y='AGE',
        color='PATIENT_TYPE',
        labels={'PATIENT_TYPE': 'Patient Type', 'AGE': 'Age'},
        title="Patient Type vs. Age Distribution",
        category_orders={'PATIENT_TYPE': [0, 1]},
        color_discrete_map={0: 'blue', 1: 'green'}
    )
    st.plotly_chart(fig_age_patient_type)

    # Age Distribution for Deceased Patients Bar Plot
    st.subheader("Age Distribution for Deceased Patients")
    deceased_data = data[data['DEATH'] == 1]
    fig_death_age = px.histogram(
        deceased_data, 
        x='AGE', 
        nbins=30,
        title="Age Distribution for Deceased Patients", 
        color_discrete_sequence=['red']
    )
    st.plotly_chart(fig_death_age)

    # Hypertension vs. Death Count Plot
    st.subheader("Hypertension vs. Death Count")
    sns.countplot(x='HIPERTENSION', hue='DEATH', data=data, palette='husl')
    plt.title("Hypertension vs. Death Count")
    plt.xlabel("Hypertension (0=No, 1=Yes)")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the plot for the next visualization

    # Diabetes vs. Death Count Plot
    st.subheader("Diabetes vs. Death Count")
    sns.countplot(x='DIABETES', hue='DEATH', data=data, palette='husl')
    plt.title("Diabetes vs. Death Count")
    plt.xlabel("Diabetes (0=No, 1=Yes)")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the plot for the next visualization

    # Sex vs. Death Count Plot
    st.subheader("Sex vs. Death Count")
    sns.countplot(x='SEX', hue='DEATH', data=data, palette='husl')
    plt.title("Sex vs. Death Count")
    plt.xlabel("Sex (0=Female, 1=Male)")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the plot for the next visualization

def display_heatmap(data):
    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

# Patient insights with radar chart
def patient_insights():
    st.header("Patient Insights")
    if "inputs" not in st.session_state:
        st.warning("Please fill out the prediction form first to view patient insights.")
        return

    inputs = st.session_state.inputs
    categories = list(inputs.keys())
    values = list(inputs.values())

    # Determine dynamic range for radar chart
    max_value = max(values) if values else 1
    min_value = min(values) if values else 0

    # Create radar chart
    fig_radar = go.Figure()
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
                range=[min_value, max_value]
            )
        ),
        showlegend=False,
        title="Patient Profile Radar Chart"
    )

    st.plotly_chart(fig_radar)

def display_dataset(data):
    # Display dataset and correlation heatmap
    if st.checkbox("Show Dataset"):
        st.write(data)


# Navigation section
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a section", ["Home", 'Dataset', "Prediction Model", "Patient Insights","Heatmap", "EDA",'About'])

    if options == "Dataset":
        st.header("Dataset")
        st.write("The dataset used in this app contains health-related information of patients.")
        display_dataset(data)
    elif options == "Prediction Model":
        prediction_model()
    elif options == "EDA":
        eda_section()
    elif options == "Patient Insights":
        patient_insights()
    elif options == "Heatmap":
        display_heatmap(data)
    elif options == "Home":
        st.title("Welcome to the Patient Survival Prediction App!")
        st.write("Use the sidebar to navigate through the app.")
    elif options == "About":
        st.title("About")
        st.write("""
        This app predicts the survival status of patients based on health-related inputs.
        It uses a trained logistic regression model to make predictions and provides visual
        insights through radar charts.
        """)

if __name__ == "__main__":
    main()

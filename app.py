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
    # Inline CSS for styling
    st.markdown(
        """
        <style>
        .header {
            color: white;
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 1em;
        }
        .input-section label {
            color: white;
            font-size: 1.2em;
            font-weight: bold;
        }
        .stSlider > div {
            color: #16A085;
        }
        .stButton > button {
            background-color: #28a745;
            color: white;
            font-size: 1.1em;
            padding: 0.5em;
            border-radius: 5px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #218838;
            color: #ffffff;
        }
        .stRadio div {
            font-size: 1.1em;
            color: white;
        }
        .success-message {
            color: green;
            font-size: 1.2em;
            font-weight: bold;
        }
        .error-message {
            color: #E74C3C;
            font-size: 1.2em;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown("<div class='header'>Prediction Model</div>", unsafe_allow_html=True)
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
        'CLASIFFICATION_FINAL': st.slider("Classification Final", 0, 10),
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
            st.markdown("<div class='success-message'>The model predicts: Patient is alive</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='error-message'>The model predicts: Patient is deceased</div>", unsafe_allow_html=True)   

def eda_section():
    # Age Distribution
    with st.container():
        st.subheader("Age Distribution")
        if st.checkbox("Show Age Distribution"):
            plt.figure(figsize=(6, 4))
            sns.histplot(data['AGE'], kde=True, bins=30, color='skyblue')
            plt.title("Age Distribution of Patients")
            plt.xlabel("Age")
            plt.ylabel("Frequency")
            st.pyplot(plt)

    # Gender Distribution
    with st.container():
        st.subheader("Gender Distribution")
        gender_counts = data['SEX'].value_counts()
        gender_labels = ['Female', 'Male']
        fig_gender = px.pie(
            values=gender_counts,
            names=gender_labels,
            title="Gender Distribution of Patients",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    # Patient Type Distribution
    with st.container():
        st.subheader("Patient Type Distribution")
        patient_type_counts = data['PATIENT_TYPE'].value_counts()
        patient_type_labels = ['Not Hospitalized', 'Hospitalized']
        fig_patient_type = px.pie(
            values=patient_type_counts,
            names=patient_type_labels,
            title="Patient Type Distribution",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_patient_type, use_container_width=True)

    # Survival vs. Death Distribution
    with st.container():
        st.subheader("Survival vs. Death Distribution")
        death_counts = data['DEATH'].value_counts()
        death_labels = ['Alive', 'Deceased']
        fig_death = px.pie(
            values=death_counts,
            names=death_labels,
            title="Survival vs. Death Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_death, use_container_width=True)

    # Patient Type vs. Age Distribution
    with st.container():
        st.subheader("Patient Type vs. Age Distribution")
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
        st.plotly_chart(fig_age_patient_type, use_container_width=True)

    # Age Distribution for Deceased Patients
    with st.container():
        st.subheader("Age Distribution for Deceased Patients")
        deceased_data = data[data['DEATH'] == 1]
        fig_death_age = px.histogram(
            deceased_data,
            x='AGE',
            nbins=30,
            title="Age Distribution for Deceased Patients",
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig_death_age, use_container_width=True)

    # Hypertension vs. Death Count
    with st.container():
        st.subheader("Hypertension vs. Death Count")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='HIPERTENSION', hue='DEATH', data=data, palette='husl')
        plt.title("Hypertension vs. Death Count")
        plt.xlabel("Hypertension (0=No, 1=Yes)")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()

    # Diabetes vs. Death Count
    with st.container():
        st.subheader("Diabetes vs. Death Count")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='DIABETES', hue='DEATH', data=data, palette='husl')
        plt.title("Diabetes vs. Death Count")
        plt.xlabel("Diabetes (0=No, 1=Yes)")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()

    # Sex vs. Death Count
    with st.container():
        st.subheader("Sex vs. Death Count")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='SEX', hue='DEATH', data=data, palette='husl')
        plt.title("Sex vs. Death Count")
        plt.xlabel("Sex (0=Female, 1=Male)")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()

def display_heatmap(data):
    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        st.info("This heatmap displays the correlation between the features in the dataset. A value closer to 1 or -1 indicates strong positive or negative correlation, respectively.")

        # Calculate correlation matrix
        correlation_matrix = data.corr()

        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap with improved styling
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            linewidths=0.5, 
            linecolor='white', 
            cbar_kws={"shrink": 0.8, "aspect": 30, "orientation": "vertical"}, 
            square=True, 
            ax=ax
        )

        # Beautify the heatmap
        ax.set_title(
            "Feature Correlation Heatmap",
            fontsize=16,
            fontweight='bold',
            pad=20,
            color='darkblue'
        )
        plt.xticks(fontsize=10, rotation=45, ha="right", color='black')
        plt.yticks(fontsize=10, rotation=0, color='black')
        plt.tight_layout()  # Ensure no clipping of labels

        # Display the heatmap
        st.pyplot(fig)


# Patient insights with radar chart
def patient_insights():
    # Check if prediction form inputs are available
    if "inputs" not in st.session_state:
        st.warning("Please fill out the prediction form first to view patient insights.")
        return

    # Retrieve inputs and prepare data
    inputs = st.session_state.inputs
    categories = list(inputs.keys())
    values = list(inputs.values())

    # Handle empty input edge cases
    max_value = max(values) if values else 1
    min_value = min(values) if values else 0

    # Define color palette for radar chart
    radar_fill_color = "rgba(0, 123, 255, 0.4)"  # Semi-transparent blue
    radar_line_color = "rgb(0, 123, 255)"       # Solid blue
    background_color = "#f9f9f9"                # Light gray background
    color='white'

    # Create radar chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=radar_fill_color,
        line=dict(color=radar_line_color, width=2),
        marker=dict(size=5, symbol="circle", color=radar_line_color),
        name='Patient Profile'
    ))

    # Update layout for better UI
    fig_radar.update_layout(
        polar=dict(
            bgcolor=background_color,  # Background for polar chart
            angularaxis=dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                tickfont=dict(size=12, color="white")  # Adjust font size and color
            ),
            radialaxis=dict(
                visible=True,
                range=[min_value, max_value],
                gridcolor="black",
                gridwidth=0.5,
                tickfont=dict(size=12, color="black")  # Adjust font size and color
            )
        ),
        template="plotly_white",
        margin=dict(t=50, b=50, l=50, r=50),  # Margins for spacing
        title=dict(
            text="Patient Profile Radar Chart",
            x=0.5,
            font=dict(size=18, color="white")  # Title font adjustments
        ),
        showlegend=False
    )

    # Display radar chart in Streamlit
    with st.container():
        st.subheader("Patient Profile Visualization")
        st.info("The radar chart below provides a detailed visualization of the patient's health-related inputs across multiple categories.")
        st.plotly_chart(fig_radar, use_container_width=True)


def display_dataset(data):

    # Add information about the dataset source
    st.markdown("""
    The dataset used in this project can be found at [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data).  
    It was provided by the [Mexican Government](https://www.gob.mx/) as part of their COVID-19 analysis initiative.
    """)

    # Display dataset with UI enhancements
    if st.checkbox("Show Dataset"):
        st.markdown("### Dataset Preview")
        st.dataframe(data.style.set_table_styles(
            [{
                'selector': 'thead',
                'props': [('background-color', '#4CAF50'), 
                          ('color', 'white'), 
                          ('font-size', '16px')]
            },
            {
                'selector': 'tbody tr:hover',
                'props': [('background-color', '#f2f2f2')]
            }]
        ), use_container_width=True)

        # Show summary stats for better insights
        if st.checkbox("Show Summary Statistics"):
            st.markdown("### Summary Statistics")
            st.dataframe(data.describe().style.background_gradient(cmap='coolwarm'))

# Navigation section
# Apply custom CSS
def apply_css():
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #2E4053;
            color: white;
        }
        .stButton > button {
            background-color: #28a745;
            color: white;
            border-radius: 5px;
        }
        .main-title {
            color: #2E86C1;
            font-size: 2.5em;
            font-weight: bold;
        }
        .header {
            color: #16A085;
            font-size: 1.5em;
            margin-bottom: 1em;
        }
        .footer {
            text-align: center;
            color: #7D3C98;
            margin-top: 2em;
            font-size: 0.9em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Navigation section
def main():
    apply_css()

    st.sidebar.title("🌟 Navigation")
    options = st.sidebar.radio(
        "📌 Select a section",
        ["🏠 Home", "📊 Dataset", "🤖 Prediction Model", "📋 Patient Insights", "🔥 Heatmap", "🔍 EDA", "ℹ️ About"]
    )

    if options == "📊 Dataset":
        st.header("📊 Dataset")
        st.write("The dataset used in this app contains health-related information of patients.")
        # Example placeholder for dataset
        display_dataset(data)
    elif options == "🤖 Prediction Model":
        st.header("🤖 Prediction Model")
        st.write("🔮 Use the model to predict survival status.")
        prediction_model()
    elif options == "🔍 EDA":
        st.header("🔍 Exploratory Data Analysis (EDA)")
        st.write("📈 Visualize and understand the dataset.")
        eda_section()
    elif options == "📋 Patient Insights":
        st.header("📋 Patient Insights")
        st.write("🩺 Explore patient-specific recommendations.")
        patient_insights()

    elif options == "🔥 Heatmap":
        st.header("🔥 Correlation Heatmap")
        st.write("🗺️ Visualize correlations between features.")
        display_heatmap(data)
    elif options == "🏠 Home":
        st.markdown("<h1 class='main-title'>Welcome to the Patient Survival Prediction App! 🩺</h1>", unsafe_allow_html=True)
        st.write("🔹 Use the sidebar to navigate through the app.")
        st.write("🔹 Explore datasets, make predictions, and gain insights!")
    elif options == "ℹ️ About":
        st.markdown("<h1 class='header'>About ℹ️</h1>", unsafe_allow_html=True)
        st.write("""
        This app predicts the survival status of patients based on health-related inputs.
        It uses a trained logistic regression model to make predictions and provides visual
        insights through radar charts.
        """)
        st.markdown("<p class='footer'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
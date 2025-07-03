import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import xgboost as xgb
import os

# Set page configuration at the very top
st.set_page_config(page_title="Healthcare Dashboard", layout="wide", page_icon="💡")

# Function to load the model
def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model("xgboost_patient_model.json")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the data
file_path = 'final_cleaned_patient_data.csv'
df = pd.read_csv(file_path)

# Sidebar navigation
st.sidebar.title('Healthcare Data Dashboard')

# Team Members Section
st.sidebar.markdown("### 🏆 Team Members:")
team_members = [
    "1. A.Hemanth Kumar",
    "2. M.Simhadri", 
    "3. V.Satya Sai Narasimha Murthy", 
    "4. P.Achyuth Veera Venkata Lakshmi Narayana", 
    "5. N.Hemanth Kumar"
]
for member in team_members:
    st.sidebar.text(member)

# Add a divider
st.sidebar.markdown("---")

# Section navigation
option = st.sidebar.selectbox('Choose a section', ['Data Overview', 'Data Visualization', 'Interactive Reports', 'Correlation Analysis', 'Data Insights', 'Patient Outcome Prediction'])

# Apply a Streamlit theme with a dark background for a modern look
st.markdown("""
    <style>
        h1 { color: #00FFAA; }
        .stApp { background-color: #121212; color: #FFFFFF; }
        .sidebar .sidebar-content { background-color: #333333; color: #FFFFFF; }
        .css-1d391kg { color: #FFFFFF; }
        .css-18e3th9 { background-color: #1E1E1E; }
    </style>
""", unsafe_allow_html=True)

# Data Overview Section
if option == 'Data Overview':
    st.title('📊 Data Overview')
    st.write(df.head())
    st.write(f"Dataset Shape: {df.shape}")
    st.write(f"Column Names: {df.columns.tolist()}")
    st.write("Basic Statistical Overview:")
    st.write(df.describe())

    if st.checkbox('Show Missing Values'): 
        st.write(df.isnull().sum())

# Data Visualization Section
elif option == 'Data Visualization':
    st.title('📈 Data Visualization')
    column = st.selectbox('Select Column for Visualization', df.columns)
    plot_type = st.radio('Choose plot type', ['Histogram', 'Boxplot', 'Violin Plot', 'Scatter Plot', 'Line Plot', 'Animated Plot'])

    if plot_type == 'Animated Plot':
        time_col = st.selectbox('Select Time Column (if applicable)', df.columns)
        fig = px.scatter(df, x=column, y=column, animation_frame=time_col, size_max=60)
    elif plot_type == 'Histogram':
        fig = px.histogram(df, x=column, marginal='box', nbins=30)
    elif plot_type == 'Boxplot':
        fig = px.box(df, y=column)
    elif plot_type == 'Violin Plot':
        fig = px.violin(df, y=column, box=True, points='all')
    elif plot_type == 'Scatter Plot':
        x_col = st.selectbox('Select X axis', df.columns)
        fig = px.scatter(df, x=x_col, y=column, color=column)
    elif plot_type == 'Line Plot':
        x_col = st.selectbox('Select X axis for Line Plot', df.columns)
        fig = px.line(df, x=x_col, y=column)
    
    st.plotly_chart(fig)

# Correlation Analysis Section
elif option == 'Correlation Analysis':
    st.title('🔎 Correlation Analysis')
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# Interactive Reports Section
elif option == 'Interactive Reports':
    st.title('📂 Interactive Reports')
    st.write("Filter and explore the data.")
    selected_columns = st.multiselect('Select columns to display', df.columns)
    st.dataframe(df[selected_columns] if selected_columns else df)

    st.write("Filter the Data:")
    filter_column = st.selectbox('Select column to filter by', df.columns)
    filter_value = st.text_input('Enter filter value')
    if filter_value:
        filtered_data = df[df[filter_column].astype(str).str.contains(filter_value, case=False)]
        st.write(filtered_data)
        
        # Download option
        csv_data = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download Filtered Data as CSV', data=csv_data, file_name='filtered_data.csv', mime='text/csv')

# Data Insights Section
elif option == 'Data Insights':
    st.title('🧠 Data Insights')
    st.write("Gain insights into the data using various metrics.")
    st.write("Total Unique Values per Column:")
    st.write(df.nunique())
    
    st.write("Top 5 Frequent Values for Each Column:")
    for col in df.columns:
        st.write(f"{col}: {df[col].value_counts().head(5)}")

# Patient Outcome Prediction Section
elif option == 'Patient Outcome Prediction':
    st.title('🤖 Patient Outcome Prediction')
    
    # Load the pre-trained model
    model = load_model()
    
    if model is not None:
        st.success("✅ Pre-trained model loaded successfully!")
        
        # Define class descriptions
        class_descriptions = {
            0: "Patient recovered and went home",
            1: "Patient transferred to another hospital",
            2: "Patient moved to rehab facility",
            3: "Patient left against medical advice",
            4: "Patient deceased or serious outcome"
        }
        
        # Display target class distribution if target column exists
        target_column = 'target'
        if target_column in df.columns:
            st.subheader("Target Class Distribution")
            target_counts = df[target_column].value_counts().reset_index()
            target_counts.columns = ['Class', 'Count']
            target_counts['Description'] = target_counts['Class'].map(class_descriptions)
            st.write(target_counts)
            
            fig = px.pie(target_counts, values='Count', names='Description', title='Target Class Distribution')
            st.plotly_chart(fig)
        
        # Prediction interface
        st.subheader("Make Predictions")
        st.write("Enter values for the features to predict the patient outcome:")
        
        # Create a more interactive UI for prediction with all input values
        col1, col2, col3 = st.columns(3)
        
        # Create input fields for all required features
        input_values = {}
        
        with col1:
            input_values['age'] = st.number_input("Age", min_value=0, max_value=120, value=65)
            input_values['gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            input_values['previous_hospitalizations'] = st.number_input("Previous Hospitalizations", min_value=0, value=1)
            input_values['heart_rate'] = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)
            input_values['respiratory_rate'] = st.number_input("Respiratory Rate", min_value=5, max_value=60, value=18)
            input_values['blood_pressure_sys'] = st.number_input("Blood Pressure (Systolic)", min_value=50, max_value=250, value=120)
            input_values['blood_pressure_dia'] = st.number_input("Blood Pressure (Diastolic)", min_value=30, max_value=150, value=80)
            input_values['temperature'] = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
            input_values['wbc_count'] = st.number_input("WBC Count", min_value=0.0, max_value=50.0, value=8.0, step=0.1)
            input_values['creatinine'] = st.number_input("Creatinine", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        with col2:
            input_values['bilirubin'] = st.number_input("Bilirubin", min_value=0.1, max_value=30.0, value=1.0, step=0.1)
            input_values['glucose'] = st.number_input("Glucose", min_value=40, max_value=500, value=120)
            input_values['bun'] = st.number_input("BUN", min_value=5, max_value=150, value=20)
            input_values['pH'] = st.number_input("pH", min_value=6.8, max_value=7.8, value=7.4, step=0.01)
            input_values['pao2'] = st.number_input("PaO2", min_value=40, max_value=300, value=95)
            input_values['pco2'] = st.number_input("PCO2", min_value=20, max_value=100, value=40)
            input_values['fio2'] = st.number_input("FiO2", min_value=0.21, max_value=1.0, value=0.21, step=0.01)
            input_values['gcs'] = st.slider("GCS Score", 3, 15, 15)
            input_values['comorbidity_index'] = st.slider("Comorbidity Index", 0, 10, 2)
            input_values['admission_source'] = st.selectbox("Admission Source", [0, 1, 2, 3], 
                                                          format_func=lambda x: ["Emergency", "OPD", "Transfer", "Other"][x])
        
        with col3:
            input_values['elective_surgery'] = st.selectbox("Elective Surgery", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            input_values['num_medications'] = st.number_input("Number of Medications", min_value=0, value=5)
            input_values['charlson_comorbidity_index'] = st.slider("Charlson Comorbidity Index", 0, 15, 3)
            input_values['ews_score'] = st.slider("EWS Score", 0, 20, 3)
            input_values['severity_score'] = st.slider("Severity Score", 0, 10, 3)
            input_values['bed_occupancy_rate'] = st.slider("Bed Occupancy Rate (%)", 50, 100, 85)
            input_values['staff_to_patient_ratio'] = st.slider("Staff to Patient Ratio", 0.1, 2.0, 0.5, step=0.1)
            input_values['past_icu_admissions'] = st.number_input("Past ICU Admissions", min_value=0, value=0)
            input_values['previous_surgery'] = st.selectbox("Previous Surgery", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            input_values['high_risk_treatment'] = st.selectbox("High Risk Treatment", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            input_values['discharge_support'] = st.selectbox("Discharge Support", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        if st.button("Predict Outcome"):
            # Define input columns (must match your model's expected input features)
            input_columns = [
                'age', 'gender', 'previous_hospitalizations', 'heart_rate',
                'respiratory_rate', 'blood_pressure_sys', 'blood_pressure_dia',
                'temperature', 'wbc_count', 'creatinine', 'bilirubin', 'glucose', 'bun',
                'pH', 'pao2', 'pco2', 'fio2', 'gcs', 'comorbidity_index',
                'admission_source', 'elective_surgery', 'num_medications',
                'charlson_comorbidity_index', 'ews_score', 'severity_score',
                'bed_occupancy_rate', 'staff_to_patient_ratio', 'past_icu_admissions',
                'previous_surgery', 'high_risk_treatment', 'discharge_support'
            ]
            
            # Create a sample input for prediction (using a template from your dataset)
            if len(df) > 0:
                sample_input = pd.DataFrame([{col: 0 for col in input_columns}])
                
                # Update with user inputs
                for feature, value in input_values.items():
                    if feature in sample_input.columns:
                        sample_input[feature] = value
                
                # Make prediction
                try:
                    prediction = model.predict(sample_input)[0]
                    prediction_proba = model.predict_proba(sample_input)[0]
                    
                    # Display prediction
                    st.subheader("Prediction Result")
                    st.write(f"Predicted Class: {prediction} - {class_descriptions.get(prediction, 'Unknown')}")
                    
                    # Display probability for each class
                    st.write("Prediction Probabilities:")
                    proba_df = pd.DataFrame({
                        'Class': [class_descriptions.get(i, f"Class {i}") for i in range(len(prediction_proba))],
                        'Probability': prediction_proba
                    })
                    fig = px.bar(proba_df, x='Class', y='Probability', title='Prediction Probabilities')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.error("Dataset is empty, cannot create input template.")
    else:
        st.error("Failed to load model. Please check if 'xgboost_patient_model.json' exists in the current directory.")

st.sidebar.write("🚀 Created with Streamlit")
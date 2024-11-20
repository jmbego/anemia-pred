import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Helper functions
def convert_age_to_group(age):
    if 15 <= age <= 19:
        return '15-19'
    elif 20 <= age <= 24:
        return '20-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 34:
        return '30-34'
    elif 35 <= age <= 39:
        return '35-39'
    elif 40 <= age <= 44:
        return '40-44'
    elif 45 <= age <= 49:
        return '45-49'
    else:
        return 'unknown'

def get_recommendation(anemia_level):
    recommendations = {
        "No Anemia": "Maintain a balanced diet rich in iron, vitamin B12, and folic acid. Consider regular checkups if you're at risk.",
        "Mild Anemia": "Increase iron-rich foods like spinach, red meat, and lentils. Consider iron supplements if needed, after consulting a healthcare provider.",
        "Moderate Anemia": "Increase iron intake, and consult a healthcare provider to discuss possible iron or vitamin supplementation.",
        "Severe Anemia": "Seek immediate medical consultation to identify the underlying cause and discuss treatment options like supplements or other interventions.",
    }
    return recommendations.get(anemia_level, "No specific recommendation available.")

# Load the pickled encoder, pipeline, and model
with open('anemia_model.pkl', 'rb') as f:
    encoder, pipeline, model = pickle.load(f)

# Streamlit App
st.title("Anemia Prediction App")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "General Info", "Factors", "Prediction"])

if page == "Home":
    st.header("Welcome to the Anemia Prediction App")
    st.write("This app helps predict anemia levels based on socio-economic factors and provides tailored recommendations.")

elif page == "General Info":
    st.header("General Information")
    st.write("Here, you can provide general information about anemia, its causes, and its impact.")

elif page == "Factors":
    st.header("Factors Affecting Anemia")
    st.write("Explore the socio-economic and health factors that contribute to anemia.")

elif page == "Prediction":
    st.header("Predict Anemia Levels")
    st.write("Fill out the form below to get a prediction.")

    # User Input Form
    with st.form("prediction_form"):
        births_last_5y = st.selectbox("Births in last five years", ["Yes", "No"])
        age_first_birth = st.number_input("Age of respondent at 1st birth", min_value=12, max_value=50)
        hemoglobin_level = st.number_input("Hemoglobin level", min_value=5.0, max_value=20.0, step=0.1)
        age = st.number_input("Age", min_value=15, max_value=49)
        area_type = st.selectbox("Residence", ["Urban", "Rural"])
        education_level = st.selectbox("Highest educational level", ["No education", "Primary", "Secondary", "Higher"])
        wealth = st.selectbox("Wealth index", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
        mosquito_net = st.selectbox("Have mosquito net", ["Yes", "No"])
        marital_status = st.selectbox("Marital status", ["Never married", "Married", "Divorced", "Widowed"])
        living_with_spouse = st.selectbox("Residing with partner", ["Yes", "No"])
        had_fever = st.selectbox("Had fever in last two weeks", ["Yes", "No"])
        taking_meds = st.selectbox("Taking iron medication", ["Yes", "No"])
        
        submitted = st.form_submit_button("Predict")

        if submitted:
            # Map inputs to DataFrame
            input_data = {
                'Births_last_5y': births_last_5y,
                'Age_first_birth': age_first_birth,
                'Hemoglobin_level': hemoglobin_level,
                'Age_group': convert_age_to_group(age),
                'Area_Type': area_type,
                'Education_level': education_level,
                'Wealth': wealth,
                'Mosquito_net': mosquito_net,
                'Marital_status': marital_status,
                'Living_with_spouse': living_with_spouse,
                'Had_fever': had_fever,
                'Taking_meds': taking_meds
            }
            
            input_df = pd.DataFrame([input_data])

            # Preprocess and predict
            processed_input = pipeline.transform(input_df)
            prediction = model.predict(processed_input)

            # Interpret prediction
            prediction_map = {
                "Not anemic": "No Anemia",
                "Mild": "Mild Anemia",
                "Moderate": "Moderate Anemia",
                "Severe": "Severe Anemia"
            }
            prediction_text = prediction_map.get(prediction[0])
            recommendation_text = get_recommendation(prediction_text)

            # Display results
            st.subheader("Results")
            st.write(f"**Predicted Anemia Level:** {prediction_text}")
            st.write(f"**Recommendation:** {recommendation_text}")

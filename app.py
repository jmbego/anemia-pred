import streamlit as st
import pickle
import pandas as pd

# Load the pickled encoder, pipeline, and model
with open('anemia_model.pkl', 'rb') as f:
    encoder, pipeline, model = pickle.load(f)


# Utility functions
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


# Map prediction to human-readable text
prediction_map = {
    "Not anemic": 'No Anemia',
    "Mild": 'Mild Anemia',
    "Moderate": 'Moderate Anemia',
    "Severe": 'Severe Anemia'
}

# Streamlit App Layout
st.title("Anemia Prediction App")

st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age", min_value=15, max_value=49, value=25, step=1)
births_last_5y = st.sidebar.selectbox("Births in last 5 years", ["Yes", "No"])
age_first_birth = st.sidebar.number_input("Age of Respondent at 1st Birth", min_value=10, max_value=49, value=20)
hemoglobin_level = st.sidebar.number_input("Hemoglobin Level", min_value=5.0, max_value=20.0, value=12.5, step=0.1)
residence = st.sidebar.selectbox("Residence", ["Urban", "Rural"])
education = st.sidebar.selectbox("Highest Educational Level", ["None", "Primary", "Secondary", "Higher"])
wealth = st.sidebar.selectbox("Wealth Index", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
mosquito_net = st.sidebar.selectbox("Have Mosquito Net", ["Yes", "No"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Widowed", "Divorced"])
living_with_spouse = st.sidebar.selectbox("Residing with Partner", ["Yes", "No"])
had_fever = st.sidebar.selectbox("Had Fever in Last Two Weeks", ["Yes", "No"])
taking_meds = st.sidebar.selectbox("Taking Iron Medication", ["Yes", "No"])

if st.sidebar.button("Predict"):
    # Map input to feature names
    input_data = {
        'Births_last_5y': births_last_5y,
        'Age_first_birth': age_first_birth,
        'Hemoglobin_level': hemoglobin_level,
        'Age_group': convert_age_to_group(age),
        'Area_Type': residence,
        'Education_level': education,
        'Wealth': wealth,
        'Mosquito_net': mosquito_net,
        'Marital_status': marital_status,
        'Living_with_spouse': living_with_spouse,
        'Had_fever': had_fever,
        'Taking_meds': taking_meds
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input data
    processed_input = pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_input)[0]
    prediction_text = prediction_map.get(prediction, "Unknown")

    # Get recommendation
    recommendation_text = get_recommendation(prediction_text)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Anemia Level:** {prediction_text}")
    st.write(f"**Recommendation:** {recommendation_text}")

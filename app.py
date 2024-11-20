import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests  # For making HTTP requests

# Define functions for data processing, prediction, and recommendation
def convert_age_to_group(age):
    if 15 <= age <= 19:
        return '15-19'
    elif 20 <= age <= 24:
        return '20-24'
    # ... define remaining age groups ...
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

def make_prediction(data):
    # Load pickled encoder, pipeline, and model (assuming these are defined outside the function)
    with open('anemia_model.pkl', 'rb') as f:
        encoder, pipeline, model = pickle.load(f)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Preprocess the input data using the pipeline
    processed_input = pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_input)

    # Convert prediction to string and map to human-readable format
    prediction_text = list(prediction)[0]
    prediction_map = {
        "Not anemic": 'No Anemia',
        "Mild": 'Mild Anemia',
        "Moderate": 'Moderate Anemia',
        "Severe": 'Severe Anemia'
    }
    prediction_text = prediction_map.get(prediction_text)

    # Get recommendation based on the anemia level
    recommendation_text = get_recommendation(prediction_text)

    return prediction_text, recommendation_text

# Streamlit app layout
st.title('Anemia Prediction App')

# User input fields
age = st.number_input('Age:', min_value=15)
births_last_five_years = st.number_input('Births in last five years:')
# ... add other input fields based on your form ...

# Button to trigger prediction
if st.button('Predict Anemia Level'):
    # Prepare user input data
    data = {
        'Births_last_5y': births_last_five_years,
        'Age_group': convert_age_to_group(age),
        # ... map other user input fields to a dictionary ...
    }

    # Send a POST request to the Flask API endpoint (replace with your Flask app URL)
    response = requests.post('http://localhost:5000/predict', json=data)
    prediction_data = response.json()

    # Display prediction and recommendation
    st.write(f"Predicted Anemia Level: {prediction_data['prediction']}")
    st.write(f"Recommendation: {prediction_data['recommendation']}")

# Define a hidden Flask app (can be placed at the bottom of the code)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_anemia():
    data = request.get_json()
    prediction_text, recommendation_text = make_prediction(data)
    return jsonify({'prediction': prediction_text, 'recommendation': recommendation_text})

if __name__ == '__main__':
    st.balloons()  # Optional: Display Streamlit balloons on startup
    app.run(debug=True)  # Run the Flask app for API endpoint
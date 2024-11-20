from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)


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
    encoder,pipeline, model = pickle.load(f)


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the general information page
@app.route('/general_info')
def general_info():
    return render_template('general_info.html')

# Route for the factors affecting anemia page
@app.route('/factors')
def factors():
    return render_template('factors.html')

# Route to display the prediction form
@app.route('/predict')
def predict():
    return render_template('prediction.html')

# Route to handle form submission, make predictions, and render results with recommendations
@app.route('/predict', methods=['POST'])
def make_prediction():
    # Map user input to the feature order in the DataFrame
    input_data = {
        'Births_last_5y': request.form['Births in last five years'],
        'Age_first_birth': request.form['Age of respondent at 1st birth'],
        'Hemoglobin_level': request.form['Hemoglobin level'],
        'Age_group':  convert_age_to_group(int(request.form['Age'])),
        'Area_Type': request.form['Residence'],
        'Education_level': request.form['Highest educational level'],
        'Wealth': request.form['Wealth index'],
        'Mosquito_net': request.form['Have mosquito net'],
        'Marital_status': request.form['marital status'],
        'Living_with_spouse': request.form['Residing with partner'],
        'Had_fever': request.form['Had fever in last two weeks'],
        'Taking_meds': request.form['Taking iron medication']
    }



    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data using the pipeline
    processed_input = pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_input)

    # Convert prediction to string
    prediction_text = list(prediction)[0] 
     # Map the prediction output to a human-readable string
    prediction_map = {
        "Not anemic": 'No Anemia',
        "Mild": 'Mild Anemia',
        "Moderate": 'Moderate Anemia',
        "Severe": 'Severe Anemia'
    }

    # Get the mapped prediction text
    prediction_text = prediction_map.get(prediction_text)

    # Get the tailored recommendation based on the anemia level
    recommendation_text = get_recommendation(prediction_text)

    # Print statements for debugging
    #print("Prediction Text:", prediction_text)
    #print("Recommendation Text:", recommendation_text)

    # Render the results page with the prediction and recommendations
    return render_template('result.html', prediction_text=f'Predicted Anemia Level: {prediction_text}', recommendation_text=recommendation_text)

if __name__ == "__main__":
    port = os.getenv('PORT', 5000)  # Default to 5000 if no port is set
    app.run(debug=True, host='0.0.0.0', port=port)
   

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model and Encoders
model = joblib.load('decision_tree_model.pkl')
le_race = joblib.load('le_race.pkl')
le_sex = joblib.load('le_sex.pkl')
le_age = joblib.load('le_age.pkl')

@app.route('/')
def home():
    # Pass unique values to the template for dropdowns
    races = sorted(le_race.classes_)
    sexes = sorted(le_sex.classes_)
    ages = sorted(le_age.classes_)
    return render_template('index.html', races=races, sexes=sexes, ages=ages)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            race = request.form['race']
            sex = request.form['sex']
            age = request.form['age']
            year = int(request.form['year'])
            week = int(request.form['week'])

            # Encode input
            race_encoded = le_race.transform([race])[0]
            sex_encoded = le_sex.transform([sex])[0]
            age_encoded = le_age.transform([age])[0]

            # Create dataframe for prediction (must match training feature order)
            # Training cols: RaceEthnicity, Sex, AgeGroup, MMWRyear, MMWRweek
            features = pd.DataFrame([[race_encoded, sex_encoded, age_encoded, year, week]], 
                                    columns=['RaceEthnicity', 'Sex', 'AgeGroup', 'MMWRyear', 'MMWRweek'])

            # Predict
            prediction = model.predict(features)[0]
            result_text = f'Predicted Excess Deaths: {int(prediction)}'
            print(result_text)

            return render_template('index.html', 
                                   prediction_text=result_text,
                                   races=sorted(le_race.classes_),
                                   sexes=sorted(le_sex.classes_),
                                   ages=sorted(le_age.classes_))

        except Exception as e:
            return render_template('index.html', 
                                   prediction_text=f'Error: {str(e)}',
                                   races=sorted(le_race.classes_),
                                   sexes=sorted(le_sex.classes_),
                                   ages=sorted(le_age.classes_))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

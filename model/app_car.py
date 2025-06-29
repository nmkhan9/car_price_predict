from flask import Flask, request, render_template, jsonify
import pandas as pd
from joblib import load
import numpy as np

app = Flask(__name__)


ridge_model = load("E:\Pycode\Project_code\ML_git\Car_price_prediction\model\ridge_model_1.joblib")
ohe = load("E:\Pycode\Project_code\ML_git\Car_price_prediction\model\onehot_encoder.pkl")
scaler = load("E:\Pycode\Project_code\ML_git\Car_price_prediction\model\scaler.pkl")


categorical_cols = ['Body_Type', 'Origin', 'Province', 'District', 'Transmission', 'Fuel_Type', 'Brand']
numerical_cols = ['age', 'mileage_num']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = {
            'Body_Type': request.form['Body_Type'],
            'Origin': request.form['Origin'],
            'Province': request.form['Province'],
            'District': request.form['District'],
            'Transmission': request.form['Transmission'],
            'Fuel_Type': request.form['Fuel_Type'],
            'Brand': request.form['Brand'],
            'age': float(request.form['age']),
            'mileage_num': float(request.form['mileage_num'])
        }


        input_df = pd.DataFrame([data])


        encoded_categorical = ohe.transform(input_df[categorical_cols])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=ohe.get_feature_names_out(categorical_cols)
        )

        scaled_numerical = scaler.transform(input_df[numerical_cols])
        scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_cols)

        processed_data = pd.concat([encoded_categorical_df, scaled_numerical_df], axis=1)

        prediction = ridge_model.predict(processed_data)[0]

        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
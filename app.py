from flask import Flask, request, render_template
import pickle
import numpy as np
import bz2

app = Flask(__name__)

# Load the saved model
model_path = '../path_to/insurance_prediction_final.pkl.bz2'
with bz2.BZ2File(model_path, 'rb') as file:
    model = pickle.load(file)

# Feature Order
feature_order = [
    'age', 'gender', 'marital_status', 'income', 'driving_history', 
    'vehicle_age', 'vehicle_value', 'coverage_type', 'num_claims',
    'claim_amount', 'policy_duration', 'driving_experience',
    'occupation_Professional', 'occupation_Retired', 'occupation_Student',
    'education_High_School', 'education_Master', 'education_PhD',
    'location_Suburban', 'location_Urban', 'vehicle_type_Sedan',
    'vehicle_type_Sports Car', 'vehicle_type_Truck'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and encode categorical variables as required
        form_values = request.form

        # Process numeric features
        age = float(form_values['age'])
        income = float(form_values['income'])
        vehicle_age = float(form_values['vehicle_age'])
        vehicle_value = float(form_values['vehicle_value'])
        claim_amount = float(form_values['claim_amount'])
        policy_duration = float(form_values['policy_duration'])
        driving_experience = float(form_values['driving_experience'])
        num_claims = float(form_values['num_claims'])

        # Process encoded categorical features
        gender = int(form_values['gender'])  # Male (1), Female (0)
        marital_status = int(form_values['marital_status'])  # Single (0), Divorced (1), Married (2)
        driving_history = int(form_values['driving_history'])  # Clean (0), Violations (1), Accidents (2)
        coverage_type = int(form_values['coverage_type'])  # Standard (0), Basic (1), Premium (2)

        # One-hot encoded features (set to 1 or 0 based on user input)
        occupation_Professional = int('occupation_Professional' in form_values)
        occupation_Retired = int('occupation_Retired' in form_values)
        occupation_Student = int('occupation_Student' in form_values)
        education_High_School = int('education_High_School' in form_values)
        education_Master = int('education_Master' in form_values)
        education_PhD = int('education_PhD' in form_values)
        location_Suburban = int('location_Suburban' in form_values)
        location_Urban = int('location_Urban' in form_values)
        vehicle_type_Sedan = int('vehicle_type_Sedan' in form_values)
        vehicle_type_Sports_Car = int('vehicle_type_Sports Car' in form_values)
        vehicle_type_Truck = int('vehicle_type_Truck' in form_values)

        # Combine all features into the input array
        input_features = [
            age, gender, marital_status, income, driving_history,
            vehicle_age, vehicle_value, coverage_type, num_claims,
            claim_amount, policy_duration, driving_experience,
            occupation_Professional, occupation_Retired, occupation_Student,
            education_High_School, education_Master, education_PhD,
            location_Suburban, location_Urban, vehicle_type_Sedan,
            vehicle_type_Sports_Car, vehicle_type_Truck
        ]
        input_array = np.array(input_features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_array)

        # Return the result
        return render_template(
            'index.html', 
            prediction_text=f'Predicted Premium: ${prediction[0]:.2f}'
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)

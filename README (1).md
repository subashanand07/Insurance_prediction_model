# Insurance Prediction Deployment

This project demonstrates the deployment of a machine learning model forpredicting insurance premiums usinga Flask-based web application.
The pre-trained model, saved using `pickle`, enables real-time premium predictions through an interactive web interface.

---

## Project Structure
insurance-premium-deployment/
app.py               # Flask app
index.html       # Web interface for input
styles.css       # Optional styling
insurance_premium_model.pkl  # Your saved ML prediction model

## Setup and Usage

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install flask numpy pickle5

  
### Steps to Run
Ensure the model file is available: Make sure the insurance_premium_model.pkl file is located in the model directory.

1. Run the Flask application:
`python app.py`

2. Access the application: Open your web browser and navigate to: `http://127.0.0.1:5000/`

3. Input features and predict: Enter the required features in the form and submit to see the predicted insurance premium

### Features
User-Friendly Web Interface: Allows users to input feature values for prediction through an intuitive HTML form.
Real-Time Predictions: Processes data quickly to provide instant results.
Pre-Trained Model: The ML model is pre-trained and saved using pickle for reuse.

### Future Enhancements
Add input validation for robust data handling.
Enhance UI/UX with better styling and usability.
Deploy the application on cloud platforms like Heroku or AWS.

### License
This project is licensed under the MIT License.


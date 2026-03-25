# Smart Agriculture System (Crop + Fertilizer + Yield Prediction)
## Project Overview

This project is a Machine Learning-based web application that helps farmers and users make better agricultural decisions.

It provides:

🌱 Crop Recommendation
🌿 Fertilizer Suggestion
📊 Crop Yield Prediction

The system uses multiple trained ML models to analyze soil nutrients, weather conditions, and other inputs.

## Features
* Predicts best crop based on soil & environment
* Suggests suitable fertilizers
* Estimates crop yield
* Simple web interface using Flask
* Fast predictions using pre-trained models

## Tech Stack

Frontend: HTML, CSS
Backend: Python (Flask)
Machine Learning: Scikit-learn
Libraries: NumPy, Pandas, Joblib

## Project Structure
crop-project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
    * crop_model.pkl
    * fertilizer_model.pkl
    * dtr.pkl
    * minmaxscalar.pkl
    * standscalar.pkl
    * preprocessor.pkl
    * crop_encoder (1).pkl
    * fertilizer_encoder (1).pkl
    * soil_encoder (1).pkl
├── data/
    * Crop_recommendation.csv
    * Fertilizer Prediction.csv
    * yield_df.csv
├── notebooks/
    * crop recommendation.ipynb
    * CropYield-Prediction.ipynb
├── app.py
├── static/
└── templates/

## Installation & Setup

🔹 Step 1: Clone Repository
git clone https://github.com/your-username/crop-project.git
cd crop-project
🔹 Step 2: Install Dependencies
pip install -r requirements.txt
🔹 Step 3: Run the Application
python app.py
🔹 Step 4: Open in Browser
http://127.0.0.1:5000/

## Input Parameters

The system may require inputs like:
Nitrogen (N), Phosphorus (P), Potassium (K)
Temperature
Humidity
pH value
Rainfall

## Output

Recommended Crop 🌱
Fertilizer Suggestion 🌿
Predicted Yield 📈


## Machine Learning Models

Crop Recommendation Model
Fertilizer Prediction Model
Crop Yield Prediction Model

Models are trained using supervised learning algorithms.

##  Limitations
Predictions depend on dataset quality
Not a replacement for expert agricultural advice
  *  Future Improvements
  * Deploy on cloud (Heroku / Render)
  *  Mobile-friendly UI
  * Real-time weather API integration
  * Deep learning models

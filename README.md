# ChargeWise AI - EV Charging Demand Forecasting

This repository contains the Machine Learning pipeline and Streamlit web application for forecasting EV charging demand. 
ChargeWise AI uses advanced Machine Learning models, primarily a hyperparameter-tuned Random Forest Regressor, to predict the demand at charging stations based on real-time and historical factors.

## Project Structure
- `models/`: Contains the saved `random_forest_ev_model.pkl` and `model_features.pkl`.
- `outputs/`: Contains the output metrics for all evaluated models `model_results.csv`.
- `notebook/`: Contains the exploratory data analysis and model training code `EV_Demand_Forecasting.ipynb`.
- `app.py`: Main Streamlit web application.
- `requirements.txt`: Python dependencies.

## Features
- **Predict Demand**: Make real-time predictions of charging demand (Low/Medium/High) and get operational recommendations.
- **Model Performance**: View and compare the metrics of various evaluated machine learning models.
- **About Project**: A detailed guide on the data science process behind this tool, including feature engineering, cross-validation, and avoiding data leakage.

## How to Run Locally
1. Clone this repository or open the project folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Access the web app at `http://localhost:8501`.

## Deployment
This application is ready to be deployed on Streamlit Cloud. Simply connect your GitHub repository and point to `app.py`.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(page_title="ChargeWise AI", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00ccaa;
        box-shadow: 0 4px 15px rgba(0, 255, 204, 0.4);
    }
    .card {
        background: #1e2127;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #888;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        border-top: 1px solid #333;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Load model and features safely
@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "random_forest_ev_model.pkl")
    features_path = os.path.join("models", "model_features.pkl")
    results_path = os.path.join("outputs", "model_results.csv")
    
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    results = pd.read_csv(results_path)
    return model, features, results

model, model_features, model_results = load_assets()

# Sidebar Navigation
st.sidebar.title("ChargeWise AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Predict Demand", "Model Performance", "About Project"])

if page == "Home":
    st.title("Welcome to ChargeWise AI")
    st.markdown("### EV Charging Demand Forecasting")
    
    # Professional metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Final Model", "Random Forest")
    with m2:
        st.metric("R² Score", "0.9875")
    with m3:
        st.metric("RMSE", "3.11")
        
    st.markdown("""
    <div class="card" style="margin-top: 20px;">
        <h4>Intelligent Forecasting for EV Infrastructure</h4>
        <p>ChargeWise AI is a professional machine learning dashboard designed to forecast EV charging demand. By leveraging robust machine learning models and analyzing various charging station and session conditions, it predicts demand levels with high accuracy, enabling better resource optimization and planning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Use the sidebar navigation to predict demand, view model performance, or read about the project's methodology.")

elif page == "Predict Demand":
    st.title("Predict Charging Demand")
    st.markdown("Enter station, vehicle, environment, and time details to estimate EV charging demand.")
    
    # Forms and Inputs
    with st.expander("Section 1: Basic Station Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            station_id = st.selectbox("Station ID", [f"ST{i:03d}" for i in range(1, 21)])
            location_type = st.selectbox("Location Type", ["Urban", "Highway", "Suburban"])
        with col2:
            queue_length = st.number_input("Queue Length", min_value=0, value=2)
            waiting_time = st.number_input("Waiting Time (mins)", min_value=0, value=10)

    with st.expander("Section 2: Vehicle and Charging Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            vehicle_type = st.selectbox("Vehicle Type", ["Car", "Two-Wheeler", "Bus", "Truck"])
            battery_capacity_kWh = st.number_input("Battery Capacity (kWh)", min_value=1.0, value=50.0)
            initial_soc = st.slider("Initial State of Charge (%)", 0.0, 100.0, 20.0)
        with col2:
            charging_power_kW = st.number_input("Charging Power (kW)", min_value=1.0, value=50.0)
            charging_priority = st.selectbox("Charging Priority", ["High", "Medium", "Low"])

    with st.expander("Section 3: Environment and Time Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            traffic_density = st.selectbox("Traffic Density", ["High", "Medium", "Low"])
            weather_condition = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy"])
            day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        with col2:
            time_slot = st.selectbox("Time Slot", ["Morning", "Afternoon", "Evening", "Night", "Peak", "Off-Peak"])
            month = st.number_input("Month", min_value=1, max_value=12, value=6)
            day = st.number_input("Day", min_value=1, max_value=31, value=15)
        with col3:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
            minute = st.number_input("Minute", min_value=0, max_value=59, value=30)
            is_weekend_input = st.selectbox("Is Weekend?", ["No", "Yes"])
            is_weekend = 1 if is_weekend_input == "Yes" else 0

    with st.expander("Section 4: Energy and Pricing", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            electricity_price = st.number_input("Electricity Price ($/kWh)", min_value=0.0, value=0.15, step=0.01)
        with col2:
            renewable_energy_ratio = st.slider("Renewable Energy Ratio", 0.0, 1.0, 0.5)

    st.markdown("---")
    
    if st.button("Predict Demand", type="primary", use_container_width=True):
        # 1. Gather User Inputs
        input_data = {
            'waiting_time': waiting_time,
            'battery_capacity_kWh': battery_capacity_kWh,
            'initial_soc': initial_soc,
            'charging_power_kW': charging_power_kW,
            'queue_length': queue_length,
            'electricity_price': electricity_price,
            'renewable_energy_ratio': renewable_energy_ratio,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'is_weekend': is_weekend,
            
            'station_id': station_id,
            'location_type': location_type,
            'vehicle_type': vehicle_type,
            'traffic_density': traffic_density,
            'weather_condition': weather_condition,
            'day_of_week': day_of_week,
            'time_slot': time_slot,
            'charging_priority': charging_priority
        }
        
        df = pd.DataFrame([input_data])
        
        # 2. Feature Engineering
        # Calculate derived features required by the model
        df['waiting_per_queue'] = df['waiting_time'] / (df['queue_length'] + 1)
        
        # Proxy station load using charging power & queue
        station_load = charging_power_kW * queue_length 
        df['load_per_queue'] = station_load / (df['queue_length'] + 1)
        
        # Proxy energy consumed
        energy_consumed_kWh = battery_capacity_kWh * ((100 - initial_soc) / 100)
        energy_consumed_kWh = max(energy_consumed_kWh, 1.0) # Avoid division by zero
        df['price_per_kWh'] = df['electricity_price'] / energy_consumed_kWh
        
        df['charging_efficiency'] = df['charging_power_kW'] / df['battery_capacity_kWh']
        df['power_to_wait_ratio'] = df['charging_power_kW'] / (df['waiting_time'] + 1)
        
        # 3. One-hot Encoding
        categorical_cols = ['station_id', 'location_type', 'vehicle_type', 'traffic_density', 
                           'weather_condition', 'day_of_week', 'time_slot', 'charging_priority']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # 4. Align with Model Features
        # Ensure all columns from training are present
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Reorder columns to match exactly
        X_predict = df_encoded[model_features]
        
        # 5. Make Prediction
        prediction = model.predict(X_predict)[0]
        
        # 6. Determine Demand Level and Recommendation
        if prediction < 20:
            demand_level = "Low"
            color = "#28a745" # Green
            recommendation = "Optimal time for charging. Operators could offer discounts to attract more users."
        elif prediction < 50:
            demand_level = "Medium"
            color = "#ffc107" # Orange
            recommendation = "Standard demand. Monitor queue lengths and ensure all chargers are operational."
        else:
            demand_level = "High"
            color = "#dc3545" # Red
            recommendation = "High congestion expected. Consider dynamic pricing or load balancing across nearby stations."
            
        # Display Results
        st.markdown("### Prediction Results")
        
        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div class="card" style="flex: 1; text-align: center; border-bottom: 4px solid #00ffcc;">
                <h4 style="margin: 0; color: #888;">Predicted Demand</h4>
                <h2 style="margin: 10px 0; color: #00ffcc;">{prediction:.2f} kW</h2>
            </div>
            <div class="card" style="flex: 1; text-align: center; border-bottom: 4px solid {color};">
                <h4 style="margin: 0; color: #888;">Demand Category</h4>
                <h2 style="margin: 10px 0; color: {color};">{demand_level}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {color};">
            <strong>Operational Recommendation:</strong><br>
            {recommendation}
        </div>
        """, unsafe_allow_html=True)

elif page == "Model Performance":
    st.title("Model Performance")
    
    st.markdown("### Project Statistics")
    st.markdown("""
    <div style="display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
        <div class="card" style="flex: 1; min-width: 150px; text-align: center; padding: 15px; margin-bottom: 0;">
            <h6 style="color: #888; margin-bottom: 5px;">Total Records</h6>
            <h3 style="color: #00ffcc; margin: 0;">8,354</h3>
        </div>
        <div class="card" style="flex: 1; min-width: 150px; text-align: center; padding: 15px; margin-bottom: 0;">
            <h6 style="color: #888; margin-bottom: 5px;">Original Features</h6>
            <h3 style="color: #00ffcc; margin: 0;">21</h3>
        </div>
        <div class="card" style="flex: 1; min-width: 150px; text-align: center; padding: 15px; margin-bottom: 0;">
            <h6 style="color: #888; margin-bottom: 5px;">Custom Features Created</h6>
            <h3 style="color: #00ffcc; margin: 0;">8</h3>
        </div>
        <div class="card" style="flex: 1; min-width: 150px; text-align: center; padding: 15px; margin-bottom: 0;">
            <h6 style="color: #888; margin-bottom: 5px;">Final Features</h6>
            <h3 style="color: #00ffcc; margin: 0;">55</h3>
        </div>
        <div class="card" style="flex: 1; min-width: 150px; text-align: center; padding: 15px; margin-bottom: 0;">
            <h6 style="color: #888; margin-bottom: 5px;">Target Variable</h6>
            <h3 style="color: #00ffcc; margin: 0;">Charging Demand</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Below are the evaluation metrics for the different Machine Learning algorithms tested during the project.")
    
    # Sort by R2 Score descending
    model_results_sorted = model_results.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    
    # Display the results dataframe beautifully
    st.dataframe(
        model_results_sorted.style.highlight_max(subset=['R2 Score'], color='#1f77b4')
        .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#1f77b4'),
        use_container_width=True
    )
    
    st.info("""
    **Model Comparison Summary**
    
    Five machine learning algorithms were evaluated using MAE, MSE, RMSE, and R² Score.
    
    Random Forest achieved the strongest balance between predictive accuracy and generalization performance, outperforming both linear and tree-based alternatives.
    """)

    st.markdown("""
    ### Metrics Explained
    - **MAE**: Average absolute difference between predicted and actual demand. (Lower is better)
    - **RMSE**: Penalizes larger errors more heavily. (Lower is better)
    - **R² Score**: Proportion of variance predictable from features. (Closer to 1.0 is better)
    
    ### Final Selected Model: Random Forest Regressor
    Random Forest was selected as the production model for the following reasons:
    - **Highest R² Score**: 0.9875
    - **Lowest RMSE**: 3.11
    - **Lowest MAE**: 2.61
    - **Better Generalization**: It demonstrated better generalization than the tuned version.
    
    ### Feature Importance
    
    **Feature Importance Analysis**
    
    Feature importance measures the relative contribution of each feature to the model's prediction process.
    
    Higher importance values indicate features that have a stronger influence on forecasting charging demand.
    """)

    # Extract feature importance from the model
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = model_features
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
        
        st.bar_chart(importance_df.set_index("Feature"))
        
    st.markdown("""
    ### Top Features
    | Feature | Importance |
    |---------|------------|
    | `time_slot_Peak` | 0.756 |
    | `renewable_energy_ratio` | 0.067 |
    | `load_per_queue` | 0.065 |
    | `queue_length` | 0.046 |
    | `traffic_density_Low` | 0.033 |

    ### Detailed Insights
    
    **Time Slot (Peak Hours)**
    - Most influential feature (~75% importance)
    - Indicates charging demand is strongly driven by peak usage behavior.
    
    **Load Per Queue**
    - Represents charging station congestion.
    - Higher queue pressure increases demand.
    
    **Renewable Energy Ratio**
    - Suggests energy availability influences charging activity.
    
    **Queue Length**
    - Captures station utilization and waiting conditions.
    """)
    
    st.info("""
    **Conclusion**
    
    The analysis indicates that charging demand is influenced more by usage behavior, station congestion, and infrastructure conditions than by individual vehicle characteristics.
    
    Peak charging periods emerged as the dominant demand driver, accounting for approximately 75% of the model's predictive signal.
    """)
    
    st.markdown("""
    ### Model Reliability & Validation
    
    **Cross Validation Scores**
    - 0.9866
    - 0.9870
    - 0.9862
    - 0.9858
    - 0.9870
    
    **Mean Cross Validation R² Score: 0.9865**
    
    *The consistency of cross-validation scores demonstrates stable model behavior across multiple train-validation splits.*
    
    **Overfitting Analysis**
    - Train R²: 0.9981
    - Test R²: 0.9875
    
    The small performance gap indicates strong generalization and minimal overfitting.
    """, unsafe_allow_html=True)
    
    st.markdown("### Business Impact")
    
    st.success("""
    **Practical Applications**
    - EV Charging Demand Forecasting
    - Charging Station Capacity Planning
    - Queue and Congestion Management
    - Energy Distribution Optimization
    - Renewable Energy Utilization Planning
    - Smart Charging Infrastructure Expansion
    
    The system can assist charging network operators in making data-driven decisions regarding charger deployment, resource allocation, peak-hour management, and future infrastructure planning.
    """)
    
    st.markdown("### Project Highlights")
    
    ph1, ph2, ph3, ph4 = st.columns(4)
    ph1.info("✓ Explored 5+ EV-related datasets")
    ph2.info("✓ Engineered 8 custom domain-specific features")
    ph3.info("✓ Removed multiple target leakage variables")
    ph4.info("✓ Evaluated 5 machine learning algorithms")
    
    ph5, ph6, ph7, ph8 = st.columns(4)
    ph5.info("✓ Achieved R² Score of 0.9875")
    ph6.info("✓ Performed Cross Validation and Overfitting Analysis")
    ph7.info("✓ Implemented Feature Importance Interpretation")
    ph8.info("✓ Built and Deployed a Streamlit Application")

elif page == "About Project":
    st.title("About ChargeWise AI")
    
    st.markdown("""
    ### 1. Project Overview
    ChargeWise AI is an end-to-end machine learning system designed to forecast EV charging demand using station conditions, vehicle characteristics, environmental variables, and temporal patterns.
    
    <div class="card" style="border-left: 4px solid #00ffcc;">
        <strong>Business Objective:</strong>
        <ul>
            <li>Reduce waiting times</li>
            <li>Improve charger allocation</li>
            <li>Support infrastructure planning</li>
            <li>Optimize charging network operations</li>
        </ul>
    </div>
    
    ### 2. Dataset Exploration & Selection
    Multiple datasets were explored:
    - EV Charging Sessions Dataset
    - EV Charging Stations Dataset
    - EV Charging Patterns Dataset
    - Weather Dataset
    - Holiday Datasets
    
    All datasets were investigated, compared, and retained in the raw data directory, but only the most suitable dataset was selected based on quality, completeness, and forecasting relevance.

    ### 3. Data Cleaning & Preparation
    - Missing value analysis
    - Duplicate detection
    - Datetime processing
    - Data consistency validation
    - Removal of unnecessary attributes

    ### 4. Exploratory Data Analysis (EDA)
    - Demand distribution analysis
    - Correlation analysis
    - Vehicle type impact
    - Weather impact
    - Traffic density impact
    - Time slot impact
    - Feature relationship analysis
    
    **Key findings:**
    - Peak hours strongly influence demand
    - Traffic density significantly affects charging behavior
    - Environmental factors contribute to demand variation
    - Queue-related variables correlate with charging demand

    ### 5. Feature Engineering
    Custom features created:
    
    - **`waiting_per_queue`**: Normalizes waiting time using queue length.
    - **`load_per_queue`**: Distributes charging load across the queue.
    - **`price_per_kWh`**: Calculates effective electricity cost.
    - **`charging_efficiency`**: Measures relationship between charging power and battery capacity.
    - **`power_to_wait_ratio`**: Balances delivered power against waiting time.

    ### 6. Leakage Detection & Feature Selection
    Target leakage occurs when a model is trained with data it would not have at prediction time. Removing leakage was critical for realistic forecasting.
    
    Removed variables:
    - `final_soc`
    - `energy_consumed_kWh`
    - `charging_duration`
    - `station_load` derived variables

    ### 7. Model Development & Comparison
    Evaluated models:
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - XGBoost Regressor
    - CatBoost Regressor
    
    Metrics used: MAE, MSE, RMSE, R²

    ### 8. Hyperparameter Tuning & Validation
    - RandomizedSearchCV
    - Cross Validation
    - Generalization Testing
    - Overfitting Analysis
    
    The tuned Random Forest underperformed the default model and therefore the default model was retained.

    ### 9. Model Interpretation
    Feature Importance was analyzed to interpret model decisions.
    
    **Top Drivers:**
    - `time_slot_Peak`
    - `renewable_energy_ratio`
    - `load_per_queue`
    - `queue_length`
    
    Peak charging periods emerged as the strongest predictor.

    ### 10. Business Impact
    <div class="card" style="border-left: 4px solid #28a745;">
        <strong>Practical Applications:</strong>
        <ul>
            <li>EV demand forecasting</li>
            <li>Charging station capacity planning</li>
            <li>Queue management</li>
            <li>Energy optimization</li>
            <li>Renewable energy integration</li>
            <li>Smart charging infrastructure planning</li>
        </ul>
    </div>

    ### 11. Project Workflow
    <div style="text-align: center; padding: 20px; background-color: #1e2127; border-radius: 12px; margin-bottom: 20px; font-weight: bold; letter-spacing: 1px;">
        Dataset Selection <br>↓<br> Data Cleaning <br>↓<br> EDA <br>↓<br> Feature Engineering <br>↓<br> Feature Selection <br>↓<br> Model Training <br>↓<br> Validation <br>↓<br> Interpretation <br>↓<br> Deployment
    </div>
    
    ### 12. Key Achievements
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Explored 5+ EV-related datasets</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Engineered 8 custom features</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Removed target leakage variables</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Evaluated 5 ML algorithms</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Achieved R² Score of 0.9875</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Performed Cross Validation</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Conducted Overfitting Analysis</div>
        <div style="background-color: #1e2127; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #00ffcc;">✓ Built and deployed a Streamlit application</div>
    </div>

    ### 13. Deployment
    - Model exported using Joblib
    - Feature schema saved
    - Deployed using Streamlit
    - Supports real-time demand prediction
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built by Somiya Namdeo | Machine Learning Project | ChargeWise AI
</div>
""", unsafe_allow_html=True)

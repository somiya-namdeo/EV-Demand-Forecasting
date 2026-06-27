import streamlit as st
import pandas as pd
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
    st.markdown("""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div class="card" style="flex: 1; text-align: center; border-bottom: 4px solid #00ffcc;">
            <h4 style="margin: 0; color: #888;">Final Model</h4>
            <h2 style="margin: 10px 0; color: #00ffcc;">Random Forest</h2>
        </div>
        <div class="card" style="flex: 1; text-align: center; border-bottom: 4px solid #00ffcc;">
            <h4 style="margin: 0; color: #888;">R² Score</h4>
            <h2 style="margin: 10px 0; color: #00ffcc;">0.9875</h2>
        </div>
        <div class="card" style="flex: 1; text-align: center; border-bottom: 4px solid #00ffcc;">
            <h4 style="margin: 0; color: #888;">RMSE</h4>
            <h2 style="margin: 10px 0; color: #00ffcc;">3.11</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="card">
        <h4>Business Problem</h4>
        <p>EV charging stations often face congestion, under-utilization, inefficient charger allocation, and peak-hour demand spikes.</p>
        <p>ChargeWise AI forecasts charging demand using station conditions, vehicle characteristics, environmental factors, and temporal usage patterns to support smarter infrastructure planning and operational decision-making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h4>Intelligent Forecasting for EV Infrastructure</h4>
        <p>ChargeWise AI is a professional machine learning dashboard designed to forecast EV charging demand. By leveraging robust machine learning models and analyzing various charging station and session conditions, it predicts demand levels with high accuracy, enabling better resource optimization and planning.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Predict Demand":
    st.title("Predict Charging Demand")
    st.markdown("Enter station, vehicle, environment, and time details to estimate EV charging demand.")
    
    # Forms and Inputs
    with st.expander("Station Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            station_id = st.selectbox("Station ID", [f"ST{i:03d}" for i in range(1, 21)])
            location_type = st.selectbox("Location Type", ["Urban", "Highway", "Suburban"])
        with col2:
            queue_length = st.number_input("Queue Length", min_value=0, value=2)
            waiting_time = st.number_input("Waiting Time (mins)", min_value=0, value=10)

    with st.expander("Vehicle Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            vehicle_type = st.selectbox("Vehicle Type", ["Car", "Two-Wheeler", "Bus", "Truck"])
            battery_capacity_kWh = st.number_input("Battery Capacity (kWh)", min_value=1.0, value=50.0)
            initial_soc = st.slider("Initial State of Charge (%)", 0.0, 100.0, 20.0)
        with col2:
            charging_power_kW = st.number_input("Charging Power (kW)", min_value=1.0, value=50.0)
            charging_priority = st.selectbox("Charging Priority", ["High", "Medium", "Low"])

    with st.expander("Environmental Factors", expanded=True):
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

    with st.expander("Energy & Pricing", expanded=True):
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
    """)

    with st.expander("Feature Importance Interpretation", expanded=False):
        st.markdown("""
        **Feature Importance Analysis**
        
        Feature importance measures the relative contribution of each feature to the model's prediction process.
        Higher importance values indicate features that have a stronger influence on forecasting charging demand.
        """)
        
        # Extract feature importance from the model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = model_features
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            top5_df = importance_df.head(5)
            
            # Build Top Features card
            top_features_html = '<div class="card" style="margin-top: 15px;"><h4>Top Features</h4><table style="width:100%; text-align:left; border-collapse: collapse;"><tr><th style="border-bottom: 1px solid #333; padding: 8px; color: #888;">Feature</th><th style="border-bottom: 1px solid #333; padding: 8px; color: #888;">Importance</th></tr>'
            for _, row in top5_df.iterrows():
                top_features_html += f'<tr><td style="padding: 8px; border-bottom: 1px solid #222;"><code>{row["Feature"]}</code></td><td style="padding: 8px; border-bottom: 1px solid #222;">{row["Importance"]:.4f}</td></tr>'
            top_features_html += '</table></div>'
            
            st.markdown(top_features_html, unsafe_allow_html=True)
            
            st.markdown("#### Feature Importance Chart")
            st.bar_chart(importance_df.head(10).set_index("Feature"))
            
    with st.expander("Detailed Insights", expanded=False):
        st.markdown("""
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
        
        **Conclusion**
        
        The analysis indicates that charging demand is influenced more by usage behavior, station congestion, and infrastructure conditions than by individual vehicle characteristics.
        Peak charging periods emerged as the dominant demand driver, accounting for approximately 75% of the model's predictive signal.
        """)
        
    with st.expander("Cross Validation Analysis", expanded=False):
        st.markdown("""
        **Cross Validation Scores**
        - 0.9866
        - 0.9870
        - 0.9862
        - 0.9858
        - 0.9870
        
        **Mean Cross Validation R² Score: 0.9865**
        
        *The consistency of cross-validation scores demonstrates stable model behavior across multiple train-validation splits.*
        """)
        
    with st.expander("Overfitting Analysis", expanded=False):
        st.markdown("""
        **Overfitting Analysis**
        - Train R²: 0.9981
        - Test R²: 0.9875
        
        The small performance gap indicates strong generalization and minimal overfitting.
        """)
        
    with st.expander("Business Impact", expanded=False):
        st.markdown("""
        **Practical Applications**
        - EV Charging Demand Forecasting
        - Charging Station Capacity Planning
        - Queue and Congestion Management
        - Energy Distribution Optimization
        - Renewable Energy Utilization Planning
        - Smart Charging Infrastructure Expansion
        
        The system can assist charging network operators in making data-driven decisions regarding charger deployment, resource allocation, peak-hour management, and future infrastructure planning.
        """)
        
    with st.expander("Project Highlights", expanded=False):
        ph1, ph2, ph3, ph4 = st.columns(4)
        ph1.info("✓ Explored 5+ EV datasets")
        ph2.info("✓ Engineered 8 custom features")
        ph3.info("✓ Removed target leakage")
        ph4.info("✓ Evaluated 5 algorithms")
        
        ph5, ph6, ph7, ph8 = st.columns(4)
        ph5.info("✓ Achieved R² of 0.9875")
        ph6.info("✓ Robust CV & Generalization")
        ph7.info("✓ Feature Interpretation")
        ph8.info("✓ Streamlit Deployment")

elif page == "About Project":
    st.title("About ChargeWise AI")
    
    st.markdown("""
    ### Project Overview
    ChargeWise AI is a machine learning system designed to forecast EV charging demand using station conditions, vehicle characteristics, environmental variables, and temporal patterns.
    
    <div class="card" style="border-left: 4px solid #00ffcc;">
        <strong>Business Objectives:</strong>
        <p>Reduce waiting times, improve charger allocation, support infrastructure planning, and optimize charging network operations.</p>
    </div>
    
    ### Dataset & Features
    Multiple datasets (Sessions, Stations, Weather) were evaluated. Data was rigorously cleaned, and target leakage variables (like final state-of-charge and energy consumed) were removed to ensure realistic forecasting. Exploratory analysis revealed that peak hours and station congestion are primary drivers of charging demand.
    
    ### Feature Engineering
    Domain-specific features were created to capture complex operational dynamics:
    - `waiting_per_queue`: Normalizes waiting time using queue length.
    - `load_per_queue`: Distributes charging load across the queue.
    - `price_per_kWh`: Calculates effective electricity cost.
    - `charging_efficiency`: Relates charging power to battery capacity.
    - `power_to_wait_ratio`: Balances delivered power against waiting time.
    
    ### Model Development
    Multiple regression algorithms were evaluated (Linear Regression, Decision Tree, Random Forest, XGBoost, CatBoost). Random Forest was selected for its optimal balance of performance and generalization.
    
    ### Model Performance
    The final Random Forest model achieved highly robust metrics:
    - **R² Score:** 0.9875
    - **RMSE:** 3.11
    - **Mean CV Score:** 0.9865
    
    Minimal gap between training (0.9981) and testing (0.9875) R² scores indicates excellent generalization with no significant overfitting.
    
    ### Business Impact
    <div class="card" style="border-left: 4px solid #28a745;">
        <strong>Practical Applications:</strong>
        <p>Enables data-driven decisions for charging station capacity planning, dynamic queue management, optimal energy distribution, and strategic infrastructure expansion.</p>
    </div>
    
    ### Deployment
    The solution is packaged using Joblib and deployed as an interactive Streamlit web application, supporting real-time operational demand prediction.
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    ChargeWise AI &bull; EV Charging Demand Forecasting<br>
    Built by Somiya Namdeo
</div>
""", unsafe_allow_html=True)

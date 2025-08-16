import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import json
import requests
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb_model
import lightgbm as lgb_model
from scipy.stats import pearsonr

# Load saved models, scaler, and weights (only once)
gb = joblib.load('gb_model.pkl')
xgb = joblib.load('xgb_model.pkl')
lgb = joblib.load('lgb_model.pkl')
weights = joblib.load('ensemble_weights.pkl')
X = joblib.load('X.pkl')
y = joblib.load('y.pkl')
y_log = np.log(y)
w_gb = weights['w_gb']
w_xgb = weights['w_xgb']
w_lgb = weights['w_lgb']

# Define SHAPAIAgent class
class SHAPAIAgent:
    """
    AI Agent that interprets SHAP values in plain English for insurance predictions
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the AI agent with optional API key
        If no API key provided, uses fallback explanations
        """
        self.gemini_api_key = api_key
        self.client = None
        
        # Feature mappings for better explanations
        self.feature_meanings = {
            'age': 'Age of the policyholder',
            'sex': 'Gender (0=Female, 1=Male)',
            'bmi': 'Body Mass Index',
            'children': 'Number of dependents',
            'smoker': 'Smoking status (0=No, 1=Yes)',
            'region_northwest': 'Located in Northwest region',
            'region_southeast': 'Located in Southeast region',
            'region_southwest': 'Located in Southwest region'
        }
        
        # Define the app environment for Gemini API to provide context-aware advice
        self.app_environment = {
            "platform": "Streamlit",
            "dataset": {
                "context": "Insurance charges of individuals",
                "type": "log-based",
            },
            "app_title": "AI-Enhanced Prediction for Actuaries",
            "features": {
                "individual_prediction": {
                    "description": "Predict insurance charges for a single policyholder using input sliders and dropdowns.",
                    "inputs": ["Age (slider: 18-100)", "BMI (slider: 15.0-50.0)", "Number of Children (slider: 0-10)", 
                              "Sex (dropdown: Male/Female)", "Smoker (dropdown: Yes/No)", "Region (dropdown: Northeast/Northwest/Southeast/Southwest)"],
                    "models": ["Gradient Boosting", "XGBoost", "LGBRegressor", "Ensemble"],
                    "options": ["Show SHAP values (Waterfall plot)", "Calculate Upperbound Value (bootstrapping)", "Get AI Explanation"]
                },
                "total_prediction": {
                    "description": "Predict total claims cost for the entire portfolio.",
                    "models": ["Gradient Boosting", "XGBoost", "LGBRegressor", "Ensemble"],
                    "shap_plots": {
                        "Gradient Boosting/XGBoost/LGBRegressor/Ensemble": ["Beeswarm", "Bar", "Scatter", "Dependence"]
                    },
                    "options": ["Show SHAP values", "Get AI Portfolio Analysis"],
                    "scatter_dependence_features": ["age", "sex", "bmi", "children", "smoker", "region_northwest", "region_southeast", "region_southwest"]
                }
            },
            "user_interaction": {
                "input_methods": ["Sliders for numerical inputs", "Dropdowns for categorical inputs", "Checkboxes for enabling SHAP and AI explanations", 
                                 "Buttons to trigger predictions"],
                "output_methods": ["Text display for predictions", "Matplotlib plots for SHAP visualizations", "Markdown for AI explanations"]
            }
        }
    
    def extract_shap_data_from_explanation(self, shap_explanation, feature_names, input_values):
        """
        Extract SHAP values from SHAP explanation object
        """
        shap_values = shap_explanation.values
        base_value = shap_explanation.base_values
        
        shap_data = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            actual_value = input_values[i]
            
            # Convert numerical encodings to readable format
            if feature == 'sex':
                readable_value = "Male" if actual_value == 1 else "Female"
            elif feature == 'smoker':
                readable_value = "Yes" if actual_value == 1 else "No"
            elif feature.startswith('region_'):
                readable_value = "Yes" if actual_value == 1 else "No"
            else:
                readable_value = actual_value
            
            impact = "increases" if shap_val > 0 else "decreases"
            magnitude = abs(shap_val)
            
            shap_data.append({
                "feature": feature,
                "feature_meaning": self.feature_meanings.get(feature, feature),
                "value": readable_value,
                "raw_value": actual_value,
                "shap_value": shap_val,
                "impact": impact,
                "magnitude": magnitude
            })
        
        # Sort by absolute SHAP value (most important features first)
        shap_data.sort(key=lambda x: x["magnitude"], reverse=True)
        
        return shap_data, base_value

    def generate_shap_explanation(self, shap_explanation, feature_names, input_values, prediction, model_type):
        shap_data, base_value = self.extract_shap_data_from_explanation(shap_explanation, feature_names, input_values)
        prompt = self._create_interpretation_prompt(shap_data, prediction, model_type, base_value)
        if self.gemini_api_key:
            try:
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.RequestException as e:
                print(f"API Request Failed: {str(e)} - Response: {response.text if 'response' in locals() else 'No response'}")
                return self._fallback_explanation(shap_data, prediction, base_value, model_type)
        return self._fallback_explanation(shap_data, prediction, base_value, model_type)
    
    def _create_interpretation_prompt(self, shap_data, prediction, model_type, base_value):
        """
        Create a structured prompt for the LLM
        """
        top_features = shap_data[:5]

        if model_type == "ensemble":
            base_value = base_value[0] 

        shap_summary = "\n".join([
            f"- {item['feature_meaning']}: {item['value']} → {item['impact']} prediction by ${np.exp(base_value + item['magnitude']) - np.exp(base_value):.2f}"
            for item in top_features
        ])
            
        # Include app environment context in the prompt
        app_context = json.dumps(self.app_environment, indent=2)
        
        prompt = f"""
        App Environment Context:
        {app_context}

        Explain this insurance claim cost prediction in clear, professional language:
        
        PREDICTION DETAILS:
        - Final Prediction: ${prediction:.2f}
        - Base Model Value: ${np.exp(base_value):.2f}
        - Model Used: {model_type}
        
        TOP IMPACT FACTORS (SHAP Analysis):
        {shap_summary}
        
        Please provide:
        1. A clear summary explaining why this person's premium is ${prediction:.2f}
        2. The 3 most important factors driving this prediction, stated how each contributed to the predicted value, according to the {shap_summary}
        3. Brief insight on whether this is high/low risk compared to average
        4. One actionable insight for the insurance professional
        5. Based on the app environment and analysis, suggest the next steps the user should take to better understand the data or explore additional perspectives (e.g., try a different model, enable SHAP plots, or use a specific feature in the app)
        
        Keep it concise (under 150 words), professional, and avoid technical jargon.
        """
        return prompt
    
    def _fallback_explanation(self, shap_data, prediction, base_value, model_type):
        """
        Fallback explanation when AI is not available
        """
        top_3 = shap_data[:3]
        risk_level = "higher than average" if prediction > 8000 else "average" if prediction > 4000 else "lower than average"

        if model_type == "ensemble":
            base_value = base_value[0] 
        
        explanation = f"""
        ### Fallback Explanation 
        Due to the overwhelming of Gemini API, **fallback (pre-programmed)** explanation is used instead. It might not have the best result, especially for graphs that require computer vision (scatter and dependence) to interpret. Please try the AI explanation again for better result.

        #### 🎯 Prediction Summary
        **Estimated Cost: ${prediction:.2f}** (Base model estimate: ${np.exp(base_value):.2f})
        
        #### 📊 Key Driving Factors:
        1. **{top_3[0]['feature_meaning']}** ({top_3[0]['value']}) {top_3[0]['impact']} the prediction by ${np.exp(base_value + top_3[0]['magnitude']) - np.exp(base_value):.2f}
        2. **{top_3[1]['feature_meaning']}** ({top_3[1]['value']}) {top_3[1]['impact']} the prediction by ${np.exp(base_value + top_3[1]['magnitude']) - np.exp(base_value):.2f}
        3. **{top_3[2]['feature_meaning']}** ({top_3[2]['value']}) {top_3[2]['impact']} the prediction by ${np.exp(base_value + top_3[2]['magnitude']) - np.exp(base_value):.2f}
        
        #### 💼 Business Insight:
        This profile represents **{risk_level}** risk. The model's confidence is reflected in how feature impacts combine to reach the final prediction.
        
        #### Next Steps:
        Try another model to compare predictions.
        """
        return explanation
    
    def extract_total_shap_statistics(self, shap_explanation, feature_names):
        """
        Extract statistical insights from total dataset SHAP values
        """
        shap_values_np = np.asarray(shap_explanation.values).astype(float)
        shap_data_np = np.asarray(shap_explanation.data).astype(float)

        # Calculate statistics for each feature
        feature_stats = []
        for i, feature in enumerate(feature_names):
            feature_shap = shap_values_np[:, i]
            feature_data = shap_data_np[:, i]

            if np.std(feature_shap) == 0 or np.std(feature_data) == 0:
                correlation = np.nan  # Assign NaN if variance is zero
            else:
                # pearsonr returns (correlation_coefficient, p_value)
                correlation, _ = pearsonr(feature_shap, feature_data)
            
            stats = {
                "feature": feature,
                "feature_meaning": self.feature_meanings.get(feature, feature),
                "mean_impact": np.mean(feature_shap),
                "mean_abs_impact": np.mean(np.abs(feature_shap)),
                "correlation": correlation,
                "impact": "increases" if correlation > 0 else "decreases"
            }
            feature_stats.append(stats)
        
        # Sort by mean absolute impact (most important overall)
        feature_stats.sort(key=lambda x: x["mean_abs_impact"], reverse=True)
        
        return feature_stats
    
    def generate_total_shap_explanation(self, shap_explanation, feature_names, total_prediction, model_type, graph, feature_name=None):
        """
        Generate AI explanation for total dataset SHAP analysis
        """
        feature_stats = self.extract_total_shap_statistics(shap_explanation, feature_names)
        prompt = self._create_total_interpretation_prompt(feature_stats, total_prediction, model_type, graph, feature_name, shap_explanation)
        
        if self.gemini_api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return self._fallback_total_explanation(feature_stats, total_prediction, model_type)
        
        return self._fallback_total_explanation(feature_stats, total_prediction, model_type)
    
    def _create_total_interpretation_prompt(self, feature_stats, total_prediction, model_type, graph, feature_name=None, shap_explanation=None):
        """
        Create a structured prompt for total dataset analysis, including scatter and dependence plot interpretations when applicable.
        """
        top_features = feature_stats[:5]

        if model_type == 'gradient boosting':  
            base_value = explainer_gb.expected_value
        elif model_type == 'xgboost':  
            base_value = explainer_xgb.expected_value
        elif model_type == 'lgbregressor':  
            base_value = explainer_lgb.expected_value
        elif model_type == 'ensemble':
            base_value = w_gb * explainer_gb.expected_value + w_xgb * explainer_xgb.expected_value + w_lgb * explainer_lgb.expected_value

        base_value = np.mean(base_value) if isinstance(base_value, (list, np.ndarray)) else base_value

        # Prepare feature summaries for Bar and Beeswarm graphs
        feature_summary_bar = "\n".join([
            f"- {item['feature_meaning']}: Contributing an average of $${np.exp(base_value + item['mean_abs_impact']) - np.exp(base_value):.2f} "
            for item in top_features
        ])

        feature_summary_beeswarm = "\n".join([
            f"- {item['feature_meaning']}: As {item['feature_meaning']} increases, the insurance charges {item['impact']}"
            for item in top_features
        ])

        # Include app environment context in the prompt
        app_context = json.dumps(self.app_environment, indent=2)

        # Base prompt structure
        prompt = f"""
        App Environment Context:
        {app_context}

        Analyze this insurance portfolio's risk factors based on SHAP analysis across all policyholders:
    
        PORTFOLIO SUMMARY:
        - Total Predicted Claims: ${total_prediction:,.2f}
        - Model Used: {model_type}
        - Dataset Size: Portfolio analysis across all policyholders
        """

        if graph == "Bar":
            prompt += f"""
            TOP RISK FACTORS (by average impact):
            {feature_summary_bar}
        
            Please provide:
            1. Portfolio risk assessment - what drives costs across the entire portfolio
            2. Top 3 factors that insurance companies should focus on for actuaries, according to {feature_summary_bar}
            3. Risk concentration insights - which factors create the most variability
            4. Strategic recommendations for portfolio management
            5. Based on the app environment and analysis, suggest the next steps the user should take to better understand the data or explore additional perspectives (e.g., try a different SHAP plot like Scatter for specific feature analysis or switch models)
        
            Keep it professional and actionable (under 200 words).
            """
        elif graph == "Beeswarm":
            prompt += f"""
            TOP RISK FACTORS (by average impact):
            {feature_summary_beeswarm}
        
            Please provide:
            1. Portfolio risk assessment - what drives costs across the entire portfolio
            2. Top 3 factors that insurance companies should focus on for actuaries, according to {feature_summary_beeswarm}
            3. Risk concentration insights - which factors create the most variability
            4. Strategic recommendations for portfolio management
            5. Based on the app environment and analysis, suggest the next steps the user should take to better understand the data or explore additional perspectives (e.g., try a Scatter plot to see how a specific feature like 'age' impacts predictions)
        
            Keep it professional and actionable (under 200 words).
            """
        elif graph == "Scatter" and feature_name and shap_explanation:
            # Extract SHAP values and feature data for scatter plot
            shap_values_for_feature = shap_explanation.values[:, shap_explanation.feature_names.index(feature_name)]
            feature_values = shap_explanation.data[:, shap_explanation.feature_names.index(feature_name)]
        
            # Combine into data points for scatter plot
            data_points = [
                {"x": float(feature_values[i]), "y": float(shap_values_for_feature[i])}
                for i in range(len(feature_values))
            ]
        
            prompt += f"""
            SCATTER PLOT ANALYSIS:
            The scatter plot shows the feature '{feature_name}' (x-axis) vs SHAP values (y-axis, impact on prediction).
            Data points (x=feature value, y=SHAP value):
            {json.dumps(data_points[:200])}
        
            Please provide:
            1. Describe how the dots are laid out (how they perform on the plot) and what we can learn from it. Focus on this to comprehensively explain the plot's behavior
            2. Point out the x value range where SHAP values are above and below 0 (e.g., x in [a,b] increases the prediction) and what we can learn from it
            3. Risk concentration insights - is this factor a significant driver of predicted charges (how far it impacts SHAP values), and how strategic it can be
            4. Strategic recommendations for portfolio management, factoring in the scatter plot analysis
            5. Based on the app environment and analysis, suggest the next steps the user should take to better understand the data or explore additional perspectives (e.g., try a Dependence plot to see interactions with another feature)
        
            Keep it professional and actionable (under 200 words).
            """
        elif graph == "Dependence" and feature_name and shap_explanation:
            # Assuming feature_name is a tuple of two features (x-axis and color)
            feature_x, feature_color = feature_name
            x_idx = shap_explanation.feature_names.index(feature_x)
            color_idx = shap_explanation.feature_names.index(feature_color)
            
            # Extract data for dependence plot
            x_values = shap_explanation.data[:, x_idx]
            shap_x_values = shap_explanation.values[:, x_idx]
            color_values = shap_explanation.data[:, color_idx]
            
            # Combine into data points for dependence plot
            data_points = [
                {"x": float(x_values[i]), "y": float(shap_x_values[i]), "color": float(color_values[i])}
                for i in range(len(x_values))
            ]
            
            prompt += f"""
            DEPENDENCE PLOT ANALYSIS:
            The dependence plot shows the feature '{feature_x}' (x-axis) vs SHAP values (y-axis), colored by '{feature_color}'.
            Data points (x={feature_x} value, y=SHAP value, color={feature_color} value):
            {json.dumps(data_points[:200])}
        
            TOP RISK FACTORS (by average impact):
            {feature_summary_beeswarm}
        
            Please provide:
            1. Describe how '{feature_x}' and '{feature_color}' relate to each other (how they perform on the plot) and what we can learn from it
            2. Risk concentration insights - is the relationship impactful (is it significant or just random), and what we can take from it for further analysis
            3. Strategic recommendations for portfolio management, factoring in the dependence plot analysis
            4. Based on the app environment and analysis, suggest the next steps the user should take to better understand the data or explore additional perspectives (e.g., try a Scatter plot for a single feature focus)
        
            Keep it professional and actionable (under 200 words).
            """
        return prompt
    
    def _fallback_total_explanation(self, feature_stats, total_prediction, model_type):
        """
        Fallback explanation for total dataset analysis
        """
        if model_type == 'gradient boosting':  
            base_value = explainer_gb.expected_value
        elif model_type == 'xgboost':  
            base_value = explainer_xgb.expected_value
        elif model_type == 'lgbregressor':  
            base_value = explainer_lgb.expected_value
        elif model_type == 'ensemble':
            base_value = w_gb * explainer_gb.expected_value + w_xgb * explainer_xgb.expected_value + w_lgb * explainer_lgb.expected_value

        base_value = np.mean(base_value) if isinstance(base_value, (list, np.ndarray)) else base_value

        top_3 = feature_stats[:3]

        explanation = f"""
        ### Fallback Explanation 
        Due to the overwhelming of Gemini API, **fallback (pre-programmed) explanation** is used instead. It might not have the best result, especially for graphs that require computer vision (scatter and dependence) to interpret. Please try the AI explanation again for better result.
        
        **Total Predicted Claims: ${total_prediction:,.2f}**
        
        #### 🎯 Key Risk Drivers Across Portfolio:
        1. **{top_3[0]['feature_meaning']}**: Average impact ${np.exp(base_value + top_3[0]['mean_abs_impact']) - np.exp(base_value):.2f} 
        2. **{top_3[1]['feature_meaning']}**: Average impact ${np.exp(base_value + top_3[1]['mean_abs_impact']) - np.exp(base_value):.2f} 
        3. **{top_3[2]['feature_meaning']}**: Average impact ${np.exp(base_value + top_3[2]['mean_abs_impact']) - np.exp(base_value):.2f} 
        
        #### 💼 Strategic Insights:
        - Focus underwriting attention on the top 3 factors above
        - Consider segment-specific pricing for high-variability factors
        - Monitor risk concentration in key demographic segments
        
        #### 📈 Portfolio Health:
        The model shows clear risk differentiation, enabling precise underwriting and competitive pricing strategies.
        
        #### Next Steps:
        Try another plot to explore specific feature impacts or switch to another model for comparison.
        """
        return explanation

# Initialize AI agents globally
GEMINI_API_KEY = "AIzaSyAnzPOShZyb0yk6rmTQ36xH02Y7W9S4tRU"
ai_agent = SHAPAIAgent(api_key=GEMINI_API_KEY)

# Streamlit app title
st.title("🏥 AI-Enhanced Prediction for Actuaries")
st.write("Enter the details below to predict insurance charges using high-accuracy machine learning models with AI-powered explanations.")

# Create input fields for user
st.header("User Input")
age = st.slider("Age", min_value=18, max_value=100, value=63, key="age_input")
bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=37.7, step=0.1, key="bmi_input")
children = st.slider("Number of Children", min_value=0, max_value=10, value=0, key="children_input")
sex = st.selectbox("Sex", options=["Male", "Female"], key="sex_input")
smoker = st.selectbox("Smoker", options=["Yes", "No"], key="smoker_input")
region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"], key="region_input")

# Convert inputs to model-compatible format
sex_encoded = 1 if sex == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

# Create a dictionary for the new person
new_person = {
    'age': age,
    'sex': sex_encoded,  
    'bmi': bmi,
    'children': children,
    'smoker': smoker_encoded,
    'region_northwest': region_northwest,
    'region_southeast': region_southeast,
    'region_southwest': region_southwest
}

# Convert to DataFrame and ensure correct column order
feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']
new_X = pd.DataFrame([new_person])
new_X = new_X[feature_columns]

# SHAP setup
explainer_gb = shap.Explainer(gb)
explainer_xgb = shap.Explainer(xgb)
explainer_lgb = shap.Explainer(lgb)

# Predict button
st.header("Prediction Result")
ind_sel = st.selectbox("Selection of ML models", options=["Gradient Boosting", "XGBoost", "LGBRegressor", "Ensemble"], key="ind_sel")
ind_show_shap = st.checkbox("Show SHAP values", key="ind_show_shap")
ind_upperbound = st.checkbox("Calculate Upperbound Value", key="ind_upperbound", help="This uses bootstrapping, which might take a little while")

# AI Explanation option
ind_show_ai = st.checkbox("🤖 Get AI Explanation", key="ind_show_ai", help="Get plain-English explanation of the prediction")

# Upper bound value
def get_bootstrap_prediction_upper_bound(model, new_X_input, num_bootstraps=500, alpha=0.95):
    bootstrap_predictions = []
    n_samples = X.shape[0]

    for _ in range(num_bootstraps):
        # 1. Resample the training data with replacement
        # Get random indices for this bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[bootstrap_indices]
        y_boot = y_log.iloc[bootstrap_indices]  # Use .iloc for Series/DataFrame

        # 2. Train a new model on each bootstrap sample
        if model == 'gb':
            boot_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                   loss='squared_error', random_state=np.random.randint(0, 10000))  # Random seed for each model
            boot_model.fit(X_boot, y_boot)
        elif model == 'xgb':
            boot_model = xgb_model.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                objective='reg:squarederror', random_state=np.random.randint(0, 10000))  # Random seed for each model
            boot_model.fit(X_boot, y_boot)
        elif model == 'lgb':
            boot_model = lgb_model.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                 objective='regression', random_state=np.random.randint(0, 10000))  # Random seed for each model
            boot_model.fit(X_boot, y_boot)
        # 3. Predict for the new data point
        # Ensure new_X_input is in the correct DataFrame format
        pred_log = boot_model.predict(new_X_input)[0]
        bootstrap_predictions.append(np.exp(pred_log))  # Store exponentiated predictions

    # 4. Calculate the desired quantile (upper bound)
    upper_bound_value = np.percentile(bootstrap_predictions, alpha * 100)
    return upper_bound_value  # Return predictions list for plotting distribution

if ind_show_shap:
    ind_sel_shap = st.selectbox("Selection of graphs", options=["Waterfall"], key="ind_sel_shap")
    shap_values_gb = explainer_gb(new_X)
    shap_values_xgb = explainer_xgb(new_X)
    shap_values_lgb = explainer_lgb(new_X)

if st.button("Predict Individual"):
    log_gb = gb.predict(new_X)
    log_xgb = xgb.predict(new_X)
    log_lgb = lgb.predict(new_X)
    
    current_prediction = 0
    current_shap_explanation = None
    current_model_type = ind_sel.lower()
    
    if ind_sel == "Gradient Boosting": 
        gb_prediction = np.exp(log_gb)[0]
        current_prediction = gb_prediction
        st.write(f"Estimated Insurance Charges: **${gb_prediction:.2f}**")
        if ind_upperbound:
            with st.spinner("This uses bootstrapping so it may take a few seconds..."):
                bootstrap_ub_gb = get_bootstrap_prediction_upper_bound('gb', new_X)
                st.write(f"Upperbound Estimated: **${bootstrap_ub_gb:.2f}**")
        if ind_show_shap:
            current_shap_explanation = shap_values_gb[0]
            st.subheader("SHAP Explanation for Gradient Boosting")
            plt.figure()
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(shap_values_gb[0])
            st.pyplot(plt.gcf())
            plt.close()
    elif ind_sel == "XGBoost": 
        xgb_prediction = np.exp(log_xgb)[0]
        current_prediction = xgb_prediction
        st.write(f"Estimated Insurance Charges: **${xgb_prediction:.2f}**")
        if ind_upperbound:
            with st.spinner("This uses bootstrapping so it may take a few seconds..."):
                bootstrap_ub_xgb = get_bootstrap_prediction_upper_bound('xgb', new_X)
                st.write(f"Upperbound Estimated: **${bootstrap_ub_xgb:.2f}**")
        if ind_show_shap:
            current_shap_explanation = shap_values_xgb[0]
            st.subheader("SHAP Explanation for XGBoost")
            plt.figure()
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(shap_values_xgb[0])
            st.pyplot(plt.gcf())
            plt.close()
    elif ind_sel == "LGBRegressor": 
        lgb_prediction = np.exp(log_lgb)[0]
        current_prediction = lgb_prediction
        st.write(f"Estimated Insurance Charges: **${lgb_prediction:.2f}**")
        if ind_upperbound:
            with st.spinner("This uses bootstrapping so it may take a few seconds..."):
                bootstrap_ub_lgb = get_bootstrap_prediction_upper_bound('lgb', new_X)
                st.write(f"Upperbound Estimated: **${bootstrap_ub_lgb:.2f}**")
        if ind_show_shap:
            current_shap_explanation = shap_values_lgb[0]
            st.subheader("SHAP Explanation for LGBRegressor")
            plt.figure()
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(shap_values_lgb[0])
            st.pyplot(plt.gcf())
            plt.close()
    elif ind_sel == "Ensemble":
        log_ensemble = w_gb * log_gb + w_xgb * log_xgb + w_lgb * log_lgb
        ensemble_prediction = np.exp(log_ensemble)[0]
        current_prediction = ensemble_prediction
        st.write(f"Estimated Insurance Charges: **${ensemble_prediction:.2f}**")
        if ind_show_shap:
            shap_values_ensemble = w_gb * shap_values_gb.values + w_xgb * shap_values_xgb.values + w_lgb * shap_values_lgb.values
            expected_value_ensemble = w_gb * explainer_gb.expected_value + w_xgb * explainer_xgb.expected_value + w_lgb * explainer_lgb.expected_value
            shap_exp_ensemble = shap.Explanation(values=shap_values_ensemble[0], base_values=expected_value_ensemble, data=new_X.iloc[0])
            current_shap_explanation = shap_exp_ensemble
            st.subheader("SHAP Explanation for Ensemble")
            plt.figure()
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(shap_exp_ensemble)
            st.pyplot(plt.gcf())
            plt.close()
    
    # AI Explanation section
    if ind_show_ai and current_shap_explanation is not None:
        st.subheader("🤖 AI-Powered Explanation")
        with st.spinner("Generating AI explanation..."):
            input_values = [age, sex_encoded, bmi, children, smoker_encoded, region_northwest, region_southeast, region_southwest]
            ai_explanation = ai_agent.generate_shap_explanation(
                shap_explanation=current_shap_explanation,
                feature_names=feature_columns,
                input_values=input_values,
                prediction=current_prediction,
                model_type=current_model_type
            )
            st.markdown(ai_explanation)
    elif ind_show_ai and current_shap_explanation is None:
        st.warning("⚠️ Please enable 'Show SHAP values' to get AI explanations")
    
# Total claim cost prediction
st.header("Total Claims Prediction")
tot_sel = st.selectbox("Selection of ML models", options=["Gradient Boosting", "XGBoost", "LGBRegressor", "Ensemble"], key="tot_sel")
tot_show_shap = st.checkbox("Show SHAP values", key="tot_show_shap")
if tot_show_shap:
    tot_sel_shap = st.selectbox("Selection of graphs", options=["Beeswarm", "Bar", "Scatter", "Dependence"], key="tot_sel_shap")
    if tot_sel_shap == "Scatter":
        tot_sel_shap_scatter = st.selectbox("Selection of feature", options=["age", "sex", "bmi", "children", "smoker", "region_northwest", "region_southeast", "region_southwest"], key="tot_sel_shap_scatter")
    elif tot_sel_shap == "Dependence":
        feature_options = ["age", "sex", "bmi", "children", "smoker", "region_northwest", "region_southeast", "region_southwest"]
        tot_sel_shap_dependence_1 = st.selectbox("Selection of feature (x-axis)", options=feature_options, key="tot_sel_shap_dependence_1")
        remaining_options = [opt for opt in feature_options if opt != tot_sel_shap_dependence_1]
        tot_sel_shap_dependence_2 = st.selectbox("Selection of feature (color)", options=remaining_options, key="tot_sel_shap_dependence_2")
    shap_values_gb_total = explainer_gb(X)
    shap_values_xgb_total = explainer_xgb(X)
    shap_values_lgb_total = explainer_lgb(X)

# AI Explanation option for total predictions
tot_show_ai = st.checkbox("🤖 Get AI Portfolio Analysis", key="tot_show_ai", help="Get AI-powered analysis of your entire portfolio's risk factors")

if st.button("Predict Total"):
    current_total_prediction = 0
    current_total_shap_explanation = None
    current_model_type = tot_sel.lower()

    log_gb_total = gb.predict(X)
    log_xgb_total = xgb.predict(X)
    log_lgb_total = lgb.predict(X)
    
    if tot_sel == "Gradient Boosting": 
        gb_prediction_total = np.exp(log_gb_total)
        current_total_prediction = sum(gb_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            current_total_shap_explanation = shap_values_gb_total
            st.subheader("SHAP Explanation for Gradient Boosting")
            plt.figure()
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(shap_values_gb_total)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(shap_values_gb_total)  
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(shap_values_gb_total[:, tot_sel_shap_scatter])
            elif tot_sel_shap == "Dependence":
                shap.plots.scatter(shap_values_gb_total[:, tot_sel_shap_dependence_1], color=shap_values_gb_total[:, tot_sel_shap_dependence_2])
            st.pyplot(plt.gcf())
            plt.close()
    elif tot_sel == "XGBoost": 
        xgb_prediction_total = np.exp(log_xgb_total)
        current_total_prediction = sum(xgb_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            current_total_shap_explanation = shap_values_xgb_total
            st.subheader("SHAP Explanation for XGBoosting")
            plt.figure()
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(shap_values_xgb_total)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(shap_values_xgb_total)  
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(shap_values_xgb_total[:, tot_sel_shap_scatter])
            elif tot_sel_shap == "Dependence":
                shap.plots.scatter(shap_values_xgb_total[:, tot_sel_shap_dependence_1], color=shap_values_xgb_total[:, tot_sel_shap_dependence_2])
            st.pyplot(plt.gcf())
            plt.close()
    elif tot_sel == "LGBRegressor": 
        lgb_prediction_total = np.exp(log_lgb_total)
        current_total_prediction = sum(lgb_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            current_total_shap_explanation = shap_values_lgb_total
            st.subheader("SHAP Explanation for LGBRegressor")
            plt.figure()
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(shap_values_lgb_total)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(shap_values_lgb_total)
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(shap_values_lgb_total[:, tot_sel_shap_scatter])  
            elif tot_sel_shap == "Dependence":
                shap.plots.scatter(shap_values_lgb_total[:, tot_sel_shap_dependence_1], color=shap_values_lgb_total[:, tot_sel_shap_dependence_2])
            st.pyplot(plt.gcf())
            plt.close()
    elif tot_sel == "Ensemble":  
        log_ensemble_total = w_gb * log_gb_total + w_xgb * log_xgb_total + w_lgb * log_lgb_total  
        ensemble_prediction_total = np.exp(log_ensemble_total)
        current_total_prediction = sum(ensemble_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            shap_values_ensemble_total = w_gb * shap_values_gb_total.values + w_xgb * shap_values_xgb_total.values + w_lgb * shap_values_lgb_total.values
            expected_value_ensemble_total = w_gb * explainer_gb.expected_value + w_xgb * explainer_xgb.expected_value + w_lgb * explainer_lgb.expected_value
            shap_exp_ensemble_total = shap.Explanation(values=shap_values_ensemble_total, base_values=expected_value_ensemble_total, data=X.values, feature_names=X.columns)
            current_total_shap_explanation = shap_exp_ensemble_total
            st.subheader("SHAP Explanation for Ensemble")
            plt.figure()
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(shap_exp_ensemble_total)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(shap_exp_ensemble_total)
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(shap_exp_ensemble_total[:, tot_sel_shap_scatter])  
            elif tot_sel_shap == "Dependence":
                shap.plots.scatter(shap_exp_ensemble_total[:, tot_sel_shap_dependence_1], color=shap_exp_ensemble_total[:, tot_sel_shap_dependence_2])
            st.pyplot(plt.gcf())
            plt.close()

    # AI Explanation for total claims
    if tot_show_ai and current_total_shap_explanation is not None:
        st.subheader("🤖 AI-Powered Portfolio Analysis")
        with st.spinner("Generating AI portfolio analysis..."):
            feature_name = None
            if tot_sel_shap == "Scatter":
                feature_name = tot_sel_shap_scatter
            elif tot_sel_shap == "Dependence":
                feature_name = (tot_sel_shap_dependence_1, tot_sel_shap_dependence_2)
            ai_total_explanation = ai_agent.generate_total_shap_explanation(
                shap_explanation=current_total_shap_explanation,
                feature_names=feature_columns,
                total_prediction=current_total_prediction,
                model_type=current_model_type,
                graph=tot_sel_shap,
                feature_name=feature_name
            )
            st.markdown(ai_total_explanation)
    elif tot_show_ai and current_total_shap_explanation is None:
        st.warning("⚠️ Please enable 'Show SHAP values' to get AI explanations")
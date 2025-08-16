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

# Import ReportLab modules
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

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

# Define SHAPAIAgent class (omitted for brevity, assume it's correctly defined)
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
        # For a full implementation, you'd initialize a Gemini client here
        # self.client = genai.GenerativeModel('gemini-1.5-flash') # Example if using google-generative-ai library

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
            # Assuming you've set up the Google AI client properly
            # response = self.client.generate_content(prompt)
            # return response.text

            # Using requests for direct API call as in your original code
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                # Fallback if API call fails
                st.warning(f"Gemini API call failed with status {response.status_code}. Using fallback explanation.")
                return self._fallback_explanation(shap_data, prediction, base_value)
        return self._fallback_explanation(shap_data, prediction, base_value)

    def _create_interpretation_prompt(self, shap_data, prediction, model_type, base_value):
        """
        Create a structured prompt for the LLM
        """
        top_features = shap_data[:5]
        shap_summary = "\n".join([
            f"- {item['feature_meaning']}: {item['value']} → {item['impact']} prediction by ${np.exp(item['magnitude']):.2f}"
            for item in top_features
        ])

        if model_type == "ensemble":
            base_value = base_value[0] # Base value for ensemble might be an array if from multiple explainers

        prompt = f"""
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

        Keep it concise (under 150 words), professional, and avoid technical jargon.
        """
        return prompt

    def _fallback_explanation(self, shap_data, prediction, base_value):
        """
        Fallback explanation when AI is not available or API fails
        """
        top_3 = shap_data[:3]
        risk_level = "higher than average" if prediction > 8000 else "average" if prediction > 4000 else "lower than average"

        explanation = f"""
        ## 🎯 Prediction Summary
        **Estimated Cost: ${prediction:.2f}** (Base model estimate: ${np.exp(base_value):.2f})

        ## 📊 Key Driving Factors:
        1. **{top_3[0]['feature_meaning']}** ({top_3[0]['value']}) {top_3[0]['impact']} the prediction by ${np.exp(top_3[0]['magnitude']):.2f}
        2. **{top_3[1]['feature_meaning']}** ({top_3[1]['value']}) {top_3[1]['impact']} the prediction by ${np.exp(top_3[1]['magnitude']):.2f}
        3. **{top_3[2]['feature_meaning']}** ({top_3[2]['value']}) {top_3[2]['impact']} the prediction by ${np.exp(top_3[2]['magnitude']):.2f}

        ## 💼 Business Insight:
        This profile represents **{risk_level}** risk. The model's confidence is reflected in how feature impacts combine to reach the final prediction.
        """
        return explanation

    def extract_total_shap_statistics(self, shap_explanation, feature_names):
        """
        Extract statistical insights from total dataset SHAP values
        """
        if hasattr(shap_explanation, 'values'):
            shap_values = shap_explanation.values
        else:
            shap_values = shap_explanation # If it's already an array of values

        # Calculate statistics for each feature
        feature_stats = []
        for i, feature in enumerate(feature_names):
            feature_shap = shap_values[:, i]

            stats = {
                "feature": feature,
                "feature_meaning": self.feature_meanings.get(feature, feature),
                "mean_impact": np.mean(feature_shap),
                "mean_abs_impact": np.mean(np.abs(feature_shap)),
                "std_impact": np.std(feature_shap),
                "max_impact": np.max(feature_shap),
                "min_impact": np.min(feature_shap),
                "positive_impact_pct": (feature_shap > 0).mean() * 100,
                "negative_impact_pct": (feature_shap < 0).mean() * 100,
                "total_positive_impact": np.sum(feature_shap[feature_shap > 0]),
                "total_negative_impact": np.sum(feature_shap[feature_shap < 0])
            }
            feature_stats.append(stats)

        # Sort by mean absolute impact (most important overall)
        feature_stats.sort(key=lambda x: x["mean_abs_impact"], reverse=True)

        return feature_stats

    def generate_total_shap_explanation(self, shap_explanation, feature_names, total_prediction, model_type, graph):
        """
        Generate AI explanation for total dataset SHAP analysis
        """
        feature_stats = self.extract_total_shap_statistics(shap_explanation, feature_names)
        prompt = self._create_total_interpretation_prompt(feature_stats, total_prediction, model_type, graph)

        if self.gemini_api_key:
            # response = self.client.generate_content(prompt)
            # return response.text
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                st.warning(f"Gemini API call failed with status {response.status_code}. Using fallback explanation.")
                return self._fallback_total_explanation(feature_stats, total_prediction)
        return self._fallback_total_explanation(feature_stats, total_prediction)

    def _create_total_interpretation_prompt(self, feature_stats, total_prediction, model_type, graph):
        """
        Create a structured prompt for total dataset analysis
        """
        top_features = feature_stats[:5]

        feature_summary = "\n".join([
            f"- {item['feature_meaning']}: Avg impact ${np.exp(item['mean_impact']):.2f}, "
            f"{item['positive_impact_pct']:.1f}% increase premiums, "
            f"{item['negative_impact_pct']:.1f}% decrease premiums"
            for item in top_features
        ])

        feature_summary_bar = "\n".join([
            f"- {item['feature_meaning']}: Contributing an average of ${np.exp(item['mean_abs_impact']):.2f}"
            for item in top_features
        ])

        # Calculate portfolio insights
        high_impact_features = [f for f in feature_stats if f['mean_abs_impact'] > 0.1] # Example threshold
        most_variable_feature = max(feature_stats, key=lambda x: x['std_impact'])

        if graph == "Bar":
            prompt = f"""
            Analyze this insurance portfolio's risk factors based on SHAP analysis across all policyholders:

            PORTFOLIO SUMMARY:
            - Total Predicted Claims: ${total_prediction:,.2f}
            - Model Used: {model_type}
            - Dataset Size: Portfolio analysis across all policyholders

            TOP RISK FACTORS:
            {feature_summary_bar}

            KEY INSIGHTS:
            - High Impact Features: {len(high_impact_features)} factors significantly affect premiums

            Please provide:
            1. Portfolio risk assessment - what drives costs across the entire portfolio
            2. Top 3 factors that insurance companies should focus on for actuaries, according to {feature_summary_bar}
            3. Risk concentration insights - which factors create the most variability
            4. Strategic recommendations for portfolio management

            Keep it professional and actionable (under 200 words).
            """
        else: # Beeswarm or Scatter
            prompt = f"""
            Analyze this insurance portfolio's risk factors based on SHAP analysis across all policyholders:

            PORTFOLIO SUMMARY:
            - Total Predicted Claims: ${total_prediction:,.2f}
            - Model Used: {model_type}
            - Dataset Size: Portfolio analysis across all policyholders

            TOP RISK FACTORS (by average impact):
            {feature_summary}

            KEY INSIGHTS:
            - Most Variable Factor: {most_variable_feature['feature_meaning']} (std: ${np.exp(most_variable_feature['std_impact']):.2f})

            Please provide:
            1. Portfolio risk assessment - what drives costs across the entire portfolio
            2. Top 3 factors that insurance companies should focus on for actuaries, according to {feature_summary}
            3. Risk concentration insights - which factors create the most variability
            4. Strategic recommendations for portfolio management

            Keep it professional and actionable (under 200 words).
            """
        return prompt

    def _fallback_total_explanation(self, feature_stats, total_prediction):
        """
        Fallback explanation for total dataset analysis
        """
        top_3 = feature_stats[:3]
        most_variable = max(feature_stats, key=lambda x: x['std_impact'])

        explanation = f"""
        ## 📊 Portfolio Risk Analysis
        **Total Predicted Claims: ${total_prediction:,.2f}**

        ## 🎯 Key Risk Drivers Across Portfolio:
        1. **{top_3[0]['feature_meaning']}**: Average impact ${np.exp(top_3[0]['mean_impact']):.2f}, affects {top_3[0]['positive_impact_pct']:.1f}% of policies positively
        2. **{top_3[1]['feature_meaning']}**: Average impact ${np.exp(top_3[1]['mean_impact']):.2f}, affects {top_3[1]['positive_impact_pct']:.1f}% of policies positively
        3. **{top_3[2]['feature_meaning']}**: Average impact ${np.exp(top_3[2]['mean_impact']):.2f}, affects {top_3[2]['positive_impact_pct']:.1f}% of policies positively

        ## 🔄 Risk Variability:
        **{most_variable['feature_meaning']}** shows the highest variability (std: ${np.exp(most_variable['std_impact']):.2f}), indicating inconsistent impact across the portfolio.

        ## 💼 Strategic Insights:
        - Focus underwriting attention on the top 3 factors above
        - Consider segment-specific pricing for high-variability factors
        - Monitor risk concentration in key demographic segments

        ## 📈 Portfolio Health:
        The model shows clear risk differentiation, enabling precise underwriting and competitive pricing strategies.
        """
        return explanation

# Initialize AI agent globally
GEMINI_API_KEY = "AIzaSyAnzPOShZyb0yk6rmTQ36xH02Y7W9S4tRU" # Replace with your actual Gemini API key
ai_agent = SHAPAIAgent(api_key=GEMINI_API_KEY)

# Streamlit app title
st.title("🏥 AI-Enhanced Prediction for Actuaries")
st.write("Enter the details below to predict insurance charges using high-accuracy machine learning models with AI-powered explanations.")

# --- Initialize Session State ---
# This ensures that these dictionaries exist on the very first run of the script
# and persist across subsequent reruns.
if 'individual_prediction_results' not in st.session_state:
    st.session_state.individual_prediction_results = {
        'prediction_value': None,
        'upper_bound_value': None,
        'shap_plot_buffer': None, # To store the BytesIO object
        'ai_explanation_text': None,
        'model_type': None,
        'input_details': None,
        'has_predicted': False # Flag to control display of results
    }
if 'total_prediction_results' not in st.session_state:
    st.session_state.total_prediction_results = {
        'total_prediction_value': None,
        'shap_plot_buffer': None,
        'ai_explanation_text': None,
        'model_type': None,
        'has_predicted': False
    }
# --- End Initialize Session State ---


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

# Predict button (Individual)
st.header("Individual Prediction Result")
ind_sel = st.selectbox("Selection of ML models", options=["Gradient Boosting", "XGBoost", "LGBRegressor", "Ensemble"], key="ind_sel")
ind_show_shap = st.checkbox("Show SHAP values", key="ind_show_shap")
ind_upperbound = st.checkbox("Calculate Upperbound Value", key="ind_upperbound", help="This uses bootstrapping, which might take a little while")
ind_show_ai = st.checkbox("🤖 Get AI Explanation", key="ind_show_ai", help="Get plain-English explanation of the prediction")

# Upper bound value function (remains the same)
def get_bootstrap_prediction_upper_bound(model, new_X_input, num_bootstraps=500, alpha=0.95):
    bootstrap_predictions = []
    n_samples = X.shape[0]

    for _ in range(num_bootstraps):
        # 1. Resample the training data with replacement
        # Get random indices for this bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[bootstrap_indices]
        y_boot = y_log.iloc[bootstrap_indices] # Use .iloc for Series/DataFrame

        # 2. Train a new model on each bootstrap sample
        if model == 'gb':
            boot_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                   loss='squared_error', random_state=np.random.randint(0, 10000)) # Random seed for each model
            boot_model.fit(X_boot, y_boot)
        elif model == 'xgb':
            boot_model = xgb_model.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                   objective='reg:squarederror', random_state=np.random.randint(0, 10000)) # Random seed for each model
            boot_model.fit(X_boot, y_boot)
        elif model == 'lgb':
            boot_model = lgb_model.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                   objective='regression', random_state=np.random.randint(0, 10000)) # Random seed for each model
            boot_model.fit(X_boot, y_boot)
        # 3. Predict for the new data point
        # Ensure new_X_input is in the correct DataFrame format
        pred_log = boot_model.predict(new_X_input)[0]
        bootstrap_predictions.append(np.exp(pred_log)) # Store exponentiated predictions

    # 4. Calculate the desired quantile (upper bound)
    upper_bound_value = np.percentile(bootstrap_predictions, alpha * 100)
    return upper_bound_value # Return predictions list for plotting distribution

if ind_show_shap: # Only show SHAP graph selection if SHAP is enabled
    ind_sel_shap = st.selectbox("Selection of individual SHAP graphs", options=["Waterfall"], key="ind_sel_shap")


# --- PDF Report Generation Function ---
# This function creates the PDF content (BytesIO object)
def create_pdf_report(
    prediction_type,
    model_used,
    prediction_value=None,
    shap_plot_bytes=None, # Expects a BytesIO object
    ai_explanation=None,
    upper_bound_value=None,
    input_details=None,
    total_prediction_value=None
):
    buffer = BytesIO() # Create a new BytesIO buffer for each PDF generation
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Insurance Prediction Report - {prediction_type}", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # General Information
    story.append(Paragraph(f"<b>Model Used:</b> {model_used}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    if prediction_type == "Individual Prediction":
        if prediction_value is not None:
            story.append(Paragraph(f"<b>Estimated Insurance Charges:</b> ${prediction_value:.2f}", styles['Normal']))
        if upper_bound_value:
            story.append(Paragraph(f"<b>Upperbound Estimated:</b> ${upper_bound_value:.2f}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        if input_details:
            story.append(Paragraph("<b>Input Details:</b>", styles['h3']))
            for key, value in input_details.items():
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{key}:</b> {value}", styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

    elif prediction_type == "Total Claims Prediction":
        if total_prediction_value is not None:
            story.append(Paragraph(f"<b>Total Claim Cost Prediction:</b> ${total_prediction_value:,.2f}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))


    # AI Explanation
    if ai_explanation:
        story.append(Paragraph("<b>AI-Powered Explanation:</b>", styles['h2']))
        # Clean up markdown for ReportLab's basic HTML parser
        ai_explanation_cleaned = ai_explanation.replace("##", "<b>").replace("\n", "<br/>").replace("###", "<b>").replace("**", "<b>").replace(":", ":</b>")
        ai_explanation_cleaned = ai_explanation_cleaned.replace("<ul>", "").replace("</ul>", "")
        ai_explanation_cleaned = ai_explanation_cleaned.replace("<li>", "&bull; ").replace("</li>", "<br/>")
        ai_explanation_cleaned = ai_explanation_cleaned.replace("<b>- ", "<b>&bull; ") # Handle list items that might be bolded

        story.append(Paragraph(ai_explanation_cleaned, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

    # SHAP Plot
    if shap_plot_bytes and isinstance(shap_plot_bytes, BytesIO): # Ensure it's a BytesIO object
        try:
            story.append(Paragraph("<b>SHAP Explanation Plot:</b>", styles['h2']))
            # Rewind the BytesIO buffer to the beginning before reading it
            shap_plot_bytes.seek(0)
            img = Image(shap_plot_bytes)
            # Ensure the image fits the page width while maintaining aspect ratio
            img_width = 6 * inch
            img_height = img.drawHeight * img_width / img.drawWidth
            img.drawWidth = img_width
            img.drawHeight = img_height
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            st.error(f"Error adding SHAP plot to PDF: {e}")
            story.append(Paragraph(f"<i>Error loading SHAP plot: {e}</i>", styles['Normal']))
    elif shap_plot_bytes: # If it's not BytesIO but something else
        st.warning("SHAP plot data is not in expected BytesIO format for PDF.")


    try:
        doc.build(story)
        buffer.seek(0) # Rewind the PDF buffer to the beginning
        return buffer
    except Exception as e:
        st.error(f"Error building PDF document: {e}")
        return None


# --- Individual Prediction Logic ---
if st.button("Predict Individual Charges"):
    # Reset has_predicted flag to False at the start of a new prediction cycle
    st.session_state.individual_prediction_results['has_predicted'] = False
    st.session_state.individual_prediction_results['shap_plot_buffer'] = None # Clear previous plot

    log_gb = gb.predict(new_X)
    log_xgb = xgb.predict(new_X)
    log_lgb = lgb.predict(new_X)

    current_prediction = 0
    current_shap_explanation = None
    current_model_type = ind_sel.lower()
    current_upper_bound = None
    shap_plot_buffer = None
    ai_explanation_text = None

    input_details_for_report = new_person.copy()
    input_details_for_report['sex'] = sex
    input_details_for_report['smoker'] = smoker
    input_details_for_report['region'] = region
    # Remove encoded regions from the report details as they are derived
    del input_details_for_report['region_northwest']
    del input_details_for_report['region_southeast']
    del input_details_for_report['region_southwest']


    if ind_sel == "Gradient Boosting":
        gb_prediction = np.exp(log_gb)[0]
        current_prediction = gb_prediction
        st.write(f"Estimated Insurance Charges: **${gb_prediction:.2f}**")
        if ind_upperbound:
            with st.spinner("Calculating Upperbound..."):
                bootstrap_ub_gb = get_bootstrap_prediction_upper_bound('gb', new_X)
                current_upper_bound = bootstrap_ub_gb
                st.write(f"Upperbound Estimated: **${bootstrap_ub_gb:.2f}**")
        if ind_show_shap:
            current_shap_explanation = explainer_gb(new_X)[0]
            st.subheader("SHAP Explanation for Gradient Boosting")
            fig, ax = plt.subplots(figsize=(10, 6))
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(current_shap_explanation, show=False)
            st.pyplot(fig) # Display the plot in Streamlit
            shap_plot_buffer = BytesIO()
            fig.savefig(shap_plot_buffer, format="png", bbox_inches="tight")
            # No need to seek(0) here yet, as st.pyplot consumed it. But BytesIO should be fine.
            plt.close(fig) # Close the figure to free memory

    elif ind_sel == "XGBoost":
        xgb_prediction = np.exp(log_xgb)[0]
        current_prediction = xgb_prediction
        st.write(f"Estimated Insurance Charges: **${xgb_prediction:.2f}**")
        if ind_upperbound:
            with st.spinner("Calculating Upperbound..."):
                bootstrap_ub_xgb = get_bootstrap_prediction_upper_bound('xgb', new_X)
                current_upper_bound = bootstrap_ub_xgb
                st.write(f"Upperbound Estimated: **${bootstrap_ub_xgb:.2f}**")
        if ind_show_shap:
            current_shap_explanation = explainer_xgb(new_X)[0]
            st.subheader("SHAP Explanation for XGBoost")
            fig, ax = plt.subplots(figsize=(10, 6))
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(current_shap_explanation, show=False)
            st.pyplot(fig)
            shap_plot_buffer = BytesIO()
            fig.savefig(shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    elif ind_sel == "LGBRegressor":
        lgb_prediction = np.exp(log_lgb)[0]
        current_prediction = lgb_prediction
        st.write(f"Estimated Insurance Charges: **${lgb_prediction:.2f}**")
        if ind_upperbound:
            with st.spinner("Calculating Upperbound..."):
                bootstrap_ub_lgb = get_bootstrap_prediction_upper_bound('lgb', new_X)
                current_upper_bound = bootstrap_ub_lgb
                st.write(f"Upperbound Estimated: **${bootstrap_ub_lgb:.2f}**")
        if ind_show_shap:
            current_shap_explanation = explainer_lgb(new_X)[0]
            st.subheader("SHAP Explanation for LGBRegressor")
            fig, ax = plt.subplots(figsize=(10, 6))
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(current_shap_explanation, show=False)
            st.pyplot(fig)
            shap_plot_buffer = BytesIO()
            fig.savefig(shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    elif ind_sel == "Ensemble":
        log_ensemble = w_gb * log_gb + w_xgb * log_xgb + w_lgb * log_lgb
        ensemble_prediction = np.exp(log_ensemble)[0]
        current_prediction = ensemble_prediction
        st.write(f"Estimated Insurance Charges: **${ensemble_prediction:.2f}**")
        if ind_show_shap:
            # Need to compute shap values for each model first for ensemble
            shap_values_gb_single = explainer_gb(new_X).values[0]
            shap_values_xgb_single = explainer_xgb(new_X).values[0]
            shap_values_lgb_single = explainer_lgb(new_X).values[0]

            shap_values_ensemble = w_gb * shap_values_gb_single + w_xgb * shap_values_xgb_single + w_lgb * shap_values_lgb_single
            expected_value_ensemble = w_gb * explainer_gb.expected_value + w_xgb * explainer_xgb.expected_value + w_lgb * explainer_lgb.expected_value
            shap_exp_ensemble = shap.Explanation(values=shap_values_ensemble, base_values=expected_value_ensemble, data=new_X.iloc[0].values, feature_names=new_X.columns.tolist())
            current_shap_explanation = shap_exp_ensemble
            st.subheader("SHAP Explanation for Ensemble")
            fig, ax = plt.subplots(figsize=(10, 6))
            if ind_sel_shap == "Waterfall":
                shap.plots.waterfall(shap_exp_ensemble, show=False)
            st.pyplot(fig)
            shap_plot_buffer = BytesIO()
            fig.savefig(shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    if ind_show_ai:
        if current_shap_explanation is not None:
            st.subheader("🤖 AI-Powered Explanation")
            with st.spinner("Generating AI explanation..."):
                input_values = [age, sex_encoded, bmi, children, smoker_encoded, region_northwest, region_southeast, region_southwest]
                ai_explanation_text = ai_agent.generate_shap_explanation(
                    shap_explanation=current_shap_explanation,
                    feature_names=feature_columns,
                    input_values=input_values,
                    prediction=current_prediction,
                    model_type=current_model_type # Use the determined model type
                )
                st.markdown(ai_explanation_text)
        else:
            st.warning("⚠️ SHAP values are required for AI explanations. Please select 'Show SHAP values' and re-run prediction.")

    # Store all results in session state
    st.session_state.individual_prediction_results['prediction_value'] = current_prediction
    st.session_state.individual_prediction_results['upper_bound_value'] = current_upper_bound
    st.session_state.individual_prediction_results['shap_plot_buffer'] = shap_plot_buffer
    st.session_state.individual_prediction_results['ai_explanation_text'] = ai_explanation_text
    st.session_state.individual_prediction_results['model_type'] = current_model_type
    st.session_state.individual_prediction_results['input_details'] = input_details_for_report
    st.session_state.individual_prediction_results['has_predicted'] = True # Set flag to True
    st.rerun() # Force a rerun to immediately display the results and download button


# --- Display Individual Prediction Results and Download Button (always runs if has_predicted is True) ---
# This block is *outside* the `if st.button("Predict Individual Charges"):` block
if st.session_state.individual_prediction_results['has_predicted']:
    st.subheader("Individual Prediction Report Overview") # Renamed for clarity

    # Display results from session state
    st.write(f"**Model Used:** {st.session_state.individual_prediction_results['model_type'].capitalize()}")
    st.write(f"**Predicted Charges:** ${st.session_state.individual_prediction_results['prediction_value']:.2f}")

    if st.session_state.individual_prediction_results['upper_bound_value'] is not None:
        st.write(f"**Upperbound Estimated:** ${st.session_state.individual_prediction_results['upper_bound_value']:.2f}")

    if st.session_state.individual_prediction_results['input_details']:
        st.write("---") # Separator
        st.write("#### Input Details for this Prediction:")
        for key, value in st.session_state.individual_prediction_results['input_details'].items():
            st.write(f"- **{key}:** {value}")
        st.write("---")

    if st.session_state.individual_prediction_results['ai_explanation_text']:
        st.write("#### AI-Powered Explanation Summary:")
        # Display full AI explanation
        st.markdown(st.session_state.individual_prediction_results['ai_explanation_text'])

    if st.session_state.individual_prediction_results['shap_plot_buffer']:
        st.write("#### SHAP Explanation Plot:")
        # To display the image from the buffer directly in Streamlit:
        # Important: st.image expects the buffer's content. .getvalue() provides this.
        # It's good practice to seek(0) the buffer before passing it if it was written to earlier
        # and its cursor might be at the end.
        st.session_state.individual_prediction_results['shap_plot_buffer'].seek(0)
        st.image(st.session_state.individual_prediction_results['shap_plot_buffer'].getvalue(), caption="SHAP Waterfall Plot")
        st.session_state.individual_prediction_results['shap_plot_buffer'].seek(0) # Rewind again for PDF generation

    st.write("---") # Separator for download button

    # Generate PDF buffer for download. This happens on *every* rerun if has_predicted is True.
    # st.download_button will then read this buffer when clicked.
    pdf_buffer_for_download = create_pdf_report(
        prediction_type="Individual Prediction",
        model_used=st.session_state.individual_prediction_results['model_type'],
        prediction_value=st.session_state.individual_prediction_results['prediction_value'],
        shap_plot_bytes=st.session_state.individual_prediction_results['shap_plot_buffer'], # Pass BytesIO object
        ai_explanation=st.session_state.individual_prediction_results['ai_explanation_text'],
        upper_bound_value=st.session_state.individual_prediction_results['upper_bound_value'],
        input_details=st.session_state.individual_prediction_results['input_details']
    )

    if pdf_buffer_for_download:
        st.download_button(
            label="Download Individual Prediction Report (PDF)",
            data=pdf_buffer_for_download, # Pass the BytesIO object directly
            file_name=f"Insurance_Individual_Report_{st.session_state.individual_prediction_results['model_type']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_individual_pdf_button" # Unique key for this specific button
        )
    else:
        st.error("Failed to generate PDF for download. Data might be missing or corrupted. Check console.")


# Total claim cost prediction
st.header("Total Claims Prediction")
tot_sel = st.selectbox("Selection of ML models", options=["Gradient Boosting", "XGBoost", "LGBRegressor", "Ensemble"], key="tot_sel")
tot_show_shap = st.checkbox("Show SHAP values", key="tot_show_shap")
if tot_show_shap:
    if tot_sel == "Ensemble":
        tot_sel_shap = st.selectbox("Selection of total SHAP graphs", options=["Beeswarm", "Bar"], key="tot_sel_shap_ensemble")
    else:
        tot_sel_shap = st.selectbox("Selection of total SHAP graphs", options=["Beeswarm", "Bar", "Scatter"], key="tot_sel_shap")
        if tot_sel_shap == "Scatter":
            tot_sel_shap_scatter = st.selectbox("Selection of feature for Scatter plot", options=["age", "sex", "bmi", "children", "smoker", "region_northwest", "region_southeast", "region_southwest"])

tot_show_ai = st.checkbox("🤖 Get AI Portfolio Analysis", key="tot_show_ai", help="Get AI-powered analysis of your entire portfolio's risk factors")

if st.button("Predict Total Claims"):
    st.session_state.total_prediction_results['has_predicted'] = False
    st.session_state.total_prediction_results['shap_plot_buffer'] = None

    log_gb_total = gb.predict(X)
    log_xgb_total = xgb.predict(X)
    log_lgb_total = lgb.predict(X)

    current_total_prediction = 0
    current_total_shap_explanation = None
    total_shap_plot_buffer = None
    total_ai_explanation_text = None
    current_model_type_total = tot_sel.lower()

    if tot_sel == "Gradient Boosting":
        gb_prediction_total = np.exp(log_gb_total)
        current_total_prediction = sum(gb_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            current_total_shap_explanation = explainer_gb(X)
            st.subheader("SHAP Explanation for Gradient Boosting (Total)")
            fig, ax = plt.subplots(figsize=(10, 6))
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(current_total_shap_explanation, show=False)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(current_total_shap_explanation, show=False)
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(current_total_shap_explanation[:, tot_sel_shap_scatter], show=False)
            st.pyplot(fig)
            total_shap_plot_buffer = BytesIO()
            fig.savefig(total_shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    elif tot_sel == "XGBoost":
        xgb_prediction_total = np.exp(log_xgb_total)
        current_total_prediction = sum(xgb_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            current_total_shap_explanation = explainer_xgb(X)
            st.subheader("SHAP Explanation for XGBoosting (Total)")
            fig, ax = plt.subplots(figsize=(10, 6))
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(current_total_shap_explanation, show=False)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(current_total_shap_explanation, show=False)
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(current_total_shap_explanation[:, tot_sel_shap_scatter], show=False)
            st.pyplot(fig)
            total_shap_plot_buffer = BytesIO()
            fig.savefig(total_shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    elif tot_sel == "LGBRegressor":
        lgb_prediction_total = np.exp(log_lgb_total)
        current_total_prediction = sum(lgb_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            current_total_shap_explanation = explainer_lgb(X)
            st.subheader("SHAP Explanation for LGBRegressor (Total)")
            fig, ax = plt.subplots(figsize=(10, 6))
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(current_total_shap_explanation, show=False)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(current_total_shap_explanation, show=False)
            elif tot_sel_shap == "Scatter":
                shap.plots.scatter(current_total_shap_explanation[:, tot_sel_shap_scatter], show=False)
            st.pyplot(fig)
            total_shap_plot_buffer = BytesIO()
            fig.savefig(total_shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    elif tot_sel == "Ensemble":
        log_ensemble_total = w_gb * log_gb_total + w_xgb * log_xgb_total + w_lgb * log_lgb_total
        ensemble_prediction_total = np.exp(log_ensemble_total)
        current_total_prediction = sum(ensemble_prediction_total)
        st.write(f"Total claim cost prediction: **${current_total_prediction:.2f}**")
        if tot_show_shap:
            # Need to compute shap values for each model first for ensemble
            shap_values_gb_total = explainer_gb(X).values
            shap_values_xgb_total = explainer_xgb(X).values
            shap_values_lgb_total = explainer_lgb(X).values

            shap_values_ensemble_total = w_gb * shap_values_gb_total + w_xgb * shap_values_xgb_total + w_lgb * shap_values_lgb_total
            expected_value_ensemble_total = w_gb * explainer_gb.expected_value + w_xgb * explainer_xgb.expected_value + w_lgb * explainer_lgb.expected_value
            shap_exp_ensemble_total = shap.Explanation(values=shap_values_ensemble_total, base_values=expected_value_ensemble_total, data=X.values, feature_names=X.columns.tolist())
            current_total_shap_explanation = shap_exp_ensemble_total
            st.subheader("SHAP Explanation for Ensemble (Total)")
            fig, ax = plt.subplots(figsize=(10, 6))
            if tot_sel_shap == "Beeswarm":
                shap.plots.beeswarm(shap_exp_ensemble_total, show=False)
            elif tot_sel_shap == "Bar":
                shap.plots.bar(shap_exp_ensemble_total, show=False)
            st.pyplot(fig)
            total_shap_plot_buffer = BytesIO()
            fig.savefig(total_shap_plot_buffer, format="png", bbox_inches="tight")
            plt.close(fig)

    if tot_show_ai:
        if current_total_shap_explanation is not None:
            st.subheader("🤖 AI-Powered Portfolio Analysis")
            with st.spinner("Generating AI portfolio analysis..."):
                total_ai_explanation_text = ai_agent.generate_total_shap_explanation(
                    shap_explanation=current_total_shap_explanation,
                    feature_names=feature_columns,
                    total_prediction=current_total_prediction,
                    model_type=current_model_type_total,
                    graph=tot_sel_shap
                )
                st.markdown(total_ai_explanation_text)
        else:
            st.warning("⚠️ SHAP values are required for AI portfolio analysis. Please select 'Show SHAP values' and re-run prediction.")

    # Store all results in session state
    st.session_state.total_prediction_results['total_prediction_value'] = current_total_prediction
    st.session_state.total_prediction_results['shap_plot_buffer'] = total_shap_plot_buffer
    st.session_state.total_prediction_results['ai_explanation_text'] = total_ai_explanation_text
    st.session_state.total_prediction_results['model_type'] = current_model_type_total
    st.session_state.total_prediction_results['has_predicted'] = True
    st.rerun() # Force a rerun to immediately display the results and download button


# --- Display Total Prediction Results and Download Button (always runs if has_predicted is True) ---
# This block is *outside* the `if st.button("Predict Total Claims"):` block
if st.session_state.total_prediction_results['has_predicted']:
    st.subheader("Total Claims Report Overview")

    # Display results from session state
    st.write(f"**Model Used:** {st.session_state.total_prediction_results['model_type'].capitalize()}")
    st.write(f"**Total Predicted Claims:** ${st.session_state.total_prediction_results['total_prediction_value']:,.2f}")

    if st.session_state.total_prediction_results['ai_explanation_text']:
        st.write("#### AI-Powered Portfolio Analysis Summary:")
        st.markdown(st.session_state.total_prediction_results['ai_explanation_text'])

    if st.session_state.total_prediction_results['shap_plot_buffer']:
        st.write("#### SHAP Explanation Plot:")
        st.session_state.total_prediction_results['shap_plot_buffer'].seek(0)
        st.image(st.session_state.total_prediction_results['shap_plot_buffer'].getvalue(), caption="SHAP Plot")
        st.session_state.total_prediction_results['shap_plot_buffer'].seek(0)

    st.write("---")

    pdf_buffer_for_download_total = create_pdf_report(
        prediction_type="Total Claims Prediction",
        model_used=st.session_state.total_prediction_results['model_type'],
        total_prediction_value=st.session_state.total_prediction_results['total_prediction_value'],
        shap_plot_bytes=st.session_state.total_prediction_results['shap_plot_buffer'],
        ai_explanation=st.session_state.total_prediction_results['ai_explanation_text'],
    )
    if pdf_buffer_for_download_total:
        st.download_button(
            label="Download Total Claims Report (PDF)",
            data=pdf_buffer_for_download_total,
            file_name=f"Insurance_Total_Report_{st.session_state.total_prediction_results['model_type']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_total_pdf_button"
        )
    else:
        st.error("Failed to generate PDF for download. Data might be missing or corrupted.")
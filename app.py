import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import glob
import json

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

# Set page config with custom icon
st.set_page_config(
    page_title="Model prediction visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Model prediction visualization")
st.markdown("""
    This tool uses feature data to make predictions and provides mechanistic explanations through SHAP visualization.
    Adjust the feature value in the sidebar to observe the changes in the prediction results and SHAP values.
""")

# Find the Excel file in the data folder
def get_excel_file():
    excel_files = glob.glob('data/*.xlsx')
    if len(excel_files) != 1:
        st.error("Expected exactly one Excel file in the 'data' folder.")
        return None
    return excel_files[0]

# Find the JSON file in the data folder
def get_json_file():
    json_files = glob.glob('data/*.json')
    if len(json_files) != 1:
        st.error("Expected exactly one JSON file in the 'data' folder.")
        return None
    return json_files[0]

# Load normalization methods from JSON
@st.cache_data
def load_norm_methods():
    json_file = get_json_file()
    if json_file is None:
        return 'none', 'none'
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
        input_norm = config.get('net_property', {}).get('input_norm', 'none')
        output_norm = config.get('net_property', {}).get('output_norm', input_norm)  # Default to input_norm if not specified
        if input_norm not in ['ext', 'avg', 'avgext', 'avgstd', 'none']:
            st.error(f"Invalid input normalization method in JSON: {input_norm}. Using 'none'.")
            input_norm = 'none'
        if output_norm not in ['ext', 'avg', 'avgext', 'avgstd', 'none']:
            st.error(f"Invalid output normalization method in JSON: {output_norm}. Using 'none'.")
            output_norm = 'none'
        return input_norm, output_norm
    except Exception as e:
        st.error(f"Error reading JSON file: {str(e)}. Using 'none' for both input and output normalization.")
        return 'none', 'none'

# Load and prepare background data and normalization parameters
@st.cache_data
def load_background_data():
    excel_file = get_excel_file()
    if excel_file is None:
        return None, None, None
    df = pd.read_excel(excel_file)
    features = df.iloc[:, :-2]  # Exclude the last column (target)
    target = df.iloc[:, -1]  # Target column
    feature_params = {
        'mean': features.mean(),
        'max': features.max(),
        'std': features.std()
    }
    target_params = {
        'mean': target.mean(),
        'max': target.max(),
        'std': target.std()
    }
    return features, feature_params, target_params

# Determine model type and class labels
@st.cache_data
def determine_model_type():
    excel_file = get_excel_file()
    if excel_file is None:
        return None, None
    df = pd.read_excel(excel_file)
    target_col = df.iloc[:, -1]  # Get the last column (target)
    unique_values = target_col[1:].nunique()  # Exclude header, count unique values
    if unique_values == 2:
        model_type = "classification"
        labels = list(target_col[1:].unique())  # Get the two unique labels
        return model_type, labels
    else:
        model_type = "regression"
        return model_type, None

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/MODEL.h5')

# Normalization operations
norm_op_dict = {
    'ext': lambda data, param: data / param.get('max', 1),
    'avg': lambda data, param: data - param.get('mean', 0),
    'avgext': lambda data, param: (data - param.get('mean', 0)) / param.get('max', 1),
    'avgstd': lambda data, param: (data - param.get('mean', 0)) / param.get('std', 1),
    'none': lambda data, param: data
}

# Denormalization operations
denorm_op_dict = {
    'ext': lambda data, param: data * param.get('max', 1),
    'avg': lambda data, param: data + param.get('mean', 0),
    'avgext': lambda data, param: data * param.get('max', 1) + param.get('mean', 0),
    'avgstd': lambda data, param: data * param.get('std', 1) + param.get('mean', 0),
    'none': lambda data, param: data
}

# Initialize data and model
background_data, feature_params, target_params = load_background_data()
if background_data is None:
    st.stop()

model = load_model()
input_norm, output_norm = load_norm_methods()

# Default values for features (use raw, non-normalized values)
default_values = background_data.iloc[0, :].to_dict()

# Determine model type
model_type, class_labels = determine_model_type()
if model_type is None:
    st.stop()

# Sidebar configuration
st.sidebar.header("Feature Inputs")
st.sidebar.markdown("Adjust values of features:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

features = list(default_values.keys())
values = {}
cols = st.sidebar.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        values[feature] = st.number_input(
            feature,
            min_value=float(background_data[feature].min()),
            max_value=float(background_data[feature].max()),
            value=default_values[feature],
            step=0.001,
            format="%.3f",
            key=feature
        )

# Prepare input data with normalization
def prepare_input_data():
    input_df = pd.DataFrame([values])
    norm_func = norm_op_dict.get(input_norm, norm_op_dict['none'])
    normalized_data = input_df.copy()
    for col in input_df.columns:
        normalized_data[col] = norm_func(input_df[col], {
            'mean': feature_params['mean'].get(col, 0),
            'max': feature_params['max'].get(col, 1),
            'std': feature_params['std'].get(col, 1)
        })
    if input_norm == 'none':
        st.warning("Input data is not normalized. Ensure the model was trained on non-normalized data, or prediction may be incorrect.")
    return normalized_data, input_df

# Main analysis
if st.button("Analyze Calculation", key="calculate"):
    normalized_input_df, original_input_df = prepare_input_data()

    # Prediction
    prediction = model.predict(normalized_input_df.values, verbose=0)[0][0]

    # Denormalize prediction for regression models
    if model_type == "regression":
        denorm_func = denorm_op_dict.get(output_norm, denorm_op_dict['none'])
        display_prediction = denorm_func(prediction, target_params)
    else:
        display_prediction = prediction

    with st.container():
        st.header("📈 Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            if model_type == "classification":
                predicted_class = class_labels[1] if prediction >= 0.5 else class_labels[0]
                st.metric(
                    "Probability",
                    f"{prediction:.4f}",
                    delta=f"Predicted Class: {predicted_class}",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "Predicted Value",
                    f"{display_prediction:.4f}"
                )
        with col2:
            if model_type == "classification":
                st.metric(
                    "Classification Threshold",
                    f"0.5"
                )
            else:
                excel_file = get_excel_file()
                df_y = pd.read_excel(excel_file).iloc[:, -1]
                st.metric(
                    "Prediction Range",
                    f"{df_y.min():.2f} - {df_y.max():.2f}"
                )

    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(normalized_input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Force Plot", "Decision Plot", "Mechanistic Insights"])

    with tab1:
        st.subheader("Force Plot")
        col1, col2 = st.columns([3, 1])
        with col1:
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                feature_names=original_input_df.columns,
                data=original_input_df.values.round(3)
            )
            shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab2:
        st.subheader("Decision Plot")
        col1, col2 = st.columns([2, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            shap.decision_plot(base_value, shap_values, original_input_df.columns, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

    with tab3:
        st.subheader("Mechanistic Insights")
        importance_df = pd.DataFrame({'Feature': original_input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))
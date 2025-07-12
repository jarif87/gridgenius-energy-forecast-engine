import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from tensorflow.keras.models import load_model
import os

# Streamlit page configuration
st.set_page_config(
    page_title="Power Consumption Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for white background and black text/borders
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .main {background-color: #ffffff;}
    .stTitle {font-family: 'Roboto', sans-serif; text-align: center; margin-bottom: 10px; font-size: 32px; font-weight: 700;}
    .stSubheader {font-family: 'Roboto', sans-serif; font-size: 22px; font-weight: 700; margin-top: 10px; margin-bottom: 10px;}
    .stMarkdown {font-family: 'Roboto', sans-serif; font-size: 16px;}
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #000000;
    }
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 18px;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        display: block;
        margin: 15px auto;
        border: 2px solid #000000;
    }
    .stButton>button:hover {
        background-color: #f0f0f0;
    }
    .stNumberInput label {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        font-size: 16px;
    }
    .stNumberInput input {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 10px;
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
    }
    .stNumberInput input:focus {
        outline: none;
        border: 2px solid #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Robust file loading
try:
    # Try root directory first (Hugging Face Spaces working directory)
    model_path = 'my_model.keras'
    scaler_x_path = 'scaler_X.pkl'
    scaler_y_path = 'scaler_y.pkl'

    # Fallback: Try relative to src/ (if files are in src/)
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), 'my_model.keras')
    if not os.path.exists(scaler_x_path):
        scaler_x_path = os.path.join(os.path.dirname(__file__), 'scaler_X.pkl')
    if not os.path.exists(scaler_y_path):
        scaler_y_path = os.path.join(os.path.dirname(__file__), 'scaler_y.pkl')

    # Alternative: If files are in a 'models/' folder (uncomment if applicable)
    # model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'my_model.keras')
    # scaler_x_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler_X.pkl')
    # scaler_y_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler_y.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure 'my_model.keras' is in the Space's root or src directory.")
    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Scaler X file not found at {scaler_x_path}. Ensure 'scaler_X.pkl' is in the Space's root or src directory.")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler Y file not found at {scaler_y_path}. Ensure 'scaler_y.pkl' is in the Space's root or src directory.")

    model = load_model(model_path)
    scaler_X = pickle.load(open(scaler_x_path, 'rb'))
    scaler_y = pickle.load(open(scaler_y_path, 'rb'))
except Exception as e:
    st.error(f"Failed to load model or scalers: {str(e)}. Ensure 'my_model.keras', 'scaler_X.pkl', and 'scaler_y.pkl' are in the Space's root or src directory. "
             "If using TensorFlow 2.18.0, try resaving the model with TensorFlow 2.18.0.")
    st.stop()

# Main app layout
st.title("Power Consumption Predictor")
st.markdown("""
    Enter values for one timestep to predict power consumption for Zone1, Zone2, and Zone3.  
    Results will be displayed as a bar plot and a table.
""")

# Input section
st.subheader("Enter Timestep Data")
st.markdown("""
    **Instructions**:
    - Enter values for the 8 features below (default values are provided).
    - **Hour**: 0 to 23 (e.g., 14 for 2 PM).
    - **DayOfWeek**: 0 to 6 (0 = Monday, 6 = Sunday).
    - **Month**: 1 to 12 (e.g., 7 for July).
    - **Other features**: Use reasonable values (e.g., Temperature in °C, Humidity as a fraction).
    - Click "Predict" to see results.
""")

# Vertical form for input
with st.container():
    feature_names = ['Hour', 'DayOfWeek', 'Month', 'Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
    default_values = [0, 6, 1, 6.559, 73.8, 0.083, 0.051, 0.119]  # From dataset
    user_input = []
    for i, (name, default) in enumerate(zip(feature_names, default_values)):
        if name in ['Hour', 'DayOfWeek', 'Month']:
            value = st.number_input(
                f"{name}",
                min_value=0,
                max_value=23 if name == 'Hour' else 6 if name == 'DayOfWeek' else 12,
                value=int(default),
                step=1,
                key=f"input_{i}"
            )
            user_input.append(value)
        else:
            value = st.number_input(
                f"{name}",
                value=float(default),
                step=0.01,
                format="%.6f",
                key=f"input_{i}"
            )
            user_input.append(value)

# Predict button
if st.button("Predict", key="predict_button"):
    try:
        # Replicate input for 24 timesteps
        custom_raw_data = np.array([user_input] * 24).reshape(1, 24, 8)
        
        # Selective scaling
        features_to_scale = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
        scale_indices = [3, 4, 5, 6, 7]
        custom_scaled = custom_raw_data.copy()
        custom_2d_to_scale = custom_raw_data[:, :, scale_indices].reshape(-1, len(scale_indices))
        custom_scaled_2d = scaler_X.transform(custom_2d_to_scale)
        custom_scaled[:, :, scale_indices] = custom_scaled_2d.reshape(1, 24, len(scale_indices))

        # Predict
        y_pred_scaled = model.predict(custom_scaled)
        if isinstance(y_pred_scaled, list):
            y_pred_combined = np.concatenate(y_pred_scaled, axis=1)
        else:
            y_pred_combined = y_pred_scaled
        y_pred_original = scaler_y.inverse_transform(y_pred_combined)

        # Store predictions
        labels = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
        st.session_state.pred_df = pd.DataFrame(y_pred_original, columns=labels, index=['User Input'])
        st.session_state.predictions = y_pred_original

    except Exception as e:
        st.error(f"Error processing input: {str(e)}")

# Display predictions if available
if 'predictions' in st.session_state and st.session_state.predictions is not None:
    st.markdown("### Predicted Power Consumption")
    fig = px.bar(
        st.session_state.pred_df.reset_index().melt(id_vars='index', value_vars=labels, var_name='Zone', value_name='Power Consumption'),
        x='index', y='Power Consumption', color='Zone', barmode='group',
        title='Predicted Power Consumption by Zone',
        labels={'index': 'Sample', 'Power Consumption': 'Power Consumption (Original Scale)'}
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Roboto', size=12),
        title_font=dict(size=18, family='Roboto'),
        xaxis_title="Sample",
        yaxis_title="Power Consumption (Original Scale)",
        legend_title="Zones",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prediction Table")
    st.dataframe(st.session_state.pred_df.style.format("{:.4f}").set_caption("Predicted Power Consumption (Original Scale)"))

# Footer
st.markdown("---")
st.markdown("**Made by Sadik Al Jarif**")
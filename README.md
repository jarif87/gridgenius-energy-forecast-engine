###  GridGenius Energy Forecast Engine
GridGenius Energy Forecast Engine is a Streamlit-based web application for predicting power consumption across three zones (Zone1, Zone2, Zone3) using a neural network model. This is a multilabel project, where the model simultaneously predicts power consumption for all three zones based on user-provided inputs. The app accepts inputs for 8 features (Hour, DayOfWeek, Month, Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows) and displays predictions as a Plotly bar plot and a table.

Developed by Sadik Al Jarif, the app features a high-contrast, minimalist design with a white background, black text, and black borders for maximum visibility. It is hosted on GitHub.

![](/images/image.png)

### Features
- Input Form: Vertical form for 8 features with default values:
    - Hour: 0–23 (default: 0)
    - DayOfWeek: 0–6 (0 = Monday, default: 6)
    - Month: 1–12 (default: 1)
    - Temperature: °C (default: 6.559)
    - Humidity: % (default: 73.8)
    - WindSpeed: m/s (default: 0.083)
    - GeneralDiffuseFlows: W/m² (default: 0.051)
    - DiffuseFlows: W/m² (default: 0.119)

- Prediction: Uses a pre-trained Keras model (my_model.keras) to predict power consumption for Zone1, Zone2, and Zone3 in a multilabel setup, outputting three values simultaneously.
- Visualization: Displays results as a Plotly bar plot and a table with predicted values.
- Design: White background, black text and borders, Roboto font, and a centered "Predict" button for a clean, user-friendly interface. The app footer credits "Made by Sadik Al Jarif".

### Setup

- Prerequisites
    - Python 3.9
    - GitHub account (for cloning the repository)
    - Git (optional, for local management)


### Local Setup
1. **Clone or download the repository from GitHub:**
    - Clone: git clone https://github.com/jarif87/gridgenius-energy-forecast-engine.git
    - Alternatively, download the ZIP file from the repository page and extract it.

2. **Navigate to the project folder (e.g., /path/to/gridgenius-energy-forecast-engine).**

3. **Create a virtual environment:**
    - Open a terminal or command prompt.
    - Run: python -m venv venv
    - Activate: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows).
4. **Install dependencies from requirements.txt:**
    - Ensure requirements.txt contains:
```
streamlit
numpy==1.26.4
tensorflow==2.18.0
plotly==5.24.1
pandas==2.2.3
keras==3.8.0
scikit-learn==1.2.2
```
- Install: pip install -r requirements.txt
5. **Verify model files (my_model.keras, scaler_X.pkl, scaler_y.pkl) are in the root directory.**

6. **Run the app:**
    - Execute: streamlit run src/streamlit_app.py
    - Open http://localhost:8501 in your browser.

### Repository Structure

```
/gridgenius-energy-forecast-engine/
├── images/
│   └── image.png    # Screenshot of the Streamlit app
├── notebooks/
│   ├                
│   └── GridGenius_Energy_Forecast_Engine.ipynb  # Notebook for model training
├── src/
│   └── streamlit_app.py             # Streamlit app code
├                  
├── my_model.keras                   # Pre-trained Keras model
├── scaler_X.pkl                     # Scaler for input features
├── scaler_y.pkl                     # Scaler for output predictions
├── requirements.txt                 # Python dependencies
                
```

### Usage

- Access the app locally at http://localhost:8501 after running streamlit run src/streamlit_app.py.
- Enter values for the 8 features in the vertical form (defaults provided).
- Click the centered "Predict" button to generate predictions.
- View results:
    - A Plotly bar plot showing power consumption for Zone1, Zone2, and Zone3.
    - A table with precise predicted values in the original scale

### Notebook

- The notebooks/GridGenius_Energy_Forecast_Engine.ipynb notebook includes:
- Data preprocessing (e.g., scaling features 3–7: Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows).
- Neural network model training and evaluation for multilabel prediction (Zone1, Zone2, Zone3).
- Saving the model (my_model.keras) and scalers (scaler_X.pkl, scaler_y.pkl).

### To run the notebook:
- Install Jupyter: pip install jupyter
- Open: jupyter notebook notebooks/GridGenius_Energy_Forecast_Engine.ipynb
- Follow the notebook instructions to train or explore the model.
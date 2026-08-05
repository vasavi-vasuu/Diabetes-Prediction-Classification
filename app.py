import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction")
st.write("Enter the patient's health details to predict diabetes risk.")

# ----------------------------------
# Load model and scaler - 8 features
# ----------------------------------
@st.cache_resource
def load_artifacts():
    # 1. Data load
    df = pd.read_csv('diabetes.csv')

    # 2. 8 features select chey
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    X = df[feature_names]
    y = df['Outcome']

    # 3. 8 features tho scaler train
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 8 features tho model train
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_artifacts()

# ----------------------------------
# Feature inputs - 8
# ----------------------------------
feature_inputs = {
    'Pregnancies': st.number_input('Pregnancies', min_value=0, max_value=20, value=1, key="preg"),
    'Glucose': st.number_input('Glucose', min_value=0, max_value=250, value=110, key="gluc"),
    'BloodPressure': st.number_input('Blood Pressure', min_value=0, max_value=150, value=70, key="bp"),
    'SkinThickness': st.number_input('Skin Thickness', min_value=0, max_value=100, value=20, key="skin"),
    'Insulin': st.number_input('Insulin', min_value=0, max_value=900, value=80, key="ins"),
    'BMI': st.number_input('BMI', min_value=0.0, max_value=70.0, value=28.0, key="bmi"),
    'DiabetesPedigreeFunction': st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.4, key="dpf"),
    'Age': st.number_input('Age', min_value=1, max_value=120, value=30, key="age"),
}

# Maintain correct feature order
input_values = list(feature_inputs.values())

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Diabetes Risk"):
    input_array = np.array(input_values).reshape(1, -1)

    # Scale input - ippudu 8 ki 8 match avthundi
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)
    proba = model.predict_proba(scaled_input)[0][1]

    if prediction[0] == 1:
        st.error(f"🩺 Predicted: **Diabetic** (probability: {proba:.1%})")
        st.warning("Please consult a healthcare professional for further evaluation.")
    else:
        st.success(f"🩺 Predicted: **Not Diabetic** (probability of diabetes: {proba:.1%})")
        st.balloons()

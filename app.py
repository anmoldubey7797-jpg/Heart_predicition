import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ------------------------------------------------
# Load trained model & scaler
# ------------------------------------------------
model = tf.keras.models.load_model('heart_disease_model.h5')
std = joblib.load('std.pkl')

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction ")
st.write("Enter the patient details below to predict the risk of heart disease.")

# ------------------------------------------------
# Input Fields
# ------------------------------------------------
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.slider("Resting ECG Results (0–2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.slider("Slope of Peak Exercise ST Segment (0–2)", 0, 2, 1)
ca = st.slider("Number of Major Vessels (0–4)", 0, 4, 0)
thal = st.slider("Thalassemia (1–3)", 1, 3, 2)

# ------------------------------------------------
# Convert categorical inputs to numeric
# ------------------------------------------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ------------------------------------------------
# Prepare data for model
# ------------------------------------------------
data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]],
                    columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                             'thalach','exang','oldpeak','slope','ca','thal'])

# ------------------------------------------------
# Predict
# ------------------------------------------------
if st.button("🔍 Predict"):
    scaled = std.transform(data)
    prob = model.predict(scaled)[0][0]
    result = "⚠️ High Risk of Heart Disease" if prob > 0.5 else "✅ Low Risk of Heart Disease"

    st.subheader("🩺 Prediction Result")
    st.write(result)
    st.write(f"**Probability:** {prob:.2f}")

    if prob > 0.5:
        st.error("Person likely has a high risk. Please consult a doctor immediately.")
    else:
        st.success("Person likely has low risk. Keep maintaining a healthy lifestyle!")

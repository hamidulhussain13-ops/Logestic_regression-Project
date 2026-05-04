import streamlit as st
import joblib
import numpy as np
import os

# load model (FIXED)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

st.title("📊 Logistic Regression Prediction App")

st.write("Enter values to predict:")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")

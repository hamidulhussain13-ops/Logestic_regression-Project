import streamlit as st
import numpy as np
import joblib
import os

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

st.title("📊 Logistic Regression Prediction App")

st.write("Enter values to predict:")

# Inputs
f1 = st.number_input("Feature 1", value=0.02)
f2 = st.number_input("Feature 2", value=0.02)
f3 = st.number_input("Feature 3", value=0.02)
f4 = st.number_input("Feature 4", value=0.02)

# Prediction button
if st.button("Predict"):

    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)

    # Map output
    if prediction[0] == 0:
        result = "Setosa"
    elif prediction[0] == 1:
        result = "Versicolor"
    else:
        result = "Virginica"

    st.success(f"Prediction: {result}")

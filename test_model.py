import joblib
import numpy as np

def test_model_load():
    model = joblib.load("model.pkl")
    assert model is not None

def test_prediction():
    model = joblib.load("model.pkl")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)
    assert len(prediction) == 1

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="Diabetes Risk Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
BASE = os.path.dirname(__file__)
model   = joblib.load(os.path.join(BASE, "model", "diabetes_model.pkl"))
scaler  = joblib.load(os.path.join(BASE, "model", "scaler.pkl"))
features = joblib.load(os.path.join(BASE, "model", "features.pkl"))


class PatientData(BaseModel):
    pregnancies: float = Field(..., ge=0, le=20,   description="Number of pregnancies")
    glucose:     float = Field(..., ge=0, le=300,  description="Plasma glucose concentration (mg/dL)")
    blood_pressure: float = Field(..., ge=0, le=180, description="Diastolic blood pressure (mmHg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin:     float = Field(..., ge=0, le=900,  description="2-Hour serum insulin (mu U/ml)")
    bmi:         float = Field(..., ge=0, le=80,   description="Body mass index (kg/m²)")
    diabetes_pedigree: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age:         float = Field(..., ge=1, le=120,  description="Age in years")


class PredictionResult(BaseModel):
    risk_percentage: float
    risk_level: str
    prediction: int
    top_factors: list[dict]
    advice: str


@app.get("/")
def root():
    return {"message": "Diabetes Risk Predictor API", "status": "running"}


@app.post("/predict", response_model=PredictionResult)
def predict(data: PatientData):
    input_array = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age,
    ]])

    input_scaled = scaler.transform(input_array)
    prediction   = int(model.predict(input_scaled)[0])
    probability  = float(model.predict_proba(input_scaled)[0][1])
    risk_pct     = round(probability * 100, 1)

    if risk_pct < 30:
        risk_level = "Low"
        advice = "Your indicators look healthy. Maintain a balanced diet and regular exercise."
    elif risk_pct < 60:
        risk_level = "Moderate"
        advice = "Some risk factors detected. Consider consulting a doctor and monitoring your glucose levels."
    else:
        risk_level = "High"
        advice = "High risk detected. Please consult a healthcare professional for a proper diagnosis."

    # Top contributing factors
    importances = model.feature_importances_
    feature_values = input_array[0]
    factors = sorted(
        [{"name": features[i], "importance": round(float(importances[i]) * 100, 1), "value": round(float(feature_values[i]), 1)}
         for i in range(len(features))],
        key=lambda x: x["importance"],
        reverse=True
    )[:4]

    return PredictionResult(
        risk_percentage=risk_pct,
        risk_level=risk_level,
        prediction=prediction,
        top_factors=factors,
        advice=advice,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

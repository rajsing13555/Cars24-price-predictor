"""
api.py — Production-style deployment layer
---------------------------------------------
Wraps the trained pipeline in a FastAPI service with request validation
(Pydantic), so the model can be called from any application (a web
front-end, a mobile app, or another internal service) rather than only
through the Streamlit demo.

Run with: uvicorn api.api:app --reload --port 8000
Docs auto-generated at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "best_model.pkl")

app = FastAPI(
    title="CARS24-style Used Car Price Prediction API",
    description="Predicts fair resale price for a used car based on its attributes.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


class CarFeatures(BaseModel):
    brand: Literal["Maruti Suzuki", "Hyundai", "Tata", "Honda", "Toyota",
                   "Mahindra", "Kia", "Ford", "Volkswagen", "Renault"]
    city: Literal["Delhi NCR", "Mumbai", "Bengaluru", "Pune", "Hyderabad",
                  "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Chandigarh"]
    car_age: int = Field(..., ge=0, le=25, description="Age of the car in years")
    km_driven: int = Field(..., ge=0, le=500000)
    fuel_type: Literal["Petrol", "Diesel", "CNG", "Electric"]
    transmission: Literal["Manual", "Automatic"]
    owner_count: int = Field(..., ge=1, le=6)
    mileage_kmpl: float = Field(..., ge=1, le=40)
    engine_cc: int = Field(..., ge=600, le=4000)
    insurance_valid: Literal["Yes", "No"]

    class Config:
        json_schema_extra = {
            "example": {
                "brand": "Hyundai", "city": "Bengaluru", "car_age": 4,
                "km_driven": 48000, "fuel_type": "Petrol", "transmission": "Manual",
                "owner_count": 1, "mileage_kmpl": 18.5, "engine_cc": 1197,
                "insurance_valid": "Yes",
            }
        }


class PredictionResponse(BaseModel):
    predicted_price_inr: float
    predicted_price_formatted: str


@app.get("/")
def root():
    return {"status": "ok", "message": "CARS24-style price prediction API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    km_per_year = car.km_driven / max(car.car_age, 1)

    input_df = pd.DataFrame([{
        "brand": car.brand, "city": car.city, "car_age": car.car_age,
        "km_driven": car.km_driven, "fuel_type": car.fuel_type,
        "transmission": car.transmission, "owner_count": car.owner_count,
        "mileage_kmpl": car.mileage_kmpl, "engine_cc": car.engine_cc,
        "insurance_valid": car.insurance_valid, "km_per_year": km_per_year,
    }])

    predicted_price = float(model.predict(input_df)[0])

    return PredictionResponse(
        predicted_price_inr=round(predicted_price, 2),
        predicted_price_formatted=f"₹{predicted_price:,.0f}",
    )

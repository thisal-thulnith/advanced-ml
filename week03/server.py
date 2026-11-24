from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mlr_mpg.joblib"

app = FastAPI(title="Linear Regression API", description="A simple API for linear regression model", version="1.0.0")

class Predictionrequest(BaseModel):
    displacement: float = Field(..., description="The displacement of the car engine in cubic inches")
    horsepower: float = Field(..., description="The horsepower of the car engine")
    weight: float = Field(..., description="The weight of the car in pounds")
    acceleration: float = Field(..., description="The acceleration of the car in seconds")
    model_year: int = Field(..., description="The model year of the car (e.g., 1970)")
    origin: Literal["usa","europe","japan"] = Field(..., description="The origin of the car (1 for USA, 2 for Europe, 3 for Japan)")


class PredictionResponse(BaseModel):
    mpg: float = Field(..., description="The predicted miles per gallon (MPG) for the car")

def load_model(model_path: Path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")



try:
    model = load_model(MODEL_PATH)
except HTTPException as e:
    print(e.detail)
    model = None

ORIGIN_MAP = {"usa": 1, "europe": 2, "japan": 3}


def build_inference_frame(request: Predictionrequest) -> pd.DataFrame:
    data = request.model_dump()
    # Match the column names/encodings the model was trained on
    year = data.pop("model_year")
    data["model year"] = year - 1900 if year >= 1900 else year
    data["origin"] = ORIGIN_MAP[data["origin"]]
    return pd.DataFrame([data])


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Predictionrequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    inference_frame = build_inference_frame(request)
    prediction = model.predict(inference_frame)[0]
    return PredictionResponse(mpg=float(prediction))

@app.get("/health", response_model=dict)
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy"}
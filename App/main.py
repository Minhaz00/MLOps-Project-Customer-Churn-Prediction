import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
MODEL_PATH = "https://poridhi-mlflow-models-123-d862805.s3.ap-southeast-1.amazonaws.com/model.pkl"
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception(f"Model not found at {MODEL_PATH}")

# Define the request schema
class CustomerFeatures(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Bank_transfer_automatic: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# Initialize FastAPI app
app = FastAPI()

# Load scaler to match preprocessing
scaler = MinMaxScaler()
large_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Preprocess input data
def preprocess_input(data: List[CustomerFeatures]):
    df = pd.DataFrame([item.dict() for item in data])
    df[large_cols] = scaler.fit_transform(df[large_cols])
    return df.values

# Define the prediction endpoint
@app.post("/predict")
def predict(data: List[CustomerFeatures]):
    try:
        input_data = preprocess_input(data)
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

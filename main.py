from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  INPUT SCHEMA (THIS WAS MISSING)
class InputData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Load everything
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_data = [[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]]

        input_df = pd.DataFrame(
            input_data,
            columns=['N','P','K','temperature','humidity','ph','rainfall']
        )

        input_scaled = scaler.transform(input_df)
        input_imputed = imputer.transform(input_scaled)

        prediction = model.predict(input_imputed)
        result = encoder.inverse_transform(prediction)

        return {"prediction": result.tolist()}

    except Exception as e:
        return {"error": str(e)}
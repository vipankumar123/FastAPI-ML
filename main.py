from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#pip install pandas
import pandas as pd
import pickle


app = FastAPI()

class HousePrediction(BaseModel):
    num_rooms : int
    square_footage : int
    age_of_house : int


with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get("/")
def read_root():
    return {"message": "Welcome to the House prediction API"}


@app.post("/predict/")
def predict_price(features : HousePrediction):
    try:
        print(features)
        data = pd.DataFrame([features.dict()])
        prediction = model.predict(data)
        result = prediction[0]
        return {"prediction_price": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
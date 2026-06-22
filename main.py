import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(9, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )
  def forward(self, x):
      return self.network(x)

app = FastAPI()
device = torch.device("cpu")
model = MLP()
model.load_state_dict(torch.load("MLP_model.pth"))
model.eval()
scalar = joblib.load("scalar.joblib")

class Prediction_Request(BaseModel):
   Hours_Studied: float
   Attendance: float
   Parental_Involvement: float  
   Access_to_Resources: float
   Extracurricular_Activities
   Sleep_Hours
   Previous_Scores
   Parental_Education_Level
   Motivation_Level

@app.post("/predict")
def predict_data(data: Prediction_Request):
   try:
      raw_features = [[data.feature1, data.feature2, data.feature3]]
      scaled_features = scalar.transform(raw_features)
      tensored_scaled_data = torch.tensor(scaled_features, dtype=torch.float32).to(device)
      with torch.no_grad():
         raw_prediction = model(tensored_scaled_data)
         prediction_value = raw_prediction.item()
      return {
         "prediction_value": round(prediction_value, 4),
        "status": "SUCCESS"
      }
   except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))


